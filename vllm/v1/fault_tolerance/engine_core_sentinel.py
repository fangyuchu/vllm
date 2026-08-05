# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCoreSentinel and fault_tolerant_wrapper for the engine core."""

import json
import threading
from collections.abc import Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import msgspec
from torch.distributed import TCPStore

from vllm.config import set_current_vllm_config
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.utils import stateless_init_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.utils.network_utils import get_open_port
from vllm.v1.engine import (
    FT_STATUS_CALL_ID,
    EngineCoreOutputs,
    EngineStatusType,
    UtilityOutput,
)
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest, FaultToleranceResult
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import UtilityResult, run_method

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

logger = init_logger(__name__)

FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance state for a single engine core."""

    def __init__(self, engine: "DPEngineCoreProc", parallel_config):
        self.engine = engine
        self.engine_index = engine.engine_index
        self.parallel_config = parallel_config
        ft_config = parallel_config.fault_tolerance_config
        self.engine_recovery_timeout_sec = ft_config.engine_recovery_timeout_sec
        self.auto_recovery = ft_config.auto_recovery

        self.resumed = threading.Event()
        self.resumed.set()
        self.status_type = EngineStatusType.HEALTHY
        self.fault_info: str | None = None
        self._dp_reinit_epoch = 0
        # Original DP ranks indexed by current (densified) rank. The public FT
        # coordinate system is the current one; original ranks never leave the
        # engine except as the cumulative dead set passed to workers.
        dp_size = parallel_config.data_parallel_size
        self._alive_dp_ranks = list(range(dp_size))
        self._initial_dp_size = dp_size
        # Guards against concurrent recovery: auto-recovery runs on the
        # busy-loop thread, external commands on the input-sockets thread.
        self._recovering = False

    def handle_command(self, client_idx: int, call_id: int, ft_args: dict):
        """Dispatch an FT command by instruction name."""
        ft_request = FaultToleranceRequest(**ft_args)
        reject_reason: str | None = None
        if self.status_type != EngineStatusType.UNHEALTHY:
            reject_reason = f"status is {self.status_type.name}"
        elif self._recovering:
            reject_reason = "recovery already in progress"
        if reject_reason is not None:
            reason = (
                f"[FT] Rejecting {ft_request.instruction} on engine "
                f"{self.engine_index}: {reject_reason}"
            )
            logger.warning(reason)
            result = FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason=reason,
            )
        else:
            self._recovering = True
            try:
                result = run_method(self, ft_request.instruction, (ft_request,), {})
            except Exception as e:
                logger.exception("[FT] Instruction '%s' failed", ft_request.instruction)
                result = FaultToleranceResult(
                    request_id=ft_request.request_id, success=False, reason=str(e)
                )
            finally:
                self._recovering = False

        uo = UtilityOutput(call_id)
        uo.result = UtilityResult(msgspec.structs.asdict(result))
        self.engine.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=uo))
        )

    def on_fault(self, exc: Exception):
        """Called by the wrapper when the busy loop raises an exception."""
        self.resumed.clear()
        logger.warning(
            "[FT] Busy loop raised %s. Waiting for recovery.", type(exc).__name__
        )

        engine = self.engine
        aborted = engine.scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
        engine._send_abort_outputs(aborted)
        if engine.batch_queue is not None:
            engine.batch_queue.clear()
        if (
            hasattr(engine.model_executor, "is_failed")
            and engine.model_executor.is_failed
        ):
            self.status_type = EngineStatusType.DEAD
        else:
            self.status_type = EngineStatusType.UNHEALTHY
        self.fault_info = f"{type(exc).__name__}"
        logger.info(
            "[FT] Engine %d status -> %s:",
            self.engine_index,
            self.status_type.name,
            exc_info=exc,
        )
        self._push_status()

        if self.auto_recovery and self.status_type == EngineStatusType.UNHEALTHY:
            try:
                self.auto_recover()
            except Exception:
                logger.exception("[FT] Auto-recovery failed")

    def _push_status(self):
        """Push current health to the client so it can refresh its cache."""
        parallel_config = self.engine.vllm_config.parallel_config
        payload = {
            "id": self.engine_index,
            "status": self.status_type.name.lower(),
            "rank": parallel_config.data_parallel_rank,
            "dp_size": parallel_config.data_parallel_size,
        }
        if self.status_type == EngineStatusType.UNHEALTHY:
            payload["fault_info"] = self.fault_info
            try:
                payload["dead_dp_ranks"] = self._dead_dp_ranks(self._query_mask())
            except Exception:
                logger.warning("[FT] Failed to query dead ranks for status push")
        outputs = EngineCoreOutputs(
            utility_output=UtilityOutput(
                call_id=FT_STATUS_CALL_ID,
                result=UtilityResult(payload),
            )
        )
        outputs.engine_index = self.engine_index
        self.engine.output_queue.put_nowait((0, outputs))

    def _query_mask(self) -> list[int]:
        """Union of all workers' all2all masks.

        A rank is excluded if any worker suspects it.
        """
        ft_request = FaultToleranceRequest(instruction="query_mask", params={})
        results = self.engine.model_executor.collective_rpc(
            "handle_ft_command", args=(ft_request,)
        )
        return [max(bits) for bits in zip(*(r["mask"] for r in results))]

    def _dead_dp_ranks(self, mask: list[int]) -> list[int]:
        """Translate an original-coordinate EP mask into current-coordinate
        dead DP ranks; ranks removed by earlier scale-downs are excluded."""
        tp_size = self.parallel_config.tensor_parallel_size
        dead_original = {r // tp_size for r, v in enumerate(mask) if v}
        return [
            current
            for current, original in enumerate(self._alive_dp_ranks)
            if original in dead_original
        ]

    def _cumulative_dead_dp_ranks(self) -> list[int]:
        """Original-coordinate DP ranks removed by all past scale-downs."""
        return sorted(set(range(self._initial_dp_size)) - set(self._alive_dp_ranks))

    def _exchange_masks(self, my_mask: list[int]) -> list[int] | None:
        """Rank 0 unions all engines' masks via dp_store and publishes it back.

        Dead ranks never write, so rank 0 skips those already masked. Returns
        None on store timeout so the caller fails closed.
        """
        parallel_config = self.engine.vllm_config.parallel_config
        dp_rank = parallel_config.data_parallel_rank
        dp_size = parallel_config.data_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        store = self.engine.dp_store
        epoch = self._dp_reinit_epoch
        final_key = f"ft_final_mask_{epoch}"

        store.set(f"ft_mask_{epoch}_{dp_rank}", json.dumps(my_mask).encode())
        if dp_rank != 0:
            try:
                return json.loads(store.get(final_key).decode())
            except RuntimeError:
                return None

        combined = list(my_mask)
        for rank in range(1, dp_size):
            # Masks use original EP coordinates; map the current rank back to
            # its original rank to find the EP slots it would report on.
            original = self._alive_dp_ranks[rank]
            ep_range = range(original * tp_size, (original + 1) * tp_size)
            if all(combined[i] for i in ep_range):
                continue  # presumed dead: it will never write its mask
            try:
                other = json.loads(store.get(f"ft_mask_{epoch}_{rank}").decode())
            except RuntimeError:
                return None
            combined = [max(a, b) for a, b in zip(combined, other)]
        store.set(final_key, json.dumps(combined).encode())
        return combined

    def auto_recover(self):
        """Auto-recover based on the cluster-wide all2all mask."""
        if self._recovering:
            logger.info("[FT] Auto-recovery skipped: recovery already in progress")
            return
        self._recovering = True
        try:
            mask = self._exchange_masks(self._query_mask())
            if mask is None:
                logger.warning(
                    "[FT] Auto-recovery aborted: mask exchange failed; "
                    "waiting for external command"
                )
                return

            dead_dp_ranks = self._dead_dp_ranks(mask)
            if not dead_dp_ranks:
                logger.info("[FT] Auto-recovery: no newly dead ranks, retrying")
                ft_request = FaultToleranceRequest(instruction="retry", params={})
                self.retry(ft_request)
                return

            my_dp_rank = self.parallel_config.data_parallel_rank
            if my_dp_rank in dead_dp_ranks:
                logger.warning(
                    "[FT] Auto-recovery aborted: this rank is masked as dead "
                    "by the cluster; waiting for external command"
                )
                return

            logger.info(
                "[FT] Auto-recovery: dead_dp_ranks=%s, scaling down", dead_dp_ranks
            )
            ft_request = FaultToleranceRequest(
                instruction="scale_down",
                params={"removed_dp_ranks": dead_dp_ranks},
            )
            self.scale_down(ft_request)
        finally:
            self._recovering = False

    def retry(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        # Workers replay masks for the cumulative dead set (original
        # coordinates), which clean_buffers would otherwise wipe.
        ft_request.params.setdefault("dead_dp_ranks", self._cumulative_dead_dp_ranks())
        return self._reinit_dp_and_dispatch_command(ft_request)

    def scale_down(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config

        if not (
            parallel_config.enable_eplb
            and parallel_config.eplb_config.num_redundant_experts > 0
        ):
            raise ValueError(
                "scale_down requires --enable-eplb with num_redundant_experts > 0"
            )

        # removed_dp_ranks uses current (densified) coordinates, sourced from
        # the dead_dp_ranks field of the status response.
        removed_set = set(ft_request.params["removed_dp_ranks"])
        old_dp_size = parallel_config.data_parallel_size
        old_dp_rank = parallel_config.data_parallel_rank
        new_dp_size = old_dp_size - len(removed_set)
        if (
            not removed_set
            or not removed_set <= set(range(old_dp_size))
            or old_dp_rank in removed_set
            or new_dp_size < 1
        ):
            raise ValueError(
                f"Invalid removed_dp_ranks {sorted(removed_set)} for engine "
                f"{self.engine_index} (dp_size={old_dp_size}, "
                f"dp_rank={old_dp_rank})"
            )

        new_alive = [
            r for i, r in enumerate(self._alive_dp_ranks) if i not in removed_set
        ]
        new_dp_rank = new_alive.index(self._alive_dp_ranks[old_dp_rank])
        ft_request.params["dead_dp_ranks"] = sorted(
            set(range(self._initial_dp_size)) - set(new_alive)
        )

        master_ip = parallel_config.data_parallel_master_ip
        # Rank 0 hosts the TCPStore master; rebuild if it was removed.
        if 0 in removed_set:
            dp_store_port = ft_request.params.get("dp_store_port")
            new_master_ip = ft_request.params.get("dp_master_ip")
            if dp_store_port is None or new_master_ip is None:
                raise ValueError(
                    "dp_store_port and dp_master_ip required when rank 0 is removed"
                )
            master_ip = new_master_ip
            self._rebuild_dp_store(master_ip, dp_store_port, new_dp_rank, new_dp_size)

        ft_request.params.update(
            {
                "new_dp_size": new_dp_size,
                "new_dp_rank": new_dp_rank,
            }
        )
        result = self._reinit_dp_and_dispatch_command(
            ft_request,
            dp_size=new_dp_size,
            dp_rank=new_dp_rank,
            master_ip=master_ip,
        )
        # Commit the alive-rank mapping only after the reinit succeeded.
        self._alive_dp_ranks = new_alive
        logger.info(
            "[FT] Engine %d scale_down complete: dp_size %d->%d, "
            "dp_rank %d->%d, removed %s",
            self.engine_index,
            old_dp_size,
            new_dp_size,
            old_dp_rank,
            new_dp_rank,
            sorted(removed_set),
        )
        return result

    def _reinit_dp_and_dispatch_command(
        self,
        ft_request: FaultToleranceRequest,
        dp_size: int | None = None,
        dp_rank: int | None = None,
        master_ip: str | None = None,
    ) -> FaultToleranceResult:
        """Reinit the DP group, commit the topology, dispatch to workers."""
        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config
        if dp_size is None:
            dp_size = parallel_config.data_parallel_size
        if dp_rank is None:
            dp_rank = parallel_config.data_parallel_rank
        if master_ip is None:
            master_ip = parallel_config.data_parallel_master_ip

        with set_current_vllm_config(engine.vllm_config):
            recovery_round = ft_request.request_id or str(self._dp_reinit_epoch)
            ft_request.params.update(
                self._reinit_dp_group(master_ip, dp_rank, dp_size, recovery_round)
            )
        ft_request.params["dp_master_ip"] = master_ip

        # Commit the topology only after the group reinit succeeded, so a
        # failed recovery leaves a consistent state that can be retried.
        parallel_config.data_parallel_size = dp_size
        parallel_config.data_parallel_rank = dp_rank
        parallel_config.data_parallel_master_ip = master_ip
        engine.dp_size = dp_size
        engine.dp_rank = dp_rank

        if hasattr(engine, "step_counter"):
            engine.step_counter = 0

        engine.model_executor.collective_rpc("handle_ft_command", args=(ft_request,))

        self.status_type = EngineStatusType.HEALTHY
        logger.info("[FT] Engine %d status -> HEALTHY", self.engine_index)
        self.resumed.set()
        self._push_status()
        return FaultToleranceResult(request_id=ft_request.request_id, success=True)

    def _rebuild_dp_store(
        self,
        host: str,
        port: int,
        dp_rank: int,
        dp_size: int,
    ) -> None:
        """Rebuild dp_store when the old master (rank 0) was removed."""
        self.engine.dp_store = TCPStore(
            host,
            port,
            dp_size,
            is_master=(dp_rank == 0),
            timeout=timedelta(seconds=self.engine_recovery_timeout_sec),
        )

    def _reinit_dp_group(
        self, master_ip: str, dp_rank: int, dp_size: int, recovery_round: str
    ) -> dict:
        """Reinit the DP process group. Returns worker params."""
        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config
        worker_key = f"ft_worker_dp_ports_{recovery_round}"
        engine_key = f"ft_engine_dp_port_{recovery_round}"
        enable_eplb = parallel_config.enable_eplb

        if dp_rank == 0:
            worker_ports = [get_open_port() for _ in range(parallel_config.world_size)]
            engine_port = get_open_port()
            engine.dp_store.set(worker_key, json.dumps(worker_ports).encode())
            engine.dp_store.set(engine_key, str(engine_port).encode())
        else:
            worker_ports = json.loads(engine.dp_store.get(worker_key).decode())
            engine_port = int(engine.dp_store.get(engine_key).decode())

        result: dict[str, Any] = {"new_stateless_dp_group_ports": worker_ports}
        if enable_eplb:
            result["new_ep_group_port"] = self._coordinate_port(
                "ft_worker_ep_port", dp_rank, recovery_round
            )
            result["new_eplb_group_port"] = self._coordinate_port(
                "ft_worker_eplb_port", dp_rank, recovery_round
            )
        self._dp_reinit_epoch += 1

        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (
            stateless_init_torch_distributed_process_group(
                master_ip,
                engine_port,
                dp_rank,
                dp_size,
                backend="gloo",
                return_store=True,
            )
        )
        return result

    def _coordinate_port(
        self, key_prefix: str, dp_rank: int, recovery_round: str
    ) -> int:
        """Rank 0 picks a fresh port and publishes it via dp_store;
        other ranks block-read it."""
        key = f"{key_prefix}_{recovery_round}"
        engine = self.engine
        if dp_rank == 0:
            port = get_open_port()
            engine.dp_store.set(key, str(port).encode())
        else:
            port = int(engine.dp_store.get(key).decode())
        return port


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """Wrap the busy loop to catch faults and delegate recovery."""

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as exc:
                if not self.enable_fault_tolerance:
                    raise
                self.ft_sentinel.on_fault(exc)
                recovered = self.ft_sentinel.resumed.wait(
                    timeout=self.ft_sentinel.engine_recovery_timeout_sec
                )
                if recovered:
                    continue
                logger.error(
                    "[FT] No recovery within %ds timeout.",
                    self.ft_sentinel.engine_recovery_timeout_sec,
                )
                raise

    return run_with_fault_tolerance
