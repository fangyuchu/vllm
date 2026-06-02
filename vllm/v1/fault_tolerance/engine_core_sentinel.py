# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCoreSentinel and fault_tolerant_wrapper for the engine core.

The EngineCoreSentinel executes recovery logic via collective_rpc to workers.
The wrapper decorates the busy loop to catch faults and delegate recovery to
the sentinel. All FT state and logic lives here — EngineCore and Worker hold
only a reference to their sentinel.
"""

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineStatusType,
    UtilityOutput,
)
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import UtilityResult, run_method

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger(__name__)

FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance state for a single engine core.

    Sends commands to workers via collective_rpc, tracks resume state,
    and executes FT instructions (retry).
    """

    def __init__(self, engine: "EngineCoreProc", parallel_config):
        self.engine = engine
        self.engine_index = engine.engine_index
        self.parallel_config = parallel_config
        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )

        self.resumed = threading.Event()
        self.resumed.set()
        self.status_type = EngineStatusType.HEALTHY

    @classmethod
    def create(cls, engine: "EngineCoreProc", parallel_config) -> "EngineCoreSentinel":
        """Create sentinel and initialize worker sentinels via collective RPC.

        Call this from EngineCoreProc.__init__ when FT is enabled.
        """
        sentinel = cls(engine=engine, parallel_config=parallel_config)
        engine.model_executor.collective_rpc(
            method="create_worker_sentinel",
            non_block=False,
        )
        return sentinel

    # ------------------------------------------------------------------
    # Command dispatch (called from process_input_sockets thread)
    # ------------------------------------------------------------------

    def handle_command(self, client_idx: int, call_id: int, ft_args: dict):
        """Dispatch an FT command by instruction name and enqueue the result."""
        ft_request = FaultToleranceRequest(**ft_args)

        try:
            result = run_method(self, ft_request.instruction, (ft_request,), {})
        except Exception as e:
            logger.exception("[FT] Instruction '%s' failed", ft_request.instruction)
            result = {
                "request_id": ft_request.request_id,
                "success": False,
                "reason": str(e),
            }

        uo = UtilityOutput(call_id)
        uo.result = UtilityResult(result)
        self.engine.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=uo))
        )

    # ------------------------------------------------------------------
    # Fault handling (called by wrapper, runs in busy-loop thread)
    # ------------------------------------------------------------------

    def on_fault(self, exc: Exception):
        """Non-blocking fault initialization. Called by the wrapper when
        the busy loop raises an exception."""
        self.resumed.clear()
        logger.warning(
            "[FT] Busy loop raised %s. Waiting for recovery.", type(exc).__name__
        )

        self._preempt_running()
        self._publish_status(EngineStatusType.UNHEALTHY, str(exc))

    # ------------------------------------------------------------------
    # Instruction handlers (method name == instruction string)
    # ------------------------------------------------------------------

    def status(self, ft_request: FaultToleranceRequest) -> dict:
        return {
            "request_id": ft_request.request_id,
            "success": True,
            "engine_id": self.engine_index,
            "status": self.status_type.name.lower(),
        }

    def retry(self, ft_request: FaultToleranceRequest) -> dict:
        engine = self.engine
        executor = engine.model_executor

        # 1) Reinit DP process group (engine side) if in DP mode.
        from vllm.config import set_current_vllm_config

        with set_current_vllm_config(engine.vllm_config):
            ft_request.params.update(self._reinit_dp_group())

        # 2) Drain stale futures/responses so the MQ channel is clean.
        self._drain_stale_responses(executor)

        # 3) Tell workers to clean state and reinit their DP group.
        executor.collective_rpc(
            method="handle_ft_command",
            args=(ft_request,),
            non_block=False,
        )

        # 4) Reset DP-specific engine state if in DP mode.
        if hasattr(engine, "dp_group"):
            engine.engines_running = False
            engine.step_counter = 0  # type: ignore[attr-defined]
            if (
                engine.has_coordinator  # type: ignore[attr-defined]
                and engine.dp_rank == 0  # type: ignore[attr-defined]
            ):  # type: ignore[attr-defined]
                engine.output_queue.put_nowait(
                    (
                        -1,
                        EngineCoreOutputs(wave_complete=engine.current_wave),  # type: ignore[attr-defined]
                    )
                )
            engine.current_wave += 1  # type: ignore[attr-defined]

        # Clear the batch queue — stale futures from before the fault
        # would return FAILURE responses if re-consumed after resume.
        if hasattr(engine, "batch_queue") and engine.batch_queue is not None:
            n_cleared = len(engine.batch_queue)
            engine.batch_queue.clear()
            if n_cleared > 0:
                logger.info(
                    "[FT] Cleared %d stale batch(es) from batch queue", n_cleared
                )

        self._publish_status(EngineStatusType.HEALTHY)
        self.resumed.set()
        return {"request_id": ft_request.request_id, "success": True}

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def _reinit_dp_group(self) -> dict:
        """Reinit DP process group if in DP mode. Returns worker params."""
        engine = self.engine
        if not hasattr(engine, "dp_group"):
            return {}

        from vllm.distributed import (
            stateless_destroy_torch_distributed_process_group,
        )
        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )
        from vllm.utils.network_utils import get_open_port

        parallel_config = engine.vllm_config.parallel_config

        if engine.dp_rank == 0:  # type: ignore[attr-defined]
            worker_port = get_open_port()
            engine_port = get_open_port()
            engine.dp_store.set(  # type: ignore[attr-defined]
                "ft_worker_dp_port",
                str(worker_port).encode(),
            )
            engine.dp_store.set(  # type: ignore[attr-defined]
                "ft_engine_dp_port",
                str(engine_port).encode(),
            )
        else:
            worker_port = int(
                engine.dp_store.get(  # type: ignore[attr-defined]
                    "ft_worker_dp_port"
                ).decode()
            )
            engine_port = int(
                engine.dp_store.get(  # type: ignore[attr-defined]
                    "ft_engine_dp_port"
                ).decode()
            )

        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (  # type: ignore[attr-defined]
            stateless_init_torch_distributed_process_group(
                parallel_config.data_parallel_master_ip,
                engine_port,
                parallel_config.data_parallel_rank,
                parallel_config.data_parallel_size,
                backend="gloo",
                return_store=True,
            )
        )
        return {"new_stateless_dp_group_port": worker_port}

    @staticmethod
    def _drain_stale_responses(executor):
        """Drain stale futures and pending responses from the executor's
        message queues before issuing a new collective_rpc."""
        if not hasattr(executor, "futures_queue"):
            return
        num_stale = len(executor.futures_queue)
        executor.futures_queue.clear()
        if num_stale == 0 and not getattr(executor, "is_failed", False):
            # No known stale items and executor is healthy — skip drain.
            return
        logger.info("[FT] Draining %d stale response(s) from response queue", num_stale)
        if executor.kv_output_aggregator is not None:
            mqs = executor.response_mqs
        else:
            mqs = (executor.response_mqs[executor.output_rank],)
        for mq in mqs:
            for _ in range(max(num_stale, 5)):
                try:
                    mq.dequeue(timeout=1)
                except Exception:
                    break

    def _preempt_running(self):
        """Preempt running requests and clear batch state."""
        engine = self.engine
        timestamp = time.monotonic()
        while engine.scheduler.running:  # type: ignore[attr-defined]
            request = engine.scheduler.running.pop()  # type: ignore[attr-defined]
            engine.scheduler._preempt_request(  # type: ignore[attr-defined]
                request, timestamp
            )
        engine.scheduler.prev_step_scheduled_req_ids.clear()  # type: ignore[attr-defined]
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: EngineStatusType, message: str | None = None):
        self.status_type = status
        if message:
            logger.info(
                "[FT] Engine %d status -> %s: %s",
                self.engine_index,
                status.name,
                message,
            )
        else:
            logger.info("[FT] Engine %d status -> %s", self.engine_index, status.name)


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """Wrap the busy loop to catch faults and delegate recovery.

    On exception: on_fault() initializes paused state, then the wrapper
    waits on the `resumed` Event. The retry command arrives via
    process_input_sockets (separate thread), executes recovery, and
    signals `resumed` — no queue polling needed.
    """

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.ft_sentinel.resumed.set()
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
