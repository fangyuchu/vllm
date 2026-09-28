# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import torch.distributed

from vllm.config import ParallelConfig
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group,
)
from vllm.distributed.elastic_ep.readiness import (
    new_worker_dist_init_ready_keys,
)
from vllm.distributed.utils import get_cached_tcp_store_client
from vllm.logger import init_logger
from vllm.v1.engine import (
    EEPNotificationType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
)
from vllm.v1.engine.core import DPEngineCoreProc

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

WorkerType = Literal["existing", "new", "removing"]


class ScaleUpExistingEngineState(enum.IntEnum):
    PREPARE = 0
    SYNC_KV_CACHE_MEMORY_SIZE = 1
    CAPTURE_NEW_RANKS = 2
    COMMIT_SCALE_UP = 3  # Blocks forward passes.
    COMPLETE = 4


class ScaleUpNewEngineState(enum.IntEnum):
    PRE_KV_INIT = 0
    PREPARE = 1
    COMPLETE = 2


class ScaleDownRemainingEngineState(enum.IntEnum):
    PREPARE = 0
    COMMIT_SCALE_DOWN = 1  # Blocks forward passes.
    COMPLETE = 2


class ScaleDownRemovingEngineState(enum.IntEnum):
    PREPARE = 0
    COMPLETE = 1


EngineState: TypeAlias = (
    ScaleUpExistingEngineState
    | ScaleUpNewEngineState
    | ScaleDownRemainingEngineState
    | ScaleDownRemovingEngineState
)


class ElasticEPScalingState:
    def __init__(
        self,
        model_executor: "Executor",
        engine_core: "DPEngineCoreProc",
        vllm_config: "VllmConfig",
        new_parallel_config: ParallelConfig,
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        reconfig_request: ReconfigureDistributedRequest | None = None,
    ):
        self.model_executor_ref = weakref.ref(model_executor)
        self.engine_core_ref = weakref.ref(engine_core)
        self.vllm_config = vllm_config
        self.old_dp_group = self.engine_core.dp_group if worker_type != "new" else None
        self.old_dp_store = self.engine_core.dp_store if worker_type != "new" else None
        self.new_parallel_config: ParallelConfig = new_parallel_config
        self.new_dp_group = self.engine_core.dp_group if worker_type == "new" else None
        self.new_dp_store = self.engine_core.dp_store if worker_type == "new" else None
        self.worker_type = worker_type
        self.scale_type = scale_type
        self.reconfig_request = reconfig_request
        self.commit_requested = False
        self._prepare_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ElasticEPPrepare"
        )
        self._prepare_future: Future[Any] | None = None
        self._drain_future: Future[Any] | None = None
        self._drain_dummy_future: Future[Any] | None = None
        self._prepare_workers_complete = False
        self._prepare_workers_synchronized = False
        self._new_dp_sync: tuple[object, Any] | None = None
        self._precommit_capture_enabled: bool | None = None
        self._precommit_operation_id = ""
        self._prepare_quiesce_started = False
        self._prepare_quiesce_complete = False
        self.state: EngineState
        if scale_type == "scale_up":
            self.state = (
                ScaleUpNewEngineState.PRE_KV_INIT
                if worker_type == "new"
                else ScaleUpExistingEngineState.PREPARE
            )
        else:
            self.state = (
                ScaleDownRemovingEngineState.PREPARE
                if worker_type == "removing"
                else ScaleDownRemainingEngineState.PREPARE
            )

    @property
    def model_executor(self) -> "Executor":
        model_executor = self.model_executor_ref()
        if model_executor is None:
            raise RuntimeError("Model executor has been garbage collected")
        return model_executor

    @property
    def engine_core(self) -> "DPEngineCoreProc":
        engine_core = self.engine_core_ref()
        if engine_core is None:
            raise RuntimeError("Engine core has been garbage collected")
        return engine_core

    def _collective_rpc(self, *args, **kwargs):
        return self.model_executor.collective_rpc(*args, **kwargs)

    def _get_local_dp_collective_state(self) -> tuple[int, int]:
        get_states = getattr(
            self.model_executor, "get_elastic_ep_dp_collective_states", None
        )
        if get_states is None:
            raise RuntimeError(
                "Elastic EP batch-boundary drain requires Worker DP "
                "collective state, but the current executor does not expose it"
            )

        states = get_states()
        if not states:
            raise RuntimeError("No Worker DP collective state is available")
        if any(state != states[0] for state in states[1:]):
            raise RuntimeError(
                "Local workers reached different DP metadata collective "
                f"epochs: {states}"
            )
        if states[0][0] < 0 or states[0][1] < 0:
            raise RuntimeError(
                "The current Worker does not publish DP metadata collective state"
            )
        return states[0]

    def is_precommit_prepare_ready(self) -> bool:
        return (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state is ScaleUpExistingEngineState.PREPARE
            and self._prepare_workers_complete
            and self._precommit_capture_enabled is True
            and not self._prepare_workers_synchronized
        )

    def begin_prepare_quiesce(self) -> None:
        """Enter drain after all old ranks agree in the regular DP-state sync."""
        assert self.is_precommit_prepare_ready()
        self._prepare_workers_synchronized = True
        self._prepare_quiesce_started = True
        self.engine_core._eep_drain_batch_queue = True
        self.engine_core._eep_force_dummy_batch = False
        logger.info(
            "[Elastic EP] All old ranks reached prepare consensus; "
            "starting inference drain: operation_id=%s, dp_step=%s",
            self._operation_id,
            self.engine_core.step_counter,
        )

    def poll_pending_batch(self, future: Future[Any]) -> bool:
        """Receive a queued batch without blocking EngineCore progress exchange."""
        if self._drain_future is None:
            # Multiproc FutureWrapper receives RPC replies lazily in result().
            # Its done() alone cannot make progress. The prepare thread is idle
            # here, and only the oldest batch may read executor replies.
            self._drain_future = self._prepare_executor.submit(future.result)
        if not self._drain_future.done():
            return False
        completed = self._drain_future
        self._drain_future = None
        completed.result()
        return True

    def start_catch_up_dummy_batch(self) -> None:
        """Run one catch-up batch while EngineCore continues exchanging epochs."""
        if self._drain_dummy_future is None:
            self._drain_dummy_future = self._prepare_executor.submit(
                self.model_executor.execute_dummy_batch
            )

    def _progress_prepare_quiesce(self) -> bool:
        """Drain at a common Worker DP collective boundary."""
        assert self.old_dp_group is not None
        self._prepare_quiesce_started = True

        if self._drain_dummy_future is not None and self._drain_dummy_future.done():
            self._drain_dummy_future.result()
            self._drain_dummy_future = None

        batch_queue = getattr(self.engine_core, "batch_queue", None)
        # Include submitted dummy work even before its Worker publishes an epoch.
        # This prevents duplicate catch-up batches and premature finalization.
        local_pending = bool(batch_queue) or self._drain_dummy_future is not None
        self.engine_core._eep_drain_batch_queue = True
        entered_epoch, completed_epoch = self._get_local_dp_collective_state()
        local_state = torch.tensor(
            [entered_epoch, completed_epoch, int(local_pending)],
            dtype=torch.int64,
        )
        gathered_states = [
            torch.empty_like(local_state) for _ in range(self.old_dp_group.size())
        ]
        torch.distributed.all_gather(
            gathered_states,
            local_state,
            group=self.old_dp_group,
        )

        states = [state.tolist() for state in gathered_states]
        target_epoch = max(state[0] for state in states)
        needs_catch_up = not local_pending and entered_epoch < target_epoch
        self.engine_core._eep_force_dummy_batch = needs_catch_up
        if needs_catch_up:
            self.engine_core.engines_running = True

        all_quiesced = all(
            entered == target_epoch and completed == target_epoch and pending == 0
            for entered, completed, pending in states
        )
        if all_quiesced:
            self.engine_core._eep_force_dummy_batch = False
            return True

        logger.info_once(
            "[Elastic EP] Waiting for a common DP batch boundary before "
            "precommit finalization: local_epoch=%s/%s, target_epoch=%s, "
            "local_pending_batches=%s, catch_up_dummy=%s",
            completed_epoch,
            entered_epoch,
            target_epoch,
            0 if batch_queue is None else len(batch_queue),
            needs_catch_up,
        )
        return False

    def _resume_model_execution_after_prepare_quiesce(self) -> None:
        # Reconfiguration requests reach EngineCores independently, so their
        # pre-quiesce running flags can differ. Resume from a common DP-state
        # sync cadence after target-topology NPU collectives have completed.
        self.engine_core.step_counter = 0
        self.engine_core.engines_running = True
        self.engine_core._eep_force_dummy_batch = False
        self.engine_core._eep_drain_batch_queue = False

    def should_skip_dummy_batch(self) -> bool:
        if not getattr(self.engine_core, "_eep_drain_batch_queue", False):
            return False
        return not getattr(self.engine_core, "_eep_force_dummy_batch", False)

    def should_defer_dp_state_sync(self) -> bool:
        return bool(
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and self.state == ScaleUpExistingEngineState.PREPARE
            and self._prepare_quiesce_started
        )

    def _wait_for_async_workers(self, done_keys: list[str]) -> None:
        assert self.reconfig_request is not None
        # TCPStore.wait holds the connection lock. Use a dedicated connection
        # so EngineCore's readiness checks can continue during preparation.
        # Connect on this background thread to keep it off the serving loop.
        coord_store = torch.distributed.TCPStore(
            self.reconfig_request.new_data_parallel_master_ip,
            self.reconfig_request.coord_store_port,
            is_master=False,
            wait_for_workers=False,
        )
        coord_store.wait(done_keys)

    def _execute_async(self, execute_method: str, *args) -> bool:
        if self._prepare_future is None:
            done_keys = self._collective_rpc(
                "elastic_ep_execute",
                args=("start_async", execute_method, *args),
            )
            self._prepare_future = self._prepare_executor.submit(
                self._wait_for_async_workers, done_keys
            )
        if not self._prepare_future.done():
            return False

        self._prepare_future.result()
        self._collective_rpc("elastic_ep_execute", args=("clear_async",))
        self._prepare_future = None
        return True

    def progress(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self._progress_new_engine()
                if self.worker_type == "new"
                else self._progress_existing_engine()
            )
        return (
            self._progress_removing_engine()
            if self.worker_type == "removing"
            else self._progress_remaining_engine()
        )

    def run_pre_kv_init_states(self) -> None:
        assert self.scale_type == "scale_up" and self.worker_type == "new"
        assert self.state == ScaleUpNewEngineState.PRE_KV_INIT
        assert self.progress()
        assert self.state == ScaleUpNewEngineState.PREPARE

    def _progress_existing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleUpExistingEngineState.PREPARE:
            if not self._prepare_workers_complete:
                if not self._prepare_workers():
                    return False
                self._prepare_workers_complete = True

            # Most platforms finish preparation entirely on the background
            # Worker thread. Ascend's V3 fast path has a second phase with NPU
            # mapping/MC2 collectives. Match the migration-source ordering:
            # background groups and weights first, then drain old-rank model
            # execution immediately before this collective phase.
            if self._uses_precommit_graph_capture():
                if not self._prepare_workers_synchronized:
                    # Idle ranks must also run dummy steps to reach the regular
                    # DP-state sync. Keep serving until that sync grants every
                    # old rank permission to drain in the same round.
                    self.engine_core.engines_running = True
                    return False

                if not self._prepare_quiesce_complete:
                    if not self._progress_prepare_quiesce():
                        return False
                    # The progress all-gather already establishes a common
                    # quiescent boundary; no additional staged barrier is needed.
                    self._prepare_quiesce_complete = True
                try:
                    self._collective_rpc(
                        "elastic_ep_execute",
                        args=(
                            "finalize_precommit_prepare",
                            self._operation_id,
                        ),
                    )
                finally:
                    self._resume_model_execution_after_prepare_quiesce()
            self.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
            return True

        elif state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            if not self._sync_kv_cache_memory_size():
                return False
            if self._uses_precommit_graph_capture():
                self.state = ScaleUpExistingEngineState.CAPTURE_NEW_RANKS
            else:
                self.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
                self._mark_ready_for_switch()
            return True

        elif state == ScaleUpExistingEngineState.CAPTURE_NEW_RANKS:
            if not self._execute_async(
                "run_new_rank_capture_companion",
                self._operation_id,
            ):
                return False
            self.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
            self._mark_ready_for_switch()
            return True

        elif state == ScaleUpExistingEngineState.COMMIT_SCALE_UP:
            if not self.commit_requested:
                return False
            self._commit_new_dp_group()
            self._collective_rpc(
                "elastic_ep_execute",
                args=("commit_scale_up", True),
            )
            self.state = ScaleUpExistingEngineState.COMPLETE
            self._update_parallel_config()
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleUpExistingEngineState.COMPLETE
            return True

    def _progress_new_engine(self) -> bool:
        state = self.state
        assert self.new_dp_group is not None and self.new_dp_store is not None

        if state == ScaleUpNewEngineState.PRE_KV_INIT:
            operation_ids = self._collective_rpc(
                "elastic_ep_execute",
                args=("prepare_new_worker", self.reconfig_request),
            )
            resolved_operation_ids = {
                str(operation_id) for operation_id in operation_ids if operation_id
            }
            if len(resolved_operation_ids) > 1:
                raise RuntimeError(
                    "Workers resolved different external Elastic EP operation "
                    f"IDs: {operation_ids}"
                )
            if resolved_operation_ids:
                self._precommit_operation_id = resolved_operation_ids.pop()
            self.engine_core.available_gpu_memory_for_kv_cache = (
                ParallelConfig.sync_kv_cache_memory_size(self.new_dp_group, -1)
            )
            self.state = ScaleUpNewEngineState.PREPARE
            return True

        elif state == ScaleUpNewEngineState.PREPARE:
            if self._uses_precommit_graph_capture():
                self._collective_rpc(
                    "elastic_ep_execute",
                    args=("capture_new_rank_graphs", self._operation_id),
                )
            else:
                self._collective_rpc(
                    "elastic_ep_execute", args=("warmup_new_worker",)
                )
            self._mark_ready_for_switch()
            self._wait_for_external_scale_commit()
            tensor = torch.tensor([0, 0, 0], dtype=torch.int32, device="cpu")
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=self.new_dp_group,
            )
            data = tensor.tolist()
            self.engine_core.engines_running = bool(data[0])
            self.engine_core.current_wave = int(data[1])
            self.engine_core.step_counter = int(data[2])
            self._collective_rpc("elastic_ep_execute", args=("commit_scale_up", False))
            self.state = ScaleUpNewEngineState.COMPLETE
            return True

        else:
            assert self.state == ScaleUpNewEngineState.COMPLETE
            return True

    def _wait_for_external_scale_commit(self) -> None:
        parallel = self.new_parallel_config
        if not parallel.data_parallel_external_lb:
            return
        from vllm.distributed.elastic_ep.external_elastic_ep import (
            ExternalElasticEPScaleCoordinator,
        )

        store = get_cached_tcp_store_client(
            parallel.data_parallel_master_ip, parallel._coord_store_port
        )
        key = ExternalElasticEPScaleCoordinator.key
        epoch = self._operation_id or store.get(key("current_epoch")).decode()
        if not store.check([key(epoch, "pause_before_commit")]):
            return
        # Old ranks still serve on the original group. Do not enter the new
        # group's all-reduce until their schedulers have collectively paused.
        while True:
            error_key = key(epoch, "error")
            if store.check([error_key]):
                raise RuntimeError(store.get(error_key).decode())
            if store.check([key(epoch, "commit_started")]):
                return
            time.sleep(0.1)

    def _progress_remaining_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleDownRemainingEngineState.PREPARE:
            if self._prepare_workers():
                self.state = ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN
                self._mark_ready_for_switch()
                return True
            return False

        elif state == ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN:
            if not self.commit_requested:
                return False
            self._commit_scale_down(removing=False)
            self._commit_new_dp_group()
            self._update_parallel_config()
            self.state = ScaleDownRemainingEngineState.COMPLETE
            self._send_reconfigure_finished()
            return True

        else:
            assert self.state == ScaleDownRemainingEngineState.COMPLETE
            return True

    def _progress_removing_engine(self) -> bool:
        state = self.state
        assert self.old_dp_group is not None

        if state == ScaleDownRemovingEngineState.PREPARE:
            assert self.old_dp_group.rank() > 0
            self._commit_scale_down(removing=True)
            self.state = ScaleDownRemovingEngineState.COMPLETE
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.SHUTDOWN_COMPLETE
            )
            return True

        else:
            assert self.state == ScaleDownRemovingEngineState.COMPLETE
            return True

    def is_ready_for_switch(self) -> bool:
        return self.worker_type == "existing" and (
            self.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
            or self.state is ScaleDownRemainingEngineState.COMMIT_SCALE_DOWN
        )

    @property
    def _operation_id(self) -> str:
        if self.reconfig_request is None:
            return self._precommit_operation_id
        return self.reconfig_request.operation_id

    def _uses_precommit_graph_capture(self) -> bool:
        if self._precommit_capture_enabled is None:
            results = self._collective_rpc(
                "elastic_ep_execute",
                args=("supports_precommit_graph_capture", self._operation_id),
            )
            enabled_values = {bool(result) for result in results}
            if len(enabled_values) != 1:
                raise RuntimeError(
                    f"Workers disagreed on pre-commit graph capture support: {results}"
                )
            self._precommit_capture_enabled = enabled_values.pop()
        return self._precommit_capture_enabled

    @property
    def ready_key(self) -> str:
        return f"eep_ready/{self.new_parallel_config.data_parallel_rank}"

    def _mark_ready_for_switch(self) -> None:
        parallel_config = self.new_parallel_config
        get_cached_tcp_store_client(
            parallel_config.data_parallel_master_ip,
            parallel_config._coord_store_port,
        ).set(self.ready_key, b"1")

    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        return (
            self.state == ScaleDownRemovingEngineState.COMPLETE
            if self.worker_type == "removing"
            else self.state == ScaleDownRemainingEngineState.COMPLETE
        )

    def _init_new_dp_group(self) -> tuple[Any, Any]:
        return self.new_parallel_config.stateless_init_dp_group(return_store=True)

    def _ensure_new_dp_group(self) -> bool:
        if self.new_dp_group is not None:
            return True

        if self._prepare_future is None:
            self._prepare_future = self._prepare_executor.submit(
                self._init_new_dp_group
            )
        if not self._prepare_future.done():
            return False

        self.new_dp_group, self.new_dp_store = self._prepare_future.result()
        self._prepare_future = None
        return True

    def _prepare_workers(self) -> bool:
        assert self.old_dp_group is not None
        if not self._ensure_new_dp_group():
            return False
        if (
            self.scale_type == "scale_up"
            and self.worker_type == "existing"
            and not self._new_workers_dist_init_ready()
        ):
            return False
        if not self._execute_async(
            "prepare_reconfiguration",
            self.reconfig_request,
            self.new_parallel_config.use_all2all,
        ):
            return False
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Prepared reconfiguration")
        return True

    def _new_workers_dist_init_ready(self) -> bool:
        """Wait to create Worker groups until every new Worker can join."""
        assert self.old_dp_group is not None
        ready_keys = new_worker_dist_init_ready_keys(
            self.new_parallel_config,
            self.old_dp_group.size(),
            self.new_parallel_config.data_parallel_size,
            self.new_parallel_config.world_size,
        )
        coord_store = get_cached_tcp_store_client(
            self.new_parallel_config.data_parallel_master_ip,
            self.new_parallel_config._coord_store_port,
        )
        if not coord_store.check(ready_keys):
            return False

        logger.info_once(
            "[Elastic EP scale-up] All new Workers reached distributed init; "
            "starting standby communication group creation"
        )
        return True

    def _sync_kv_cache_memory_size(self) -> bool:
        assert self.engine_core.available_gpu_memory_for_kv_cache > 0
        assert self.new_dp_group is not None and self.old_dp_group is not None

        if self._new_dp_sync is None:
            tensor = torch.tensor(
                [self.engine_core.available_gpu_memory_for_kv_cache],
                dtype=torch.int64,
                device="cpu",
            )
            work = torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.new_dp_group,
                async_op=True,
            )
            self._new_dp_sync = (tensor, work)
            return False

        _, work = self._new_dp_sync
        if not work.is_completed():
            return False
        work.wait()
        self._new_dp_sync = None
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] Synced KV cache memory size to new workers")
        return True

    def _commit_new_dp_group(self):
        old_dp_group = self.old_dp_group
        stateless_destroy_torch_distributed_process_group(old_dp_group)
        assert self.new_dp_group is not None
        new_dp_group = self.new_dp_group
        self.engine_core.dp_group = new_dp_group
        self.engine_core.dp_rank = new_dp_group.rank()
        self.engine_core.dp_store = self.new_dp_store
        engines_running = int(self.engine_core.engines_running)
        current_wave = self.engine_core.current_wave
        step_counter = self.engine_core.step_counter
        tensor = torch.tensor(
            [engines_running, current_wave, step_counter],
            dtype=torch.int32,
            device="cpu",
        )
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=new_dp_group
        )
        data = tensor.tolist()
        self.engine_core.engines_running = bool(data[0])
        self.engine_core.current_wave = int(data[1])
        self.engine_core.step_counter = int(data[2])
        if new_dp_group.rank() == 0:
            logger.info("[Elastic EP] Switched to new setup")

    def _send_reconfigure_finished(self):
        assert self.new_dp_group is not None
        if (
            self.new_dp_group.rank() == 0
            or self.vllm_config.parallel_config.data_parallel_external_lb
        ):
            self.engine_core._eep_send_engine_core_notification(
                EEPNotificationType.RECONFIGURE_FINISHED
            )

    def _commit_scale_down(self, removing: bool):
        assert self.reconfig_request is not None and self.old_dp_group is not None
        self._collective_rpc(
            "elastic_ep_execute",
            args=(
                "commit_scale_down",
                self.reconfig_request.new_data_parallel_size,
                removing,
            ),
        )
        if self.old_dp_group.rank() == 0:
            logger.info("[Elastic EP] EPLB reshuffle completed")

    def _update_parallel_config(self):
        assert self.reconfig_request is not None
        reconfig_request = self.reconfig_request
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        parallel_config._data_parallel_master_port_list = (
            reconfig_request.new_data_parallel_master_port_list
        )
        parallel_config._coord_store_port = reconfig_request.coord_store_port

        if self.scale_type == "scale_up":
            ft_sentinel = getattr(self.engine_core, "ft_sentinel", None)
            if ft_sentinel is not None:
                ft_sentinel.reset_after_scale_up()
