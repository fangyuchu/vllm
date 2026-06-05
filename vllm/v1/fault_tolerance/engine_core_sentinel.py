# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import msgspec.msgpack
import zmq

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreRequestType, EngineStatusType
from vllm.v1.engine.exceptions import EngineLoopPausedError
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.serial_utils import run_method

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

logger = init_logger(__name__)


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for:
      1. Receiving fault signals (exceptions raised in EngineCore busy loop)
      2. Receiving and executing commands from ClientSentinel
      3. Reporting execution results or faults back to the ClientSentinel
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        engine_index: int,
        engine_input_q: queue.Queue,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        worker_cmd_addr: str,
        engine: "EngineCoreProc",
    ):
        self.engine_index = engine_index
        super().__init__(
            parallel_config,
            f"DP_{engine_index}",
            sentinel_identity,
        )
        self.engine_identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engine = engine
        self.data_parallel_size = parallel_config.data_parallel_size
        self.fault_signal_q: queue.Queue[Exception] = queue.Queue()
        self.cmd_q: queue.Queue[FaultToleranceRequest | None] = queue.Queue(maxsize=1)

        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )
        self.stop_busy_loop = threading.Event()
        self.busy_loop_paused = threading.Event()
        self.engine_input_q = engine_input_q
        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=worker_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_cmd_poller = zmq.Poller()
        self.worker_cmd_poller.register(self.worker_cmd_socket, zmq.POLLIN)
        self.worker_identities = [
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(parallel_config.tensor_parallel_size)
            for pp_rank in range(parallel_config.pipeline_parallel_size)
        ]

        # Client <-> EngineCoreSentinel sockets
        self.engine_fault_socket = make_zmq_socket(
            self.ctx,
            engine_fault_socket_addr,
            zmq.DEALER,
            bind=False,
            identity=sentinel_identity,
        )

        threading.Thread(
            target=self.run, daemon=True, name="EngineCoreSentinelMonitorThread"
        ).start()

    def run(self):
        """Continuously poll for fault signals and report to client sentinel."""
        while not self.sentinel_dead:
            # Check for engine fault signals
            self.poll_and_report_fault_events()

    def poll_and_report_fault_events(self):
        try:
            engine_exception = self.fault_signal_q.get(timeout=1)
            logger.error(
                "%s Detected exception %s: %s\n Call Stack:\n%s",
                self.sentinel_name,
                type(engine_exception).__name__,
                engine_exception,
                "".join(traceback.format_tb(engine_exception.__traceback__)),
            )
            engine_status = (
                EngineStatusType.PAUSED
                if isinstance(engine_exception, EngineLoopPausedError)
                else EngineStatusType.UNHEALTHY
            )
            msg = FaultInfo.from_exception(
                engine_exception, self.engine_index, engine_status, self.engine_identity
            )
            msg_bytes = msgspec.msgpack.encode(msg)
            self.engine_fault_socket.send_multipart([b"", msg_bytes])
        except queue.Empty:
            pass

    def handle_fault(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        return self._execute_cmd(ft_request)

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """Pause the busy loop of engine core safely."""
        logger.info("Start pausing EngineCore")
        timeout = ft_request.params["timeout"]
        deadline = time.monotonic() + timeout
        # set the flag to signal busy loop should pause
        self.stop_busy_loop.set()
        # Put a wakeup request to unblock the busy loop
        # if it's blocked on input_queue.get()
        self.engine_input_q.put((EngineCoreRequestType.WAKEUP, None))
        self._execute_command_on_workers(
            FaultToleranceRequest(str(uuid.uuid4()), "pause", ft_request.params),
            self.worker_identities,
            timeout=timeout,
        )
        remaining_timeout = max(0, deadline - time.monotonic())
        success = self.busy_loop_paused.wait(remaining_timeout)
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=success,
            reason=None if success else "Busy loop did not pause within timeout.",
        )

    def retry(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """
        Handle the retry instruction from the ClientSentinel.
        This instruction tells the EngineCore to continue its busy loop
        after being suspended due to an exception.
        """
        if not self.busy_loop_paused.is_set():
            return FaultToleranceResult(ft_request.request_id, True)
        timeout = ft_request.params["timeout"]
        self.parallel_config._coord_store_port = ft_request.params["coord_store_port"]
        res = self._execute_command_on_workers(
            FaultToleranceRequest(str(uuid.uuid4()), "retry", ft_request.params),
            self.worker_identities,
            timeout=timeout,
        )
        if not res.success:
            return res
        if self.data_parallel_size > 1:
            # If the Gloo communication times out,
            # the data parallel group (dp_group) needs to be reinitialized
            reinit_request = FaultToleranceRequest(
                instruction="reinit_dp_group_on_fault_tolerance",
                request_id=str(uuid.uuid4()),
                params={},
            )
            self.cmd_q.put(reinit_request)
        else:
            self.cmd_q.put(None)

        self.stop_busy_loop.clear()
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=True,
        )

    def _calculate_exclude_ep_ranks(
        self, exclude_dp_ranks: list[int], vllm_config: VllmConfig
    ) -> list[int]:
        """Calculate excluded EP ranks from excluded DP ranks."""
        tensor_model_parallel_size = vllm_config.parallel_config.tensor_parallel_size
        exclude_ep_ranks: list[int] = []
        for dp_rank in exclude_dp_ranks:
            start = dp_rank * tensor_model_parallel_size
            end = (dp_rank + 1) * tensor_model_parallel_size
            exclude_ep_ranks.extend(range(start, end))

        exclude_ep_ranks = sorted(list(set(exclude_ep_ranks)))
        return exclude_ep_ranks

    def _calculate_parallel_config(
        self, vllm_config: VllmConfig, exclude_dp_ranks_list: list[int]
    ):
        """Parse excluded DP ranks to
        calculate scaled-down EP/DP sizes and DP rank mapping."""
        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallel is not supported for scaling down."
            )
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        old_dp_size = vllm_config.parallel_config.data_parallel_size

        new_dp_size = old_dp_size - len(exclude_dp_ranks_list)
        new_ep_size = new_dp_size * tp_size

        return new_ep_size, new_dp_size

    def _build_vllm_config_update_dict(
        self,
        parallel_config: Any,
        new_ep_size: int,
        data_parallel_size: int,
        rank_mapping: Any,
    ) -> dict[str, Any]:
        """Build dictionary of VLLM config updates for downstream workers.

        Args:
            parallel_config: Current parallel configuration object
            new_ep_size: New expert parallelism size
            data_parallel_size: New data parallelism size
            rank_mapping: New rank mapping after exclusion

        Returns:
            Dict[str, Any]: VLLM configuration update parameters
        """
        return {
            "ep_world_size": new_ep_size,
            "rank_mapping": rank_mapping,
            "data_parallel_size": data_parallel_size,
            "data_parallel_rank": parallel_config.data_parallel_rank,
            "data_parallel_size_local": parallel_config.data_parallel_size_local,
            "expert_parallel_size": (
                data_parallel_size
                * parallel_config.pipeline_parallel_size
                * parallel_config.tensor_parallel_size
            ),
            "data_parallel_master_port": parallel_config.data_parallel_master_port,
        }

    def reinit_dp_group_on_fault_tolerance(self):
        if not isinstance(self.engine, DPEngineCoreProc):
            return
        stateless_destroy_torch_distributed_process_group(self.engine.dp_group)
        self.engine.dp_group = (
            self.engine.vllm_config.parallel_config.stateless_init_dp_group()
        )
        self.engine.step_counter = 0

    def scale_down(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """Scale down the engine cluster by removing specified DP ranks.

        This method adjusts parallel configuration parameters,
        broadcasts the scale_down command to downstream workers,
        and reinitializes the DP group for fault tolerance.
        """
        # Validate required keyword arguments
        # Extract and type-cast parameters from kwargs
        timeout = ft_request.params["timeout"]
        original_to_new = ft_request.params["original_to_new"]
        exclude_dp_ranks = ft_request.params["exclude_dp_ranks"]
        self.parallel_config._coord_store_port = ft_request.params["coord_store_port"]
        deadline = time.monotonic() + timeout
        original_to_new = {int(k): v for k, v in original_to_new.items()}
        self.engine_index = original_to_new[self.engine_index]
        self.sentinel_tag = f"DP_{self.engine_index}"
        exclude_ep_ranks = self._calculate_exclude_ep_ranks(
            exclude_dp_ranks, self.engine.vllm_config
        )

        new_ep_size, data_parallel_size = self._calculate_parallel_config(
            self.engine.vllm_config, exclude_dp_ranks
        )
        with set_current_vllm_config(self.engine.vllm_config):
            parallel_config = self.engine.vllm_config.parallel_config
            self.engine.update_parallel_config(data_parallel_size, original_to_new)
            vllm_config_update_dict = self._build_vllm_config_update_dict(
                parallel_config, new_ep_size, data_parallel_size, original_to_new
            )
            res = self._execute_command_on_workers(
                FaultToleranceRequest(
                    request_id=str(uuid.uuid4()),
                    instruction="scale_down",
                    params={
                        "timeout": timeout,
                        "exclude_ep_ranks": exclude_ep_ranks,
                        "vllm_config_update_dict": vllm_config_update_dict,
                        "coord_store_port": self.parallel_config._coord_store_port,
                    },
                ),
                self.worker_identities,
                timeout=timeout,
            )
            if not res.success:
                return res

        reinit_request = FaultToleranceRequest(
            instruction="reinit_dp_group_on_fault_tolerance",
            request_id=str(uuid.uuid4()),
            params={},
        )
        # Clear stop_busy_loop BEFORE enqueueing so the loop won't
        # immediately re-pause after executing the reinit command.
        self.stop_busy_loop.clear()
        self.cmd_q.put(reinit_request)

        # Poll until busy_loop_paused is cleared (loop resumed) or timeout.
        # If reinit fails, busy_loop_paused stays set → timeout → failure.
        while self.busy_loop_paused.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return FaultToleranceResult(
                    request_id=ft_request.request_id,
                    success=False,
                    reason="Engine did not resume within timeout.",
                )
            time.sleep(0.1)

        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=True,
        )

    def _execute_command_on_workers(
        self,
        ft_request: FaultToleranceRequest,
        target_worker_sentinels: list[bytes],
        timeout: int = 5,
    ) -> FaultToleranceResult:
        request_bytes = msgspec.msgpack.encode(ft_request)
        for identity in target_worker_sentinels:
            self.worker_cmd_socket.send_multipart([identity, b"", request_bytes])

        results: dict[bytes, FaultToleranceResult] = {}
        pending = set(target_worker_sentinels)
        deadline = time.monotonic() + timeout

        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            events = dict(self.worker_cmd_poller.poll(timeout=int(remaining * 1000)))
            if self.worker_cmd_socket not in events:
                continue

            identity, _, msg = self.worker_cmd_socket.recv_multipart()

            res = msgspec.msgpack.decode(msg, type=FaultToleranceResult)

            # Only consider responses that match the current request ID.
            if identity not in pending or res.request_id != ft_request.request_id:
                continue

            results[identity] = res
            pending.remove(identity)

        # For any workers that did not respond within the timeout, mark them as failed.
        for identity in pending:
            results[identity] = FaultToleranceResult(
                request_id=ft_request.request_id,
                success=False,
                reason=f"did not respond within {timeout}s",
            )

        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=all(result.success for result in results.values()),
            reason="\n".join(
                f"Worker {identity.decode()}: {result.reason}"
                for identity, result in results.items()
                if not result.success
            )
            or None,
        )

    def check_worker_responsive(self) -> bool:
        # Check if workers are responsive. Should only be called in busy_loop thread.
        try:
            self.engine.model_executor.check_health()
            return True
        except TimeoutError:
            logger.warning("Executor check_health() timeout; worker not responsive.")
            return False
        except Exception as err:
            logger.error(
                "Worker health check raised unexpected exception, shutdown the engine."
            )
            raise SystemExit from err

    def shutdown_engine_core(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        shutdown_request = FaultToleranceRequest(
            instruction="shutdown",
            request_id=str(uuid.uuid4()),
            params={},
        )

        self.cmd_q.put(shutdown_request)

        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=True,
            reason=None,
        )

    def shutdown(self):
        close_sockets([self.engine_fault_socket, self.worker_cmd_socket])
        super().shutdown()


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.engine_core_sentinel.busy_loop_paused.clear()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    deadline = (
                        time.monotonic()
                        + self.engine_core_sentinel.engine_recovery_timeout_sec
                    )
                    self.engine_core_sentinel.busy_loop_paused.set()
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised a %s exception.",
                        type(original_exc).__name__,
                    )
                    while (
                        not self.engine_core_sentinel.check_worker_responsive()
                        and time.monotonic() < deadline
                    ):
                        logger.warning(
                            "[BusyLoopWrapper] Worker is not responsive. Checking..."
                        )
                        time.sleep(1)
                    logger.warning(
                        "[BusyLoopWrapper] Engine loop suspended. "
                        "Wait for fault tolerance instructions."
                    )
                    self.engine_core_sentinel.fault_signal_q.put(original_exc)
                    # Put running requests into waiting list.
                    timestamp = time.monotonic()
                    while self.scheduler.running:  # type: ignore[attr-defined]
                        request = self.scheduler.running.pop()  # type: ignore[attr-defined]
                        self.scheduler.preempt_request(request, timestamp)  # type: ignore[attr-defined]
                    self.scheduler.prev_step_scheduled_req_ids.clear()  # type: ignore[attr-defined]
                    if self.batch_queue is not None:
                        self.batch_queue.clear()

                    try:
                        # Block until recovery command received
                        ft_request = self.engine_core_sentinel.cmd_q.get(
                            timeout=max(0, deadline - time.monotonic())
                        )

                        if ft_request is not None:
                            logger.debug(
                                "[BusyLoopWrapper] Received fault tolerance "
                                "command: %s",
                                ft_request.instruction,
                            )
                            method, params = (ft_request.instruction, ft_request.params)
                            if method == "shutdown":
                                sys.exit()
                            run_method(self, method, args=(), kwargs=params)
                        # recovery succeeded; restart the busy loop
                        continue
                    except queue.Empty:
                        # No handling instruction received within predefined
                        # timeout period.
                        logger.error(
                            "[BusyLoopWrapper] Fault tolerance instruction not received"
                            " within timeout. Proceeding with default exception "
                            "handling."
                        )
                    except Exception as cmd_exc:
                        raise RuntimeError(
                            "Fault tolerance execution failed."
                        ) from cmd_exc

                    # Fault tolerance not enabled OR no instruction received
                    # before timeout. Re-raise the original exception
                    # for upper level handling.
                raise original_exc

    return run_with_fault_tolerance
