# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault-tolerant wrapper and EngineCoreSentinel for the engine core.

The EngineCoreSentinel manages worker communication (ZMQ ROUTER), tracks
pause state, and executes recovery logic. The wrapper decorates the busy loop
to catch faults and delegate recovery to the sentinel.
"""
import queue
import time
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import msgspec.msgpack
import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequestType,
    EngineStatusType,
    UtilityOutput,
)
from vllm.v1.fault_tolerance.utils import (
    FaultToleranceCommand,
    FaultToleranceRequest,
    FaultToleranceResult,
)
from vllm.v1.serial_utils import UtilityResult

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger(__name__)

# Method name used for FT utility requests routed through the input queue.
FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance for a single engine core.

    Responsibilities:
    - Send out-of-band commands to WorkerSentinels via ZMQ ROUTER
    - Track pause state (threading.Event)
    - Execute recovery: preempt requests, pause workers, poll for commands,
      dispatch retry/scale_down, publish health status
    """

    def __init__(self, engine_index: int, worker_cmd_addr: str,
                 parallel_config):
        self.engine_index = engine_index
        self.parallel_config = parallel_config
        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )
        self.worker_cmd_addr = worker_cmd_addr

        # ZMQ ROUTER to send commands to WorkerSentinels
        self.ctx = zmq.Context()
        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=worker_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_identities: list[bytes] = [
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(parallel_config.tensor_parallel_size)
            for pp_rank in range(parallel_config.pipeline_parallel_size)
        ]

        self.paused = threading.Event()

    def send_cmd_to_workers(self, instruction: str,
                            params: dict | None = None):
        """Send a command to all workers and wait for acks."""
        cmd = FaultToleranceCommand(
            instruction=instruction, params=params or {}
        )
        encoded = msgspec.msgpack.encode(cmd)
        for identity in self.worker_identities:
            self.worker_cmd_socket.send_multipart(
                [identity, b"", encoded]
            )
        for _ in self.worker_identities:
            self.worker_cmd_socket.recv_multipart()

    def handle_running_command(self, ft_request: FaultToleranceRequest) -> dict:
        """Handle FT command while engine is still running (not paused).

        Setting paused causes the wrapper to raise EngineLoopPausedError
        on the next busy loop iteration.
        """
        instruction = ft_request.instruction
        if instruction == "pause":
            self.send_cmd_to_workers("pause")
            self.paused.set()
        return {
            "request_id": ft_request.request_id,
            "success": True,
            "reason": None,
        }

    def recover(self, engine: "EngineCoreProc", original_exc: Exception):
        """Execute the full recovery sequence after a fault.

        Returns True if recovery succeeded and the busy loop should restart.
        Raises on unrecoverable failure.
        """
        self.paused.set()
        logger.warning(
            "[FT] Busy loop raised %s. Pausing for recovery.",
            type(original_exc).__name__,
        )

        self._preempt_running_requests(engine)
        self._pause_workers()
        self._publish_unhealthy(engine, original_exc)

        try:
            client_idx, call_id, ft_request = self._poll_for_command(engine)
            logger.info("[FT] Executing: %s", ft_request.instruction)
            result = self._execute_instruction(engine, ft_request)

            self._send_result(engine, client_idx, call_id, result)

            if result.success:
                self._publish_healthy(engine)
                engine.model_executor.drain_stale_responses()
                return True
            else:
                logger.error("[FT] Command failed: %s", result.reason)
        except queue.Empty:
            logger.error(
                "[FT] No recovery command within %ds timeout.",
                self.engine_recovery_timeout_sec,
            )
        except Exception as cmd_exc:
            raise RuntimeError(
                "Fault tolerance execution failed."
            ) from cmd_exc

        raise original_exc

    def shutdown(self):
        self.worker_cmd_socket.close()
        self.ctx.term()

    # ------------------------------------------------------------------
    # Internal recovery steps
    # ------------------------------------------------------------------

    def _preempt_running_requests(self, engine: "EngineCoreProc"):
        timestamp = time.monotonic()
        while engine.scheduler.running:
            request = engine.scheduler.running.pop()
            engine.scheduler.preempt_request(request, timestamp)
        engine.scheduler.prev_step_scheduled_req_ids.clear()
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

    def _pause_workers(self):
        try:
            self.send_cmd_to_workers("pause")
        except Exception:
            logger.exception("[FT] Failed to pause workers")

    def _publish_unhealthy(self, engine: "EngineCoreProc",
                           exc: Exception):
        engine.output_queue.put_nowait((
            0,
            EngineCoreOutputs(
                engine_index=engine.engine_index,
                health_status=EngineStatusType.UNHEALTHY,
                health_message=str(exc),
            ),
        ))

    def _publish_healthy(self, engine: "EngineCoreProc"):
        engine.output_queue.put_nowait((
            0,
            EngineCoreOutputs(
                engine_index=engine.engine_index,
                health_status=EngineStatusType.HEALTHY,
            ),
        ))

    def _send_result(self, engine: "EngineCoreProc", client_idx: int,
                     call_id: int, result: FaultToleranceResult):
        output = UtilityOutput(call_id)
        output.result = UtilityResult({
            "request_id": result.request_id,
            "success": result.success,
            "reason": result.reason,
        })
        engine.output_queue.put_nowait((
            client_idx,
            EngineCoreOutputs(utility_output=output),
        ))

    def _poll_for_command(
        self, engine: "EngineCoreProc"
    ) -> tuple[int, int, FaultToleranceRequest]:
        """Poll input queue for an FT utility request.

        Non-FT requests are re-queued. Blocks up to engine_recovery_timeout_sec.
        """
        deadline = time.monotonic() + self.engine_recovery_timeout_sec

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty("FT command timeout")

            try:
                req = engine.input_queue.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue

            request_type, request_data = req

            if request_type == EngineCoreRequestType.UTILITY:
                client_idx, call_id, method_name, args = request_data
                if method_name == FT_UTILITY_METHOD:
                    ft_request = args[0]
                    if not isinstance(ft_request, FaultToleranceRequest):
                        ft_request = FaultToleranceRequest(**ft_request)
                    return client_idx, call_id, ft_request

            engine.input_queue.put_nowait(req)

    def _execute_instruction(
        self, engine: "EngineCoreProc", ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        """Dispatch and execute an FT instruction."""
        instruction = ft_request.instruction
        params = ft_request.params

        try:
            if instruction == "retry":
                self._do_retry(engine, params)
            elif instruction == "pause":
                pass
            else:
                return FaultToleranceResult(
                    ft_request.request_id, False,
                    f"Unknown instruction: {instruction}"
                )
            return FaultToleranceResult(ft_request.request_id, True)
        except Exception as e:
            logger.exception("[FT] Instruction '%s' failed", instruction)
            return FaultToleranceResult(
                ft_request.request_id, False, str(e)
            )

    def _do_retry(self, engine: "EngineCoreProc", params: dict):
        """Execute retry: reinit workers and engine DP group."""
        from vllm.distributed.utils import (
            get_cached_tcp_store_client,
            stateless_destroy_torch_distributed_process_group,
        )
        from vllm.utils.network_utils import get_open_port

        parallel_config = engine.vllm_config.parallel_config
        worker_params: dict = {}

        if parallel_config.data_parallel_size > 1:
            # Coordinate worker DP group port via coord store.
            # dp_rank=0 picks a port and publishes; others read the same port.
            store = get_cached_tcp_store_client(
                parallel_config.data_parallel_master_ip,
                parallel_config._coord_store_port,
            )
            key = "ft_worker_dp_port"
            if parallel_config.data_parallel_rank == 0:
                port = get_open_port()
                store.set(key, str(port).encode())
            else:
                port = int(store.get(key).decode())
            worker_params["new_stateless_dp_group_port"] = port

        self.send_cmd_to_workers("retry", worker_params)

        # Reinit the engine-side DP group (only for DPEngineCoreProc).
        if hasattr(engine, "dp_group"):
            stateless_destroy_torch_distributed_process_group(engine.dp_group)
            engine.dp_group, engine.dp_store = (
                parallel_config.stateless_init_dp_group(return_store=True)
            )
            engine.step_counter = 0


def fault_tolerant_wrapper(busy_loop_func: Callable):
    """Wrap the busy loop to catch faults and delegate recovery to sentinel."""

    def run_with_fault_tolerance(self: "EngineCoreProc"):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.ft_sentinel.paused.clear()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as exc:
                if not self.enable_fault_tolerance:
                    raise
                recovered = self.ft_sentinel.recover(self, exc)
                if recovered:
                    continue
                raise

    return run_with_fault_tolerance
