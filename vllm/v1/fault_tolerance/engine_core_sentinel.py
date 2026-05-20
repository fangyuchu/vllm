# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCoreSentinel and fault_tolerant_wrapper for the engine core.

The EngineCoreSentinel manages worker communication (ZMQ ROUTER), tracks
pause state, and executes recovery logic. The wrapper decorates the busy loop
to catch faults and delegate recovery to the sentinel. All FT state and logic
lives here — EngineCore and Worker hold only a reference to their sentinel.
"""
import queue
import threading
import time
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
from vllm.v1.engine.exceptions import EngineLoopPausedError
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import UtilityResult
from vllm.v1.utils import get_engine_client_zmq_addr

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCoreProc

logger = init_logger(__name__)

FT_UTILITY_METHOD = "handle_fault_tolerance"


class EngineCoreSentinel:
    """Manages fault tolerance state for a single engine core.

    Sends commands to WorkerSentinels via ZMQ ROUTER, tracks pause/resume
    state, and executes FT instructions (pause, retry).
    """

    def __init__(self, engine: "EngineCoreProc", worker_cmd_addr: str,
                 parallel_config):
        self.engine = engine
        self.engine_index = engine.engine_index
        self.parallel_config = parallel_config
        self.engine_recovery_timeout_sec = (
            parallel_config.fault_tolerance_config.engine_recovery_timeout_sec
        )

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
        self.resumed = threading.Event()
        self.resumed.set()
        self.status = EngineStatusType.HEALTHY

    @classmethod
    def create(cls, engine: "EngineCoreProc",
               parallel_config) -> "EngineCoreSentinel":
        """Create sentinel and initialize worker sentinels via collective RPC.

        Call this from EngineCoreProc.__init__ when FT is enabled.
        """
        worker_cmd_addr = get_engine_client_zmq_addr(True, "0.0.0.0")
        sentinel = cls(
            engine=engine,
            worker_cmd_addr=worker_cmd_addr,
            parallel_config=parallel_config,
        )
        engine.model_executor.collective_rpc(
            method="create_worker_sentinel",
            args=(worker_cmd_addr,),
            non_block=False,
        )
        return sentinel

    # ------------------------------------------------------------------
    # Worker communication
    # ------------------------------------------------------------------

    def send_cmd_to_workers(self, instruction: str,
                            params: dict | None = None):
        cmd = FaultToleranceRequest(
            instruction=instruction, params=params or {}
        )
        encoded = msgspec.msgpack.encode(cmd)
        for identity in self.worker_identities:
            self.worker_cmd_socket.send_multipart([identity, b"", encoded])
        for _ in self.worker_identities:
            self.worker_cmd_socket.recv_multipart()

    # ------------------------------------------------------------------
    # Busy-loop guard
    # ------------------------------------------------------------------

    def check_paused(self):
        """Raise if the engine has been paused for FT recovery."""
        if self.paused.is_set():
            raise EngineLoopPausedError("Engine busy loop is paused.")

    # ------------------------------------------------------------------
    # Command dispatch (called from process_input_sockets thread)
    # ------------------------------------------------------------------

    def handle_command(self, client_idx: int, call_id: int, ft_args):
        """Parse, execute an FT command and enqueue the result."""
        if isinstance(ft_args, dict):
            ft_request = FaultToleranceRequest(**ft_args)
        else:
            ft_request = ft_args
        result = self.execute(ft_request)
        uo = UtilityOutput(call_id)
        uo.result = UtilityResult(result)
        self.engine.output_queue.put_nowait(
            (client_idx, EngineCoreOutputs(utility_output=uo)))

    # ------------------------------------------------------------------
    # Fault handling (called by wrapper, runs in busy-loop thread)
    # ------------------------------------------------------------------

    def on_fault(self, exc: Exception):
        """Non-blocking fault initialization. Called by the wrapper when
        the busy loop raises an exception."""
        was_already_paused = self.paused.is_set()
        self.paused.set()
        self.resumed.clear()
        logger.warning("[FT] Busy loop raised %s. Pausing for recovery.",
                       type(exc).__name__)

        self._preempt_running()

        if was_already_paused:
            self._publish_status(EngineStatusType.PAUSED)
        else:
            try:
                self.send_cmd_to_workers("pause")
            except Exception:
                logger.exception("[FT] Failed to pause workers")
            self._publish_status(EngineStatusType.UNHEALTHY, str(exc))

    # ------------------------------------------------------------------
    # Instruction execution (called from process_input_sockets thread)
    # ------------------------------------------------------------------

    def execute(self, ft_request: FaultToleranceRequest) -> dict:
        """Dispatch an FT instruction. Returns a result dict."""
        instruction = ft_request.instruction
        try:
            if instruction == "pause":
                self._do_pause(ft_request)
            elif instruction == "retry":
                self._do_retry(ft_request)
            elif instruction == "status":
                return {
                    "request_id": ft_request.request_id,
                    "success": True,
                    "engine_id": self.engine_index,
                    "status": self.status.name.lower(),
                }
            else:
                return {"request_id": ft_request.request_id, "success": False,
                        "reason": f"Unknown instruction: {instruction}"}
            return {"request_id": ft_request.request_id, "success": True}
        except Exception as e:
            logger.exception("[FT] Instruction '%s' failed", instruction)
            return {"request_id": ft_request.request_id, "success": False,
                    "reason": str(e)}

    def _do_pause(self, ft_request: FaultToleranceRequest):
        exclude = ft_request.params.get("exclude_engine_index", [])
        if self.engine_index in exclude:
            return
        self.send_cmd_to_workers("pause")
        self.paused.set()
        self.resumed.clear()
        self._publish_status(EngineStatusType.PAUSED)
        self.engine.input_queue.put(
            (EngineCoreRequestType.WAKEUP, None))

    def _do_retry(self, ft_request: FaultToleranceRequest):
        worker_params = self._prepare_retry()
        self.send_cmd_to_workers("retry", worker_params)
        self._on_retry()
        self.drain_stale_requests()
        time.sleep(0.5)
        self.drain_stale_requests()
        if hasattr(self.engine.model_executor, 'drain_stale_responses'):
            self.engine.model_executor.drain_stale_responses()

        self._publish_status(EngineStatusType.HEALTHY)
        self.paused.clear()
        self.resumed.set()

    def _prepare_retry(self) -> dict:
        """Reinit DP process group if in DP mode. Returns worker params."""
        engine = self.engine
        if not hasattr(engine, 'dp_group'):
            return {}

        from vllm.distributed import (
            stateless_destroy_torch_distributed_process_group,
        )
        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )
        from vllm.utils.network_utils import get_open_port

        parallel_config = engine.vllm_config.parallel_config

        if engine.dp_rank == 0:
            worker_port = get_open_port()
            engine_port = get_open_port()
            engine.dp_store.set("ft_worker_dp_port",
                                str(worker_port).encode())
            engine.dp_store.set("ft_engine_dp_port",
                                str(engine_port).encode())
        else:
            worker_port = int(
                engine.dp_store.get("ft_worker_dp_port").decode())
            engine_port = int(
                engine.dp_store.get("ft_engine_dp_port").decode())

        stateless_destroy_torch_distributed_process_group(engine.dp_group)
        engine.dp_group, engine.dp_store = (
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

    def _on_retry(self):
        """Reset DP-specific state if in DP mode."""
        engine = self.engine
        if not hasattr(engine, 'dp_group'):
            return

        engine.engines_running = False
        engine.step_counter = 0
        if engine.has_coordinator and engine.dp_rank == 0:
            engine.output_queue.put_nowait((
                -1,
                EngineCoreOutputs(wave_complete=engine.current_wave),
            ))
        engine.current_wave += 1

    # ------------------------------------------------------------------
    # Recovery helpers
    # ------------------------------------------------------------------

    def _preempt_running(self):
        """Preempt running requests and clear batch state."""
        engine = self.engine
        timestamp = time.monotonic()
        while engine.scheduler.running:
            request = engine.scheduler.running.pop()
            engine.scheduler.preempt_request(request, timestamp)
        engine.scheduler.prev_step_scheduled_req_ids.clear()
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

    def drain_stale_requests(self):
        """Drain stale ADD requests from the input queue and preempt
        all scheduler requests (running + waiting)."""
        engine = self.engine
        kept: list[tuple] = []
        drained = 0
        while not engine.input_queue.empty():
            try:
                item = engine.input_queue.get_nowait()
            except queue.Empty:
                break
            req_type, _ = item
            if req_type == EngineCoreRequestType.ADD:
                drained += 1
            else:
                kept.append(item)
        for item in kept:
            engine.input_queue.put_nowait(item)

        timestamp = time.monotonic()
        while engine.scheduler.running:
            request = engine.scheduler.running.pop()
            engine.scheduler.preempt_request(request, timestamp)
        while engine.scheduler.waiting:
            request = engine.scheduler.waiting.pop()
            engine.scheduler.preempt_request(request, timestamp)
        engine.scheduler.prev_step_scheduled_req_ids.clear()
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

        if drained > 0:
            logger.info("[FT] Drained %d stale ADD request(s).", drained)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish_status(self, status: EngineStatusType,
                        message: str | None = None):
        self.status = status
        if message:
            logger.info("[FT] Engine %d status -> %s: %s",
                        self.engine_index, status.name, message)
        else:
            logger.info("[FT] Engine %d status -> %s",
                        self.engine_index, status.name)

    def shutdown(self):
        self.worker_cmd_socket.close()
        self.ctx.term()


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
                    self.ft_sentinel.paused.clear()
                    self.ft_sentinel.resumed.set()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as exc:
                if not self.enable_fault_tolerance:
                    raise
                self.ft_sentinel.on_fault(exc)
                recovered = self.ft_sentinel.resumed.wait(
                    timeout=self.ft_sentinel.engine_recovery_timeout_sec)
                if recovered:
                    continue
                logger.error("[FT] No recovery within %ds timeout.",
                             self.ft_sentinel.engine_recovery_timeout_sec)
                raise

    return run_with_fault_tolerance
