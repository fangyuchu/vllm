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
    EngineStatusType,
)
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

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

        self._preempt_running_requests()

        if was_already_paused:
            # Orchestrator already sent pause — workers are paused.
            self._publish_status(EngineStatusType.PAUSED)
        else:
            # Self-detected fault — need to pause workers ourselves.
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
        from vllm.v1.engine import EngineCoreRequestType

        exclude = ft_request.params.get("exclude_engine_index", [])
        if self.engine_index in exclude:
            return
        self.send_cmd_to_workers("pause")
        self.paused.set()
        self.resumed.clear()
        self._publish_status(EngineStatusType.PAUSED)
        # Unblock the busy loop so it sees the paused state.
        self.engine.input_queue.put(
            (EngineCoreRequestType.WAKEUP, None))

    def _do_retry(self, ft_request: FaultToleranceRequest):
        from vllm.distributed.utils import (
            stateless_destroy_torch_distributed_process_group,
            stateless_init_torch_distributed_process_group,
        )
        from vllm.utils.network_utils import get_open_port

        engine = self.engine
        parallel_config = engine.vllm_config.parallel_config
        worker_params: dict = {}
        engine_dp_port: int | None = None

        if parallel_config.data_parallel_size > 1 and hasattr(
                engine, "dp_store"):
            if parallel_config.data_parallel_rank == 0:
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
            worker_params["new_stateless_dp_group_port"] = worker_port
            engine_dp_port = engine_port

        self.send_cmd_to_workers("retry", worker_params)

        if hasattr(engine, "dp_group"):
            stateless_destroy_torch_distributed_process_group(engine.dp_group)
            if engine_dp_port is not None:
                engine.dp_group, engine.dp_store = (
                    stateless_init_torch_distributed_process_group(
                        parallel_config.data_parallel_master_ip,
                        engine_dp_port,
                        parallel_config.data_parallel_rank,
                        parallel_config.data_parallel_size,
                        backend="gloo",
                        return_store=True,
                    )
                )
            else:
                engine.dp_group, engine.dp_store = (
                    parallel_config.stateless_init_dp_group(return_store=True)
                )
            engine.step_counter = 0

        # Reset engines_running so that the coordinator wakeup mechanism
        # ("FIRST_REQ") properly notifies all engines on the first
        # post-recovery request. Without this, the stale True value
        # causes only one engine to get the request, deadlocking the
        # DP allreduce in _run_ar.
        if hasattr(engine, 'engines_running'):
            engine.engines_running = False

        # Reset the DP coordinator's wave state. During the fault, the
        # coordinator never received a wave_complete so it still thinks
        # engines_running=True. If we don't reset it, the coordinator
        # will ignore the next FIRST_REQ from the client, preventing
        # it from sending START_DP_WAVE to idle engines. This causes
        # the first post-recovery request to hang in _run_ar allreduce
        # because only one engine participates.
        # Rank 0 sends wave_complete for the current wave; the
        # coordinator will set engines_running=False and advance the
        # wave. We also increment current_wave locally to stay in sync.
        if (hasattr(engine, 'has_coordinator') and engine.has_coordinator
                and hasattr(engine, 'current_wave')):
            if hasattr(engine, 'dp_rank') and engine.dp_rank == 0:
                engine.output_queue.put_nowait((
                    -1,
                    EngineCoreOutputs(
                        wave_complete=engine.current_wave),
                ))
            engine.current_wave += 1

        # Drain stale requests from the input queue that arrived during
        # the fault/recovery window.  These requests were dispatched to
        # this engine but the other DP engine never received them, so
        # executing them unilaterally would cause a Gloo allreduce hang.
        # We also re-preempt any requests that slipped into the scheduler
        # while the retry was being processed (from the input_sockets
        # thread running concurrently).
        # Two passes with a small gap to catch stragglers from the ZMQ
        # socket buffer (process_input_sockets runs concurrently and may
        # enqueue a request between passes).
        self._drain_stale_requests()
        time.sleep(0.5)
        self._drain_stale_requests()

        self._publish_status(EngineStatusType.HEALTHY)
        if hasattr(engine.model_executor, 'drain_stale_responses'):
            engine.model_executor.drain_stale_responses()

        # Signal the wrapper to resume the busy loop.
        self.paused.clear()
        self.resumed.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_stale_requests(self):
        """Drain stale ADD requests from both the input queue and scheduler.

        During the fault/recovery window, the API server may have routed
        new requests to this engine.  These arrive via the ZMQ input
        socket thread and land in the input_queue.  Since the peer DP
        engine(s) did NOT receive these requests, executing them would
        cause a unilateral allreduce in _run_ar, hanging for a Gloo
        timeout and triggering another fault cycle.

        We discard ADD requests from the queue and re-preempt anything
        that reached the scheduler.  Non-ADD items (WAKEUP, ABORT, etc.)
        are kept.
        """
        engine = self.engine
        from vllm.v1.engine import EngineCoreRequestType

        kept = []
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

        # Re-preempt any requests that slipped into the scheduler.
        timestamp = time.monotonic()
        while engine.scheduler.running:
            request = engine.scheduler.running.pop()
            engine.scheduler.preempt_request(request, timestamp)
        # Also preempt waiting requests.
        while engine.scheduler.waiting:
            request = engine.scheduler.waiting.pop()
            engine.scheduler.preempt_request(request, timestamp)
        engine.scheduler.prev_step_scheduled_req_ids.clear()
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

        if drained > 0:
            logger.info("[FT] Drained %d stale ADD request(s) from input queue.", drained)

    def _preempt_running_requests(self):
        engine = self.engine
        timestamp = time.monotonic()
        while engine.scheduler.running:
            request = engine.scheduler.running.pop()
            engine.scheduler.preempt_request(request, timestamp)
        engine.scheduler.prev_step_scheduled_req_ids.clear()
        if engine.batch_queue is not None:
            engine.batch_queue.clear()

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
