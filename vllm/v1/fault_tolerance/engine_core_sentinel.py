# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback
import uuid

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
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


class EngineCoreSentinel(BaseSentinel):
    """
    EngineCoreSentinel monitors a single EngineCore instance, responsible for:
      1. Receiving fault signals (exceptions raised in EngineCore busy loop)
      2. Receiving and executing commands from ClientSentinel
      3. Reporting execution results or faults back to the ClientSentinel
    """

    def __init__(
        self,
        engine_index: int,
        fault_signal_q: queue.Queue,
        cmd_q: queue.Queue,
        busy_loop_paused: threading.Event,
        stop_busy_loop: threading.Event,
        engine_input_q: queue.Queue,
        engine_fault_socket_addr: str,
        sentinel_identity: bytes,
        vllm_config: VllmConfig,
    ):
        self.engine_index = engine_index
        super().__init__(
            sentinel_tag=f"DP_{engine_index}",
            vllm_config=vllm_config,
            identity=sentinel_identity,
        )

        self.fault_signal_q = fault_signal_q
        self.cmd_q = cmd_q
        self.stop_busy_loop = stop_busy_loop
        self.busy_loop_paused = busy_loop_paused
        self.engine_input_q = engine_input_q
        assert vllm_config.fault_tolerance_config.worker_cmd_addr is not None
        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=vllm_config.fault_tolerance_config.worker_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_cmd_poller = zmq.Poller()
        self.worker_cmd_poller.register(self.worker_cmd_socket, zmq.POLLIN)
        self.worker_identities = [
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(vllm_config.parallel_config.tensor_parallel_size)
            for pp_rank in range(vllm_config.parallel_config.pipeline_parallel_size)
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
            exception = self.fault_signal_q.get(timeout=1)
            self.logger(
                "Detected exception %s: %s\n Call Stack:\n%s",
                type(exception).__name__,
                exception,
                "".join(traceback.format_tb(exception.__traceback__)),
                level="error",
            )
            engine_status = (
                EngineStatusType.PAUSED
                if isinstance(exception, EngineLoopPausedError)
                else EngineStatusType.UNHEALTHY
            )
            msg = FaultInfo.from_exception(exception, self.engine_index, engine_status)
            msg_bytes = msgspec.msgpack.encode(msg)
            self.engine_fault_socket.send_multipart([b"", msg_bytes])
        except queue.Empty:
            pass

    def handle_fault(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        return self._execute_cmd(ft_request)

    def pause(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        """Pause the busy loop of engine core safely."""
        self.logger("Start pausing EngineCore", level="info")
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
        new_stateless_dp_group_port = ft_request.params.get(
            "new_stateless_dp_group_port"
        )
        self._execute_command_on_workers(
            FaultToleranceRequest(str(uuid.uuid4()), "retry", ft_request.params),
            self.worker_identities,
            timeout=timeout,
        )
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            # If the Gloo communication times out,
            # the data parallel group (dp_group) needs to be reinitialized
            reinit_request = FaultToleranceRequest(
                instruction="reinit_dp_group_on_fault_tolerance",
                request_id=str(uuid.uuid4()),
                params={"new_stateless_dp_group_port": new_stateless_dp_group_port},
            )
            self.cmd_q.put(reinit_request)
        else:
            self.cmd_q.put(None)

        self.stop_busy_loop.clear()
        return FaultToleranceResult(
            request_id=ft_request.request_id,
            success=True,
            reason=None if True else "Worker don't recovered within timeout.",
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

    def shutdown(self):
        close_sockets([self.engine_fault_socket, self.worker_cmd_socket])
        super().shutdown()


def busy_loop_wrapper(busy_loop_func):
    """
    Wrap the busy loop function to perform fault tolerance.
    """
    from vllm.v1.engine.core import logger

    def run_with_fault_tolerance(self):
        while True:
            try:
                if self.enable_fault_tolerance:
                    self.busy_loop_paused.clear()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    self.busy_loop_paused.set()
                    self.fault_signal_q.put(original_exc)
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised a %s exception. "
                        "Suspended and waiting for fault tolerance instructions.",
                        type(original_exc).__name__,
                    )
                    # Put running requests into waiting list.
                    timestamp = time.monotonic()
                    while self.scheduler.running:
                        request = self.scheduler.running.pop()
                        self.scheduler.preempt_request(request, timestamp)
                    self.scheduler.prev_step_scheduled_req_ids.clear()
                    if self.batch_queue is not None:
                        self.batch_queue.clear()

                    try:
                        # Block until recovery command received
                        ft_request = self.cmd_q.get(
                            timeout=self.engine_recovery_timeout_sec
                        )

                        if ft_request is not None:
                            logger.debug(
                                "[BusyLoopWrapper] Received fault tolerance "
                                "command: %s",
                                ft_request.instruction,
                            )
                            method, params = (ft_request.instruction, ft_request.params)
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
