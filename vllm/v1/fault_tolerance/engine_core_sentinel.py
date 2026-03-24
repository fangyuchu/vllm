# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import threading
import time
import traceback

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineStatusType
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
        busy_loop_active: threading.Event,
        engine_input_q: queue.Queue,
        downstream_cmd_addr: str,
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
        self.busy_loop_active = busy_loop_active
        self.engine_input_q = engine_input_q
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.dp_size = vllm_config.parallel_config.data_parallel_size

        self.worker_cmd_socket = make_zmq_socket(
            ctx=self.ctx,
            path=downstream_cmd_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )
        self.worker_cmd_poller = zmq.Poller()
        self.worker_cmd_poller.register(self.worker_cmd_socket, zmq.POLLIN)

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
            self.logger(
                "Detected exception %s: %s\n Call Stack:\n%s",
                type(engine_exception).__name__,
                engine_exception,
                "".join(traceback.format_tb(engine_exception.__traceback__)),
                level="error",
            )
            msg = FaultInfo.from_exception(
                engine_exception, self.engine_index, EngineStatusType.UNHEALTHY
            )
            msg_bytes = msgspec.msgpack.encode(msg)
            self.engine_fault_socket.send_multipart([b"", msg_bytes])
        except queue.Empty:
            pass

    def handle_fault(self, ft_request: FaultToleranceRequest) -> FaultToleranceResult:
        return self._execute_cmd(ft_request)

    # todo: implement pause and retry, check if we need to put wakeup request in
    #  input_queue

    # todo: make this a member in the initialization
    def _get_target_worker_identity(self):
        return {
            f"PP{pp_rank}_TP{tp_rank}".encode()
            for tp_rank in range(self.tp_size)
            for pp_rank in range(self.pp_size)
        }

    # todo implement execute cmd on workers and gather responses mechanism

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
                    self.busy_loop_active.set()
                busy_loop_func(self)
            except SystemExit:
                raise
            except Exception as original_exc:
                if self.enable_fault_tolerance:
                    self.busy_loop_active.clear()
                    self.fault_signal_q.put(original_exc)
                    logger.warning(
                        "[BusyLoopWrapper] EngineCore busy loop raised an exception. "
                        "Suspended and waiting for fault tolerance instructions."
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
                raise

    return run_with_fault_tolerance
