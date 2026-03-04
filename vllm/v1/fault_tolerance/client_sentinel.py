# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import queue
import threading
import time
import uuid
from collections.abc import Callable

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
    FaultToleranceZmqAddresses,
)


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        fault_callback: Callable,
        core_engines: list[bytes],
    ):
        self.core_engines = core_engines
        self.fault_callback = fault_callback
        super().__init__(
            upstream_cmd_addr=None,
            downstream_cmd_addr=fault_tolerance_addresses.engine_core_sentinel_cmd_addr,
            sentinel_identity=None,
            sentinel_tag=None,
            vllm_config=vllm_config,
        )

        self.fault_receiver_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.engine_fault_socket_addr,
            socket_type=zmq.ROUTER,
            bind=True,
        )

        self.fault_state_pub_socket = make_zmq_socket(
            ctx=self.ctx,
            path=fault_tolerance_addresses.fault_state_pub_socket_addr,
            socket_type=zmq.PUB,
            bind=True,
        )

        threading.Thread(
            target=self.run, daemon=True, name="ClientSentinelCmdAndFaultReceiverThread"
        ).start()

        threading.Thread(
            target=self._process_ft_requests_loop,
            daemon=True,
            name="ClientSentinelFtRequestsLoopThread",
        ).start()

    def _process_ft_requests_loop(self) -> None:
        """
        Worker loop to process Fault Tolerance (FT) requests
        """
        try:
            while not self.sentinel_dead:
                try:
                    identity, ft_request = self.ft_request_queue.get(timeout=1)
                    ft_result = self._execute_cmd(ft_request)
                    self.ft_result_queue.put((identity, ft_result))
                    self.inproc_res_send_socket.send(msgspec.msgpack.encode(ft_result))
                except queue.Empty:
                    pass
        except zmq.ZMQError:
            # Socket is closed.
            pass

    async def retry(self, timeout: int = 1, **kwargs) -> bool:
        retry_request = FaultToleranceRequest(
            request_id=str(uuid.uuid4()),
            instruction="retry",
            params={
                "timeout": timeout,
            },
        )
        ft_result = await self._broad_cast_cmd(retry_request)
        return ft_result.success

    def pause(self, timeout: int = 1, **kwargs) -> bool:
        """Pause engine cores, return True if successful. Best-effort operation."""
        self.logger(
            "Pause operation is best-effort only. Due to the complexity of "
            "collective communications (e.g., timing dependencies and "
            "synchronization barriers), pausing may not always succeed. If "
            "the process remains unresponsive or collective operations "
            "cannot be interrupted, consider shutting down and restarting "
            "the instance.",
            level="warning",
        )
        exclude_engine_index = kwargs.get("exclude_engine_index")
        soft_pause = kwargs.get("soft_pause", False)
        with self.engine_status_lock:
            alive_engines = {
                identity
                for index, identity in self.engine_core_sentinel_identities.items()
                if self.engine_status_dict[index]["status"] != EngineStatusType.DEAD
                and (exclude_engine_index is None or index not in exclude_engine_index)
            }
        success, responses = self._execute_command_on_downstreams(
            "pause",
            alive_engines,
            timeout=timeout,
            soft_pause=soft_pause,
        )
        identity_to_index = {
            identity: index
            for index, identity in self.engine_core_sentinel_identities.items()
        }
        with self.engine_status_lock:
            for engine_identity, ft_result in responses.items():
                if ft_result.success:
                    i = identity_to_index[engine_identity]
                    engine_status = self.engine_status_dict[i]["status"]
                    if engine_status == EngineStatusType.HEALTHY:
                        self.engine_status_dict[i] = {"status": EngineStatusType.PAUSED}
        return success

    def _pub_engine_status(self, engine_status: EngineStatusType) -> None:
        topic = self.ft_config.fault_state_pub_topic.encode()
        self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    async def _broad_cast_cmd(
        self, ft_request: FaultToleranceRequest
    ) -> FaultToleranceResult:
        coroutines = []
        for core_engine in self.core_engines:
            coro = self.fault_callback("handle_fault", ft_request, engine=core_engine)
            coroutines.append(coro)
        results = await asyncio.gather(*coroutines)
        return FaultToleranceResult(ft_request.request_id, all(results), reason=None)

    def handle_ft_command(self, ft_request: FaultToleranceRequest) -> bool:
        return self._execute_cmd(ft_request).success

    def _alert_and_pause(self):
        """Receive fault info from engine and pause engines if first fault."""
        try:
            identity, _, message = self.fault_receiver_socket.recv_multipart()
            fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
            engine_status = (
                EngineStatusType.DEAD
                if "dead" in fault_info.type
                else EngineStatusType.UNHEALTHY
            )
            with self.engine_status_lock:
                self.engine_status_dict[int(fault_info.engine_id)] = {
                    "status": engine_status
                }
            self._pub_engine_status()
            if not self.is_faulted.is_set():
                self.is_faulted.set()
                pause_request = FaultToleranceRequest(
                    request_id=str(uuid.uuid4()),
                    instruction="pause",
                    params={
                        "timeout": self.ft_config.gloo_comm_timeout + 5,
                        "soft_pause": False,
                    },
                )
                while not self._dispatch_fault_tolerance_request(pause_request, None):
                    # If the queue is full, it means another fault tolerance
                    # command is being executed.
                    # Wait and retry until we can add the pause command to
                    # the queue.
                    time.sleep(0.1)

        except zmq.ZMQError:
            # Socket was closed during polling, exit loop.
            self.logger("Fault receiver socket closed, stopping thread.", level="info")
            raise

    def _dispatch_fault_tolerance_request(
        self, ft_request: FaultToleranceRequest, identity: bytes | None
    ) -> bool:
        """Add fault tolerance request to queue, return False if busy."""
        try:
            self.ft_request_queue.put_nowait((identity, ft_request))
        except queue.Full:
            return False
        return True

    def shutdown(self):
        close_sockets(
            [
                self.fault_receiver_socket,
                self.fault_state_pub_socket,
            ]
        )
        super().shutdown()
