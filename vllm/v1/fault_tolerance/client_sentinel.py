# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import uuid
from asyncio import AbstractEventLoop
from collections.abc import Callable

import msgspec.msgpack
import zmq

from vllm.config import VllmConfig
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs as FTUtilityOutputs
from vllm.v1.engine import EngineStatusType, UtilityOutput
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
    FaultToleranceZmqAddresses,
)
from vllm.v1.serial_utils import MsgpackEncoder, UtilityResult


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        fault_callback: Callable,
        core_engines: list[bytes],
    ):
        self.core_engines = core_engines
        # This func is call_utility_async from AsyncMPClient
        # todo: change variable name.
        self.fault_callback = fault_callback
        super().__init__(
            upstream_cmd_addr=None,
            downstream_cmd_addr=None,
            sentinel_identity=None,
            sentinel_tag=None,
            vllm_config=vllm_config,
        )

        try:
            self._loop: AbstractEventLoop = asyncio.get_running_loop()
        except RuntimeError as err:
            raise RuntimeError(
                "ClientSentinel must be initialized within an asyncio event loop."
            ) from err
        self.is_faulted = threading.Event()
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

        self._utility_encoder = MsgpackEncoder()
        threading.Thread(
            target=self.run, daemon=True, name="ClientSentinelCmdAndFaultReceiverThread"
        ).start()

        threading.Thread(
            target=self._monitor_and_pause_on_fault,
            daemon=True,
            name="ClientSentinelMonitorThread",
        ).start()

    def _send_utility_result(
        self,
        client_index: int,
        call_id: int,
        result: FaultToleranceResult,
    ) -> None:
        uo = UtilityOutput(call_id=call_id)
        uo.result = UtilityResult(result)

        outputs = FTUtilityOutputs(utility_output=uo)
        buffers = self._utility_encoder.encode(outputs)
        # todo: create output sockets for each client/api server
        self.output_sockets[client_index].send_multipart(buffers, copy=False)

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

    async def pause(self, timeout: int = 1, **kwargs) -> bool:
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
        retry_request = FaultToleranceRequest(
            request_id=str(uuid.uuid4()),
            instruction="retry",
            params={
                "timeout": timeout,
            },
        )
        ft_result = await self._broad_cast_cmd(retry_request)
        return ft_result.success

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

    def _monitor_and_pause_on_fault(self):
        """Receive fault info from engine and pause engines."""
        try:
            while True:
                _, _, message = self.fault_receiver_socket.recv_multipart()
                fault_info = msgspec.msgpack.decode(message, type=FaultInfo)
                engine_status = (
                    EngineStatusType.DEAD
                    if "dead" in fault_info.type
                    else EngineStatusType.UNHEALTHY
                )
                self._pub_engine_status(engine_status)
                if not self.is_faulted.is_set():
                    self.is_faulted.set()
                    # Schedule the pause coroutine safely from this thread.
                    asyncio.run_coroutine_threadsafe(
                        self.pause(timeout=self.ft_config.gloo_comm_timeout + 5),
                        self._loop,
                    )

        except zmq.ZMQError:
            # Socket was closed during polling, exit loop.
            self.logger("Fault receiver socket closed, stopping thread.", level="info")
            raise

    def shutdown(self):
        close_sockets(
            [
                self.fault_receiver_socket,
                self.fault_state_pub_socket,
            ]
        )
        super().shutdown()
