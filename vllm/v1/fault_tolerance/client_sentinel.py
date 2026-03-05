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
from vllm.utils.network_utils import close_sockets, get_open_port, make_zmq_socket
from vllm.v1.engine import EngineCoreOutputs as FTUtilityOutputs
from vllm.v1.engine import EngineStatusType, UtilityOutput
from vllm.v1.fault_tolerance.sentinel import BaseSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
    FaultToleranceResult,
    FaultToleranceZmqAddresses,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult


class ClientSentinel(BaseSentinel):
    def __init__(
        self,
        vllm_config: VllmConfig,
        fault_tolerance_addresses: FaultToleranceZmqAddresses,
        fault_callback: Callable,
        core_engines: list[bytes],
        input_addresses: list[str],
        output_addresses: list[str],
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
        self.input_sockets = [
            make_zmq_socket(
                self.ctx,
                input_address,
                zmq.DEALER,
                identity=b"client_sentinel",
                bind=False,
            )
            for input_address in input_addresses
        ]

        self.output_sockets = [
            make_zmq_socket(self.ctx, output_address, zmq.PUSH, linger=4000)
            for output_address in output_addresses
        ]

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
        self.is_faulted = threading.Event()
        self.engine_status_dict: dict[int, dict[str, EngineStatusType]] = {
            int.from_bytes(core_engine, byteorder="little"): {
                "status": EngineStatusType.HEALTHY
            }
            for core_engine in self.core_engines
        }
        self.engine_status_lock = threading.Lock()

        threading.Thread(
            target=self.run,
            daemon=True,
            name="ClientSentinelFtRequestsLoopThread",
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

    def retry(self, timeout: int = 1, **kwargs) -> bool:
        """Retry engine cores, return True if successful."""

        # 定义内部异步函数，封装所有原异步逻辑
        async def _async_retry():
            with self.engine_status_lock:
                # 先检查是否有DEAD的引擎核心，有则直接返回False
                for engine_status in self.engine_status_dict.values():
                    if engine_status["status"] == EngineStatusType.DEAD:
                        self.logger(
                            "Engine core is dead; retry won't work.",
                            level="warning",
                        )
                        return False

            # 构造重试请求参数
            params = {
                "timeout": timeout,
                "new_stateless_dp_group_port": get_open_port(),
            }
            request = FaultToleranceRequest(
                str(uuid.uuid4()),
                "retry",
                params,
            )
            # 调用异步的广播命令方法
            ft_result = await self._broad_cast_cmd(request, self.core_engines)

            # 更新引擎状态为HEALTHY
            with self.engine_status_lock:
                for index in self.engine_status_dict:
                    self.engine_status_dict[index] = {
                        "status": EngineStatusType.HEALTHY
                    }

            # 重置故障标记
            if ft_result.success:
                self.is_faulted.clear()
                # self._pub_engine_status()  # 如需启用可取消注释

            return ft_result.success

        # 同步执行异步逻辑，添加异常处理保证健壮性
        try:
            # 核心：用asyncio.run执行内部异步函数（Python 3.7+推荐）
            return asyncio.run(_async_retry())
        except Exception as e:
            # 捕获所有异常，记录日志并返回False
            self.logger(f"Retry operation failed with error: {str(e)}", level="error")
            # 异常时可选择重置状态（根据业务需求调整）
            with self.engine_status_lock:
                self.is_faulted.set()  # 标记为故障状态（可选）
            return False

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

        async def _async_pause():
            with self.engine_status_lock:
                alive_engines = await self._get_alive_engines(exclude_engine_index)
            params = {
                "timeout": timeout,
                "soft_pause": kwargs.get("soft_pause", False),
            }
            request = FaultToleranceRequest(str(uuid.uuid4()), "retry", params)
            ft_result = await self._broad_cast_cmd(request, alive_engines)
            with self.engine_status_lock:
                if ft_result.success:
                    for ind, engine_status in self.engine_status_dict.items():
                        if engine_status == EngineStatusType.HEALTHY:
                            self.engine_status_dict[ind] = {
                                "status": EngineStatusType.PAUSED
                            }
            return ft_result.success

        try:
            # 检查是否有正在运行的事件循环
            loop = asyncio.get_running_loop()
            # 如果有，使用create_task + run_until_complete（注意：仅主线程可用）
            task = loop.create_task(_async_pause())
            return loop.run_until_complete(task)
        except RuntimeError:
            # 没有正在运行的循环，使用asyncio.run
            return asyncio.run(_async_pause())
        except Exception as e:
            self.logger(f"Pause operation failed: {str(e)}", level="error")
            return False

    async def _get_alive_engines(self, exclude_engine_index):
        alive_engines = [
            core_engine
            for core_engine in self.core_engines
            if self.engine_status_dict[int.from_bytes(core_engine, byteorder="little")][
                "status"
            ]
            != EngineStatusType.DEAD
            and (
                exclude_engine_index is None
                or int.from_bytes(core_engine, byteorder="little")
                not in exclude_engine_index
            )
        ]
        return alive_engines

    def _pub_engine_status(self, engine_status: EngineStatusType) -> None:
        topic = self.ft_config.fault_state_pub_topic.encode()
        self.fault_state_pub_socket.send_multipart(
            (topic, msgspec.msgpack.encode(engine_status))
        )

    async def _broad_cast_cmd(
        self, ft_request: FaultToleranceRequest, target_engines: list[bytes]
    ) -> FaultToleranceResult:
        coroutines = []
        for core_engine in target_engines:
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

    def run(self):
        """Poll for fault messages and commands, dispatch to handlers."""
        poller = zmq.Poller()
        generic_decoder = MsgpackDecoder()

        for input_socket in self.input_sockets:
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)
        while not self.sentinel_dead:
            # Received fault tolerance command from client.
            # Add corresponding command to the queue.
            for input_socket, _ in poller.poll():
                _, *data_frames = input_socket.recv_multipart(copy=False)
                client_index, call_id, method_name, args = generic_decoder.decode(
                    data_frames
                )
                self.logger(
                    f"####################{method_name}, ####{args}", level="info"
                )
                ft_request = FaultToleranceRequest(**args[0])
                ft_result = self._execute_cmd(ft_request)

                if not ft_result.success:
                    # If we're busy, reply with a busy message.
                    msg = (
                        "System busy, vLLM is executing another fault "
                        "tolerance instruction."
                    )
                    ft_result = FaultToleranceResult(ft_request.request_id, False, msg)
                    self._send_utility_result(client_index, call_id, ft_result)
                self._send_utility_result(client_index, call_id, ft_result)

    def shutdown(self):
        close_sockets(
            [
                self.fault_receiver_socket,
                self.fault_state_pub_socket,
            ]
        )
        super().shutdown()
