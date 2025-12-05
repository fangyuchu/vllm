# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from abc import abstractmethod

import zmq

from vllm.logger import init_logger
from vllm.utils.network_utils import make_zmq_socket, recv_router_dealer_message
from vllm.v1.engine.utils import broadcast_instruction, wait_for_instruction_result
from vllm.v1.serial_utils import (
    deserialize_method_call,
    run_method,
)

logger = init_logger(__name__)
POLL_TIMEOUT_MS = 100


class BaseLLMSentinel:
    """
    Abstract and constrain the core functionalities of the Sentinel.

    Core functionalities covered:
    - Fault listening
    - Fault tolerance instruction reception
    - Fault tolerance instruction execution
    - Upstream and downstream communication

    This class serves as the base abstraction for all LLM-related Sentinel
    implementations, enforcing standardized fault tolerance behavior across
    the system.
    """

    def __init__(
        self,
        upstream_cmd_addr: str | None,
        downstream_cmd_addr: str | None,
        sentinel_identity: bytes | None,
        sentinel_index: str | None,
    ):
        self.is_sentinel_dead = False
        self.ctx = zmq.Context()
        self.sentinel_index = sentinel_index
        self.sentinel_name = f"{self.__class__.__name__}"
        self.logger = self._make_logger(
            f"{self.__class__.__name__}_{self.sentinel_index}"
        )
        if upstream_cmd_addr is not None and sentinel_identity is not None:
            self.upstream_cmd_socket = make_zmq_socket(
                self.ctx,
                upstream_cmd_addr,
                zmq.DEALER,
                bind=False,
                identity=sentinel_identity,
            )
        if downstream_cmd_addr is not None:
            self.downstream_cmd_socket = make_zmq_socket(
                ctx=self.ctx,
                path=downstream_cmd_addr,
                socket_type=zmq.ROUTER,
                bind=True,
            )

    def _make_logger(self, prefix):
        def log(msg, *args, level="info", **kwargs):
            """
            level: "info", "warning", "error", "debug"
            msg: log message
            """
            getattr(logger, level)(prefix + msg, *args, **kwargs)

        return log

    @abstractmethod
    def run(self) -> None:
        """
        The run() method is typically launched as a separate thread when a Sentinel
        instance is created, and is used for continuous error monitoring and instruction
        reception.

        This background thread runs persistently to ensure real-time detection of errors
        and timely reception of fault tolerance instructions from upstream components
        (e.g., EngineCoreSentinel).
        """
        raise NotImplementedError

    def receive_execute_cmd(self, cmd_str: str | None = None) -> bool:
        try:
            if cmd_str is None:
                has_msg, _, cmd_str = recv_router_dealer_message(
                    self.upstream_cmd_socket,
                    use_poller=True,
                    poll_timeout=POLL_TIMEOUT_MS,
                )
            else:
                has_msg = True
        except zmq.ZMQError:
            self.logger(
                "Socket closed, terminating %s", self.sentinel_name, level="info"
            )
            return False

        if has_msg:
            self.logger("Received cmd: %s", cmd_str, level="info")
            self._execute_cmd(cmd_str)
        return True

    @abstractmethod
    def fault_listener(self) -> bool:
        raise NotImplementedError

    def _execute_cmd(self, cmd_str):
        """
        Execute a command received from ClientSentinel.
        """
        method, method_uuid, method_params = deserialize_method_call(cmd_str)
        self.logger("Executing command: %s", method, level="info")
        try:
            success = run_method(self, method, args=(), kwargs=method_params)
            self.logger("Command (%s) succeeded: %s", method, success, level="info")
            reason = None
        except Exception as e:
            self.logger(
                "Error executing method %s: %s, %s",
                method,
                type(e).__name__,
                e,
                level="error",
            )
            success = False
            reason = f"{type(e).__name__}: {e}"
        self._send_execution_result(success, method_uuid, reason)

    @abstractmethod
    def pause(self, timeout: int = 1, soft_pause: bool = True) -> bool:
        raise NotImplementedError

    @abstractmethod
    def retry(self, new_stateless_dp_group_port: int, timeout: int = 1) -> bool:
        raise NotImplementedError

    def _send_execution_result(
        self, success: bool, method_uuid: str, reason: str | None
    ):
        msg = {
            "engine_index": self.sentinel_index,
            "success": success,
            "method_uuid": method_uuid,
        }
        if not success and reason is not None:
            msg["reason"] = reason
        msg_bytes = json.dumps(msg).encode("utf-8")
        self.upstream_cmd_socket.send_multipart([b"", msg_bytes])

    def _execute_downstream_method(
        self,
        method_name,
        target_downstream_sentinels,
        response_timeout: int = 5,
        **kwargs,
    ):
        method_uuid = broadcast_instruction(
            self.downstream_cmd_socket,
            target_downstream_sentinels,
            method_name,
            **kwargs,
        )

        downstream_sentinel_responses = wait_for_instruction_result(
            self.downstream_cmd_socket,
            target_downstream_sentinels,
            method_name,
            response_timeout,
            method_uuid,
        )

        # check the execution results
        all_success = True
        for sentinel_identity in target_downstream_sentinels:
            response = downstream_sentinel_responses.get(sentinel_identity)

            if response is None:
                self.logger(
                    "EngineCoreSentinel[%s] did not respond"
                    ' to command "%s" within timeout.',
                    sentinel_identity,
                    method_name,
                    level="info",
                )
                all_success = False
            elif not response.get("success", False):
                self.logger(
                    "EngineCoreSentinel[%s] failed to execute "
                    'command "%s" (reason: %s)',
                    sentinel_identity,
                    method_name,
                    response.get("reason", "unknown"),
                    level="error",
                )
                all_success = False
        for sentinel in target_downstream_sentinels:
            response = downstream_sentinel_responses.get(sentinel)
            if response is None or not response.get("success", False):
                return False

        return all_success, downstream_sentinel_responses

    def shutdown(self):
        if self.upstream_cmd_socket is not None:
            self.upstream_cmd_socket.close()
        if self.ctx is not None:
            self.ctx.term()
        self.is_sentinel_dead = True
