# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import uuid
from typing import Any

import msgspec
import zmq

from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.engine import EngineStatusType
from vllm.v1.utils import get_engine_client_zmq_addr


class FaultToleranceResult(msgspec.Struct):
    request_id: str
    success: bool
    reason: str | None = None


class FaultToleranceRequest(msgspec.Struct):
    instruction: str
    params: dict[str, Any]
    request_id: str = ""


class FaultInfo(msgspec.Struct):
    type: str
    message: str
    engine_id: str
    engine_status: EngineStatusType


def make_engine_down_report_socket(vllm_config):
    zmq_ctx = zmq.Context()
    zmq_addr = get_engine_client_zmq_addr(
        local_only=False,
        host=vllm_config.parallel_config.data_parallel_master_ip,
        port=vllm_config.parallel_config.fault_tolerance_config
        .internal_fault_report_port,
    )
    engine_down_socket = make_zmq_socket(
        ctx=zmq_ctx,
        path=zmq_addr,
        socket_type=zmq.DEALER,
        bind=False,
        identity=str(uuid.uuid4()).encode("utf8"),
    )
    return zmq_ctx, engine_down_socket


def notify_engine_down(engine_down_socket, engine_id):
    fault_info = FaultInfo(
        type="EngineDeadError",
        message="Engine died unexpectedly.",
        engine_id=str(engine_id),
        engine_status=EngineStatusType.DEAD,
    )
    with contextlib.suppress(zmq.ZMQError):
        engine_down_socket.send_multipart(
            [b"", msgspec.msgpack.encode(fault_info)]
        )
