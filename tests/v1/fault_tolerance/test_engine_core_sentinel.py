# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import socket
import time

import pytest
import zmq
from msgspec import msgpack

from vllm.config import (
    DeviceConfig,
    FaultToleranceConfig,
    ParallelConfig,
    VllmConfig,
)
from vllm.v1.fault_tolerance import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    port = _find_free_port()
    return {
        "engine_fault_socket_addr": f"tcp://127.0.0.1:{port}",
    }


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        device_config=DeviceConfig(device="cpu"),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        vllm_config=vllm_cfg,
    )


pytestmark = pytest.mark.skip_global_cleanup


def test_engine_core_sentinel_initialization(addr_dict):
    fault_signal_q: queue.Queue = queue.Queue()

    sentinel = create_engine_core_sentinel(fault_signal_q, addr_dict)

    assert sentinel.engine_index == 0
    assert sentinel.identity == b"engine_sentinel_0"
    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict):
    """
    Verify that when the busy loop reports an exception to fault_signal_q,
    EngineCoreSentinel forwards a FaultInfo payload through the engine fault
    socket.
    """
    fault_signal_q: queue.Queue = queue.Queue()
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        fault_signal_q, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to receive fault reports from the sentinel DEALER socket.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    time.sleep(0.1)
    fault_signal_q.put(RuntimeError("test exception"))

    if not engine_fault_receiver.poll(timeout=5000):
        pytest.fail("Timeout waiting for engine fault message from sentinel")

    parts = engine_fault_receiver.recv_multipart()
    assert len(parts) >= 2
    fault_info = msgpack.decode(parts[-1], type=FaultInfo)
    assert fault_info.engine_id == "0"
    assert "test exception" in fault_info.message

    engine_fault_receiver.close()
    sentinel.shutdown()
    ctx.term()
