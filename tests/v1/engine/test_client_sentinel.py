# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import msgspec
import pytest
import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.utils.collection_utils import ThreadSafeDict
from vllm.v1.engine import EngineStatusType
from vllm.v1.engine.core_client import ClientSentinel
from vllm.v1.engine.exceptions import FaultInfo

FAULT_RECEIVER_ADDR = "tcp://127.0.0.1:8844"
FAULT_PUB_ADDR = "tcp://127.0.0.1:8846"
FAULT_PUB_TOPIC = "vllm_fault"


def create_test_thread_safe_dict(initial_data=None):
    if initial_data is None:
        initial_data = {1: {"status": "Healthy"}}

    tsd = ThreadSafeDict()
    if initial_data:
        for k, v in initial_data.items():
            tsd[k] = v
    return tsd


def create_client_sentinel():
    # Build a minimal VllmConfig that enables fault tolerance and has
    # at least 2 data-parallel ranks so tests can address engine 0 and 1.
    parallel = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_master_ip="127.0.0.1",
    )
    vconfig = VllmConfig(
        parallel_config=parallel,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return ClientSentinel(
        vllm_config=vconfig,
        engine_fault_socket_addr=FAULT_RECEIVER_ADDR,
        engine_core_sentinel_identities={
            0: b"engine_identity",
            1: b"engine_identity_1",
        },
        fault_state_pub_socket_addr=FAULT_PUB_ADDR,
    )


def test_client_sentinel_initialization():
    sentinel = create_client_sentinel()

    # New field name for identities and engine_status_dict shape
    assert sentinel.engine_core_sentinel_identities[0] == b"engine_identity"
    assert not sentinel.sentinel_dead

    assert 0 in sentinel.engine_status_dict
    # engine status now stores EngineStatusType enum values
    assert sentinel.engine_status_dict[0]["status"] == EngineStatusType.HEALTHY

    assert sentinel.fault_receiver_socket.type == zmq.ROUTER
    assert sentinel.fault_state_pub_socket.type == zmq.PUB

    sentinel.shutdown()


def test_fault_receiver():
    sentinel = create_client_sentinel()

    def send_test_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"test_sender")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="dead", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode("utf-8")])
        socket.close()
        ctx.term()

    sender_thread = threading.Thread(target=send_test_message, daemon=True)
    sender_thread.start()

    def check_published_message():
        ctx = zmq.Context()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.connect(FAULT_PUB_ADDR)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "vllm_fault")

        if not sub_socket.poll(timeout=2000):  # 2-second timeout
            pytest.fail("Timeout waiting for published message")
        parts = sub_socket.recv_multipart()
        sub_socket.close()
        ctx.term()

        prefix = parts[0]
        data_bytes = parts[1]
        assert prefix == b"vllm_fault"
        parsed = msgspec.msgpack.decode(data_bytes)
        # msgpack-decoded structure maps int engine_id -> {"status": int}
        assert parsed.get(1, {}).get("status") == EngineStatusType.DEAD

    check_thread = threading.Thread(target=check_published_message, daemon=True)
    check_thread.start()

    # Wait a short time for sentinel to process the fault
    time.sleep(0.2)

    # Verify engine_status_dict updated
    assert sentinel.engine_status_dict[1]["status"] == EngineStatusType.DEAD

    sentinel.shutdown()


def test_fault_receiver_unhealthy():
    sentinel = create_client_sentinel()

    def send_unhealthy_message():
        ctx = zmq.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, b"engine_identity")
        socket.connect(FAULT_RECEIVER_ADDR)

        test_fault = FaultInfo(engine_id="1", type="error", message="test error")
        socket.send_multipart([b"", test_fault.serialize().encode()])
        socket.close()
        ctx.term()

    threading.Thread(target=send_unhealthy_message, daemon=True).start()
    time.sleep(0.2)

    assert sentinel.engine_status_dict[1]["status"] == EngineStatusType.UNHEALTHY

    sentinel.shutdown()


def test_shutdown_sentinel():
    sentinel = create_client_sentinel()

    original_fault_sock = sentinel.fault_receiver_socket
    original_pub_sock = sentinel.fault_state_pub_socket
    original_ctx = sentinel.ctx

    sentinel.shutdown()

    assert sentinel.sentinel_dead is True

    with pytest.raises(zmq.ZMQError):
        original_fault_sock.recv()

    with pytest.raises(zmq.ZMQError):
        original_pub_sock.send(b"test")

    assert original_ctx.closed
