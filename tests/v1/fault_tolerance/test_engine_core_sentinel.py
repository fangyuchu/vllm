# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import queue
import socket
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest
import zmq
from msgspec import msgpack

from vllm.config import FaultToleranceConfig, ParallelConfig
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.fault_tolerance import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
)
from vllm.v1.utils import get_engine_client_zmq_addr

pytestmark = pytest.mark.skip_global_cleanup


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addr_dict():
    ports = [_find_free_port() for _ in range(3)]
    return {
        "client_cmd_addr": f"tcp://127.0.0.1:{ports[0]}",
        "worker_cmd_addr": f"tcp://127.0.0.1:{ports[1]}",
        "engine_fault_socket_addr": f"tcp://127.0.0.1:{ports[2]}",
    }


@pytest.fixture
def mock_parallel_config():
    """Create mock ParallelConfig object"""
    config = Mock(spec=ParallelConfig)

    config.data_parallel_index = 0
    config.data_parallel_size = 2
    config.data_parallel_size_local = 2
    config.tensor_parallel_size = 1
    config.pipeline_parallel_size = 1
    config.local_engines_only = False

    config.fault_tolerance_config = FaultToleranceConfig(engine_recovery_timeout_sec=10)
    return config


def create_engine_core_sentinel(
    parallel_config: ParallelConfig,
    addr_dict: dict,
    sentinel_identity: bytes = b"engine_sentinel_0",
):
    worker_cmd_addr = get_engine_client_zmq_addr(True, "0.0.0.0")
    input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
    return EngineCoreSentinel(
        parallel_config,
        engine_index=0,
        engine_input_q=input_queue,
        engine_fault_socket_addr=addr_dict["engine_fault_socket_addr"],
        sentinel_identity=sentinel_identity,
        worker_cmd_addr=worker_cmd_addr,
        engine_core=Mock(),
    )


def test_engine_core_sentinel_initialization(addr_dict, mock_parallel_config):
    sentinel = create_engine_core_sentinel(mock_parallel_config, addr_dict)

    assert sentinel.engine_index == 0
    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


def test_busy_loop_exception_forwarded_to_client(addr_dict, mock_parallel_config):
    """
    Verify that when an engine exception is put into fault_signal_q,
    EngineCoreSentinel forwards a FaultInfo message to the
    client-facing engine fault socket.
    """
    sentinel_identity = b"engine_sentinel_0"
    sentinel = create_engine_core_sentinel(
        mock_parallel_config, addr_dict, sentinel_identity=sentinel_identity
    )

    # Bind a ROUTER to the engine_fault_socket_addr to receive the fault report.
    ctx = zmq.Context()
    engine_fault_receiver = ctx.socket(zmq.ROUTER)
    engine_fault_receiver.bind(addr_dict["engine_fault_socket_addr"])

    try:
        time.sleep(0.1)
        sentinel.fault_signal_q.put(RuntimeError("test exception"))
        # Wait for the sentinel to forward the fault to the engine_fault socket.
        if not engine_fault_receiver.poll(timeout=5000):
            pytest.fail("Timeout waiting for engine fault message from sentinel")

        parts = engine_fault_receiver.recv_multipart()
        assert len(parts) >= 2
        fault_info = msgpack.decode(parts[-1], type=FaultInfo)
        assert fault_info.type == "RuntimeError"
        assert fault_info.engine_id == "0"
        assert fault_info.message == "test exception"
    finally:
        engine_fault_receiver.close(linger=0)
        sentinel.shutdown()
        ctx.term()


@pytest.mark.parametrize("dp_size", [1, 2])
def test_retry(mock_parallel_config, addr_dict, dp_size):
    mock_parallel_config.data_parallel_size = dp_size
    sentinel_identity = b"engine_sentinel_0"
    engine_core_sentinel = create_engine_core_sentinel(
        mock_parallel_config, addr_dict, sentinel_identity=sentinel_identity
    )
    engine_core_sentinel.busy_loop_paused.set()
    patch.object(engine_core_sentinel, "_execute_command_on_workers")
    ft_req = FaultToleranceRequest(
        "1", "retry", {"timeout": 2, "coord_store_port": 54321}
    )

    engine_core_sentinel.retry(ft_req)

    assert mock_parallel_config._coord_store_port == 54321
    if dp_size > 1:
        assert not engine_core_sentinel.cmd_q.empty()
        cmd = engine_core_sentinel.cmd_q.get()
        assert cmd.instruction == "reinit_dp_group_on_fault_tolerance"
        assert not engine_core_sentinel.stop_busy_loop.is_set()
    else:
        assert engine_core_sentinel.cmd_q.get() is None
