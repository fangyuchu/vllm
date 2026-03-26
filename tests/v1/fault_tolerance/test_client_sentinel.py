# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq

from vllm.config import FaultToleranceConfig, VllmConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import (
    FaultInfo,
    FaultToleranceRequest,
)

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def mock_vllm_config():
    """Create mock VllmConfig object"""
    config = Mock(spec=VllmConfig)
    config.parallel_config = Mock(
        data_parallel_index=0,
        data_parallel_size=2,
        data_parallel_size_local=2,
        local_engines_only=False,
    )
    config.fault_tolerance_config = FaultToleranceConfig(engine_recovery_timeout_sec=10)
    return config


@pytest.fixture
def mock_ft_addresses():
    """Create mock FaultToleranceZmqAddresses object"""
    addresses = Mock()
    addresses.ft_request_addresses = ["tcp://127.0.0.1:5555"]
    addresses.ft_result_addresses = ["tcp://127.0.0.1:5556"]
    addresses.engine_fault_socket_addr = "tcp://127.0.0.1:5557"
    addresses.fault_state_pub_socket_addr = "tcp://127.0.0.1:5558"
    addresses.ft_config = Mock(fault_state_pub_topic="vllm_fault")
    return addresses


@pytest.fixture
def mock_call_utility_async():
    """Create mock call_utility_async function"""
    return AsyncMock(
        return_value={"request_id": "request_id", "success": True, "reason": None}
    )


@pytest.fixture
def client_sentinel(mock_vllm_config, mock_ft_addresses, mock_call_utility_async):
    """Fixed ClientSentinel fixture (mock Poller)"""
    # 1. Mock Poller class and return mock Poller object
    mock_poller = Mock()
    mock_poller.register = Mock()
    mock_poller.poll = AsyncMock(return_value=[])  # Return empty events by default
    mock_poller_class = Mock(return_value=mock_poller)

    # 2. Mock make_zmq_socket to return mock Socket
    mock_socket = AsyncMock()
    # Add necessary attributes to mock socket (avoid errors in other places)
    mock_socket.fd = Mock(return_value=1)  # Mock file descriptor
    mock_socket.getsockopt = Mock(return_value=0)

    # 3. Batch mock related dependencies
    with (
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.make_zmq_socket",
            return_value=mock_socket,
        ),
        patch("zmq.asyncio.Poller", mock_poller_class),
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.asyncio.create_task"
        ) as mock_create_task,
    ):
        # 4. Disable real async tasks (avoid run/_monitor_and_pause_on_fault execution)
        mock_create_task.return_value = Mock()

        def _capture_task(coro):
            # ClientSentinel starts run() in __init__; close it in tests to avoid
            # "coroutine was never awaited" warnings when create_task is mocked.
            coro.close()
            return Mock()

        mock_create_task.side_effect = _capture_task
        shutdown_callback = AsyncMock()
        sentinel = ClientSentinel(
            vllm_config=mock_vllm_config,
            fault_tolerance_addresses=mock_ft_addresses,
            call_utility_async=mock_call_utility_async,
            core_engines=[b"engine_0", b"engine_1"],
        )

    sentinel.instance_shutdown_callback = shutdown_callback
    return sentinel


# -------------------------- Test Cases --------------------------
@pytest.mark.asyncio
async def test_client_sentinel_initialization(client_sentinel: ClientSentinel):
    """Test ClientSentinel initialization logic."""
    assert client_sentinel.engine_status_dict == {
        0: {"status": "healthy"},
        1: {"status": "healthy"},
    }
    assert client_sentinel.start_rank == 0
    assert client_sentinel.fault_receiver_socket is not None
    assert client_sentinel.fault_state_pub_socket is not None
    # Verify ZMQ sockets are created
    assert len(client_sentinel.ft_request_sockets) == 1
    assert len(client_sentinel.ft_result_sockets) == 1


@pytest.mark.asyncio
async def test_monitor_and_report_on_fault(client_sentinel: ClientSentinel):
    """Fault should update status and publish fault-state report."""
    fault_info = FaultInfo(
        engine_id="0",
        type="EngineDeadError",
        message="dead",
        engine_status=EngineStatusType.DEAD,
    )
    client_sentinel.fault_receiver_socket.recv_multipart = AsyncMock(
        side_effect=[
            [b"", b"", msgspec.msgpack.encode(fault_info)],
            zmq.ZMQError(),
        ]
    )

    await client_sentinel.run()

    assert client_sentinel.engine_status_dict[0]["status"] == "dead"
    client_sentinel.fault_state_pub_socket.send_multipart.assert_awaited_once()

    sent_topic, sent_payload = (
        client_sentinel.fault_state_pub_socket.send_multipart.await_args.args[0]
    )
    assert sent_topic == b"vllm_fault"
    assert msgspec.msgpack.decode(sent_payload) == {
        "total_engines": 2,
        "engines": [{"id": 0, "status": "dead"}, {"id": 1, "status": "healthy"}],
    }


@pytest.mark.asyncio
async def test_retry_success(client_sentinel: ClientSentinel, mock_call_utility_async):
    """Test retry method (success scenario)"""
    # Mock engine to return successful result
    mock_call_utility_async.return_value = {
        "request_id": "request_id",
        "success": True,
        "reason": "success",
    }

    # Execute retry
    result = await client_sentinel.retry(timeout=5)

    # Verify result
    assert result is True
    assert not client_sentinel.is_faulted.is_set()

    # Verify call parameters
    mock_call_utility_async.assert_awaited()
    call_args = mock_call_utility_async.call_args[0]
    assert call_args[0] == "handle_fault"
    assert isinstance(call_args[1], FaultToleranceRequest)
    assert call_args[1].instruction == "retry"
    assert call_args[1].params["timeout"] == 5


@pytest.mark.asyncio
async def test_retry_failure(client_sentinel: ClientSentinel, mock_call_utility_async):
    """Test retry method (failure scenario)"""
    # Mock one engine to return failure
    mock_call_utility_async.side_effect = [
        {"success": True},
        {"success": False, "reason": "engine dead"},
    ]

    # Mark one engine as DEAD first
    client_sentinel.engine_status_dict[1] = {"status": EngineStatusType.DEAD}

    # Execute retry
    result = await client_sentinel.retry()

    # Verify result
    assert result is False


@pytest.mark.asyncio
async def test_pause_operation(
    client_sentinel: ClientSentinel, mock_call_utility_async
):
    """Test pause method"""
    # Mock all engines to pause successfully
    mock_call_utility_async.return_value = {
        "request_id": "request_id",
        "success": True,
        "reason": None,
    }

    # Execute pause
    result = await client_sentinel.pause(timeout=3, soft_pause=True)

    # Verify result
    assert result is True
    assert client_sentinel.engine_status_dict[0]["status"] == EngineStatusType.PAUSED
    assert client_sentinel.engine_status_dict[1]["status"] == EngineStatusType.PAUSED

    # Verify call parameters
    call_args = mock_call_utility_async.call_args[0][1]
    assert call_args.instruction == "pause"
    assert call_args.params["timeout"] == 3
    assert call_args.params["soft_pause"] is True


@pytest.mark.asyncio
async def test_shutdown(client_sentinel: ClientSentinel):
    """Test shutdown method."""
    with patch("vllm.v1.fault_tolerance.client_sentinel.close_sockets") as mock_close:
        client_sentinel.shutdown()

    mock_close.assert_called_once_with(
        [
            client_sentinel.fault_receiver_socket,
            client_sentinel.fault_state_pub_socket,
        ]
    )
    assert client_sentinel.sentinel_dead is True
