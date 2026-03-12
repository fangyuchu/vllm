# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import AsyncMock, Mock, patch

import msgspec.msgpack
import pytest
import zmq

from vllm.config import FaultToleranceConfig
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.client_sentinel import ClientSentinel
from vllm.v1.fault_tolerance.utils import FaultInfo, FaultToleranceZmqAddresses


@pytest.fixture
def mock_vllm_config():
    config = Mock()
    config.parallel_config = Mock(
        data_parallel_index=0,
        data_parallel_size=2,
        data_parallel_size_local=2,
        local_engines_only=False,
    )
    config.fault_tolerance_config = FaultToleranceConfig(
        engine_recovery_timeout_sec=10,
        fault_state_pub_topic="fault_state",
    )
    return config


@pytest.fixture
def mock_ft_addresses():
    return FaultToleranceZmqAddresses(
        fault_state_pub_socket_addr="tcp://127.0.0.1:5558",
        engine_fault_socket_addr="tcp://127.0.0.1:5557",
        engine_core_sentinel_identities={0: b"engine_0", 1: b"engine_1"},
    )


@pytest.fixture
def shutdown_callback():
    return AsyncMock()


@pytest.fixture
def client_sentinel(mock_vllm_config, mock_ft_addresses, shutdown_callback):
    fault_receiver_socket = AsyncMock()
    fault_state_pub_socket = AsyncMock()

    with (
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.make_zmq_socket",
            side_effect=[fault_receiver_socket, fault_state_pub_socket],
        ),
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.asyncio.create_task"
        ) as mock_create_task,
    ):
        mock_create_task.return_value = Mock(done=Mock(return_value=False))

        sentinel = ClientSentinel(
            vllm_config=mock_vllm_config,
            fault_tolerance_addresses=mock_ft_addresses,
            shutdown_callback=shutdown_callback,
        )

    sentinel.fault_receiver_socket = fault_receiver_socket
    sentinel.fault_state_pub_socket = fault_state_pub_socket
    return sentinel


@pytest.mark.asyncio
async def test_client_sentinel_initialization(client_sentinel: ClientSentinel):
    assert client_sentinel.engine_status_dict == {
        0: {"status": EngineStatusType.HEALTHY},
        1: {"status": EngineStatusType.HEALTHY},
    }
    assert client_sentinel.start_rank == 0
    assert client_sentinel.fault_receiver_socket is not None
    assert client_sentinel.fault_state_pub_socket is not None


@pytest.mark.asyncio
async def test_pub_engine_status(client_sentinel: ClientSentinel):
    await client_sentinel._pub_engine_status()

    client_sentinel.fault_state_pub_socket.send_multipart.assert_awaited_once()
    frames = client_sentinel.fault_state_pub_socket.send_multipart.await_args.args[0]
    assert frames[0] == b"fault_state"
    decoded_status = msgspec.msgpack.decode(frames[1])
    assert decoded_status == {
        0: {"status": int(EngineStatusType.HEALTHY)},
        1: {"status": int(EngineStatusType.HEALTHY)},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault_type", "expected_status"),
    [
        ("EngineLoopPausedError", EngineStatusType.PAUSED),
        ("EngineDeadError", EngineStatusType.DEAD),
        ("OtherError", EngineStatusType.UNHEALTHY),
    ],
)
async def test_run_updates_status_and_publishes(
    client_sentinel: ClientSentinel, fault_type: str, expected_status: EngineStatusType
):
    fault_info = FaultInfo(engine_id="1", type=fault_type, message="boom")
    client_sentinel.fault_receiver_socket.recv_multipart = AsyncMock(
        side_effect=[
            [b"", b"", msgspec.msgpack.encode(fault_info)],
            zmq.ZMQError("stop"),
        ]
    )

    # Avoid waiting in background timeout logic; only verify scheduling happened.
    with (
        patch.object(client_sentinel, "_shutdown_after_timeout", AsyncMock()),
        patch(
            "vllm.v1.fault_tolerance.client_sentinel.asyncio.create_task"
        ) as mock_create_task,
    ):
        mock_create_task.return_value = Mock(done=Mock(return_value=False))
        await client_sentinel.run()

    assert client_sentinel.engine_status_dict[1]["status"] == expected_status
    client_sentinel.fault_state_pub_socket.send_multipart.assert_awaited_once()
    mock_create_task.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_after_timeout_calls_callback(
    client_sentinel: ClientSentinel, shutdown_callback: AsyncMock
):
    with patch("vllm.v1.fault_tolerance.client_sentinel.asyncio.sleep", AsyncMock()):
        await client_sentinel._shutdown_after_timeout()

    shutdown_callback.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown(client_sentinel: ClientSentinel):
    with (
        patch("vllm.v1.fault_tolerance.client_sentinel.close_sockets") as mock_close,
        patch.object(client_sentinel.ctx_async, "term") as mock_ctx_async_term,
        patch.object(client_sentinel.ctx, "term") as mock_ctx_term,
    ):
        client_sentinel.shutdown()

    mock_close.assert_called_once_with(
        [
            client_sentinel.fault_receiver_socket,
            client_sentinel.fault_state_pub_socket,
        ]
    )
    mock_ctx_async_term.assert_called_once()
    mock_ctx_term.assert_called_once()
    assert client_sentinel.sentinel_dead is True
