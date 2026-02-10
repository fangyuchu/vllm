# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading

import zmq

from vllm.config import FaultToleranceConfig
from vllm.v1.engine.core import (
    EngineCoreSentinel,
)

CLIENT_CMD_ADDR = "tcp://127.0.0.1:8844"
WORKER_CMD_ADDR = "tcp://127.0.0.1:8845"
ENGINE_FAULT_SOCKET_ADDR = "tcp://127.0.0.1:8846"
DEALER_SOCKET_IDENTITY = b"engine_sentinel_0"


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue, busy_loop_active: threading.Event
):
    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        busy_loop_active=busy_loop_active,
        client_cmd_addr=CLIENT_CMD_ADDR,
        worker_cmd_addr=WORKER_CMD_ADDR,
        engine_fault_socket_addr=ENGINE_FAULT_SOCKET_ADDR,
        dealer_socket_identity=DEALER_SOCKET_IDENTITY,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )


def test_engine_core_sentinel_initialization():
    fault_signal_q: queue.Queue = queue.Queue()
    busy_loop_active = threading.Event()

    sentinel = create_engine_core_sentinel(fault_signal_q, busy_loop_active)

    assert sentinel.engine_index == 0
    assert sentinel.tp_size == 1
    assert sentinel.pp_size == 1
    assert sentinel.engine_running is True

    assert sentinel.engine_fault_socket.type == zmq.DEALER

    sentinel.shutdown()


# todo: test the fault reporting of engine core sentinel
