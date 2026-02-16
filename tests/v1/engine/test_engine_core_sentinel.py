# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import threading

import zmq

from vllm.config import FaultToleranceConfig, ParallelConfig, VllmConfig
from vllm.v1.engine.core import (
    EngineCoreSentinel,
)

CLIENT_CMD_ADDR = "tcp://127.0.0.1:8844"
WORKER_CMD_ADDR = "tcp://127.0.0.1:8845"
ENGINE_FAULT_SOCKET_ADDR = "tcp://127.0.0.1:8846"
SENTINEL_IDENTITY = b"engine_sentinel_0"


def create_engine_core_sentinel(
    fault_signal_q: queue.Queue, busy_loop_active: threading.Event
):
    # Construct a minimal VllmConfig with the required parallel and fault-tolerance
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        fault_tolerance_config=FaultToleranceConfig(enable_fault_tolerance=True),
    )

    return EngineCoreSentinel(
        engine_index=0,
        fault_signal_q=fault_signal_q,
        busy_loop_active=busy_loop_active,
        engine_fault_socket_addr=ENGINE_FAULT_SOCKET_ADDR,
        sentinel_identity=SENTINEL_IDENTITY,
        vllm_config=vllm_cfg,
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
