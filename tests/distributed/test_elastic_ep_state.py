# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleUpExistingEngineState,
)


def _commit_state_after_fault(new_size=4):
    from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel

    parallel_config = SimpleNamespace(
        data_parallel_rank=3,
        data_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_rank=0,
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
    )
    engine = SimpleNamespace(
        engine_index=3,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        output_queue=Queue(),
        dp_store=MagicMock(),
        model_executor=SimpleNamespace(failure_callback=None),
    )
    sentinel = engine.ft_sentinel = EngineCoreSentinel(engine, parallel_config)
    sentinel._recover_and_vote = MagicMock()
    sentinel._dead_dp_ranks = {2}

    state = object.__new__(ElasticEPScalingState)
    state.engine_core_ref = lambda: engine
    state.vllm_config = engine.vllm_config
    state.scale_type = "scale_up"
    state.old_dp_group = object()
    state.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state.commit_requested = True
    state.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_size,
        new_data_parallel_rank=3,
        new_data_parallel_rank_local=0,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29860,
        new_data_parallel_master_port_list=[],
        coord_store_port=29861,
        existing_dp_ranks=[0, 1, 3],
        data_parallel_master_rank=0,
    )
    state._commit_new_dp_group = MagicMock()
    state._collective_rpc = MagicMock()
    state._send_reconfigure_finished = MagicMock()
    return state, sentinel


@pytest.mark.parametrize("new_size,removed_rank", [(4, 2), (5, 4)])
def test_scale_up_rebases_fault_state_for_the_next_failure(new_size, removed_rank):
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(new_size)
    assert state._progress_existing_engine()
    request = FaultToleranceRequest(
        "scale_down", {"removed_dp_ranks": [removed_rank]}, request_id="shrink-2"
    )
    sentinel.scale_down(request)

    # A replacement may reuse a previously dead rank. Only this cycle's
    # failure belongs in the worker recovery request.
    assert request.params["dead_dp_ranks"] == [removed_rank]
    _, output = sentinel.engine.output_queue.get_nowait()
    assert output.engine_index == 3
    assert output.utility_output.result.result == {
        "id": 3,
        "status": "healthy",
    }


def test_failed_scale_up_keeps_fault_state():
    state, sentinel = _commit_state_after_fault()
    state._collective_rpc.side_effect = RuntimeError("worker commit failed")
    with pytest.raises(RuntimeError, match="worker commit failed"):
        state._progress_existing_engine()
    assert sentinel._dead_dp_ranks == {2}
    assert sentinel.parallel_config.data_parallel_rank == 3
    assert sentinel.engine.output_queue.empty()
