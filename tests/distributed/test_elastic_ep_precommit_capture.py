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


def test_existing_engine_waits_for_precommit_capture_before_commit():
    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
    state.old_dp_group = object()
    state.reconfig_request = SimpleNamespace(operation_id="scale-1")
    state._sync_kv_cache_memory_size = MagicMock(return_value=True)
    state._uses_precommit_graph_capture = MagicMock(return_value=True)
    state._execute_async = MagicMock(side_effect=[False, True])
    state._mark_ready_for_switch = MagicMock()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.CAPTURE_NEW_RANKS
    state._mark_ready_for_switch.assert_not_called()

    assert not state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.CAPTURE_NEW_RANKS
    state._mark_ready_for_switch.assert_not_called()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state._execute_async.assert_called_with(
        "run_new_rank_capture_companion",
        "scale-1",
    )
    state._mark_ready_for_switch.assert_called_once_with()


def test_existing_engine_without_fast_capture_keeps_normal_transition():
    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
    state.old_dp_group = object()
    state._sync_kv_cache_memory_size = MagicMock(return_value=True)
    state._uses_precommit_graph_capture = MagicMock(return_value=False)
    state._mark_ready_for_switch = MagicMock()

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state._mark_ready_for_switch.assert_called_once_with()


def test_ready_keys_use_dense_ranks_after_fault_tolerance_scale_down():
    keys = []
    for old_rank, new_rank in [(0, 0), (1, 1), (3, 2), (3, 3)]:
        state = object.__new__(ElasticEPScalingState)
        engine_core = SimpleNamespace(dp_rank=old_rank)
        state.engine_core_ref = lambda engine_core=engine_core: engine_core
        state.new_parallel_config = SimpleNamespace(data_parallel_rank=new_rank)
        keys.append(state.ready_key)

    assert keys == ["eep_ready/0", "eep_ready/1", "eep_ready/2", "eep_ready/3"]


def _commit_state_after_fault(old_rank, new_rank, new_size=4):
    from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    parallel_config = SimpleNamespace(
        data_parallel_rank=old_rank,
        data_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
    )
    engine = SimpleNamespace(
        engine_index=old_rank,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        output_queue=Queue(),
    )
    sentinel = engine.ft_sentinel = EngineCoreSentinel(engine, parallel_config)
    sentinel._recover_and_vote = MagicMock()
    sentinel.scale_down(FaultToleranceRequest("scale_down", {"removed_dp_ranks": [2]}))
    engine.output_queue.get_nowait()
    sentinel._dp_reinit_epoch = 1

    state = object.__new__(ElasticEPScalingState)
    state.engine_core_ref = lambda: engine
    state.vllm_config = engine.vllm_config
    state.scale_type = "scale_up"
    state.old_dp_group = object()
    state.state = ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state.commit_requested = True
    state.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_size,
        new_data_parallel_rank=new_rank,
        new_data_parallel_rank_local=0,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29860,
        new_data_parallel_master_port_list=[],
        coord_store_port=29861,
    )
    state._commit_new_dp_group = MagicMock()
    state._collective_rpc = MagicMock()
    state._send_reconfigure_finished = MagicMock()
    return state, sentinel


@pytest.mark.parametrize(
    "old_rank,new_rank,new_size,removed_rank",
    [(3, 2, 4, 3), (0, 0, 4, 2), (3, 2, 5, 4)],
)
def test_scale_up_rebases_fault_state_for_the_next_failure(
    old_rank, new_rank, new_size, removed_rank
):
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(old_rank, new_rank, new_size)
    assert state._progress_existing_engine()
    request = FaultToleranceRequest("scale_down", {"removed_dp_ranks": [removed_rank]})
    sentinel.scale_down(request)

    # A replacement may reuse a previously dead rank. Only this cycle's
    # failure belongs in the worker recovery request.
    assert request.params["dead_dp_ranks"] == [removed_rank]
    assert sentinel._dp_reinit_epoch == 0
    _, output = sentinel.engine.output_queue.get_nowait()
    assert output.engine_index == old_rank
    assert output.utility_output.result.result == {
        "id": old_rank,
        "dp_rank": new_rank,
        "status": "healthy",
    }


def test_failed_scale_up_keeps_fault_state():
    state, sentinel = _commit_state_after_fault(3, 2)
    state._collective_rpc.side_effect = RuntimeError("worker commit failed")
    with pytest.raises(RuntimeError, match="worker commit failed"):
        state._progress_existing_engine()
    assert sentinel._dead_dp_ranks == {2}
    assert sentinel._dp_reinit_epoch == 1
    assert sentinel.parallel_config.data_parallel_rank == 3
    assert sentinel.engine.output_queue.empty()


def test_scale_up_does_not_allow_removing_the_survivor_at_its_new_rank():
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(3, 2)
    state._progress_existing_engine()
    request = FaultToleranceRequest("scale_down", {"removed_dp_ranks": [2]})
    with pytest.raises(ValueError, match=r"dp_rank=2, dead_dp_ranks=\[\]"):
        sentinel.scale_down(request)
