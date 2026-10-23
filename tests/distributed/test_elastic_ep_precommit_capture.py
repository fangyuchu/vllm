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


def test_existing_engine_is_ready_without_capture_companion():
    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE
    state.old_dp_group = object()
    state._sync_kv_cache_memory_size = MagicMock(return_value=True)
    state._execute_async = MagicMock(
        side_effect=AssertionError("capture companion started")
    )
    state._mark_ready_for_switch = MagicMock()
    state._commit_new_dp_group = MagicMock()
    state.commit_requested = False
    state.prepare_controller = None

    assert state._progress_existing_engine()
    assert state.state is ScaleUpExistingEngineState.COMMIT_SCALE_UP
    state._mark_ready_for_switch.assert_called_once_with()
    # Publishing local readiness does not switch groups before all new ranks
    # have captured and the frontend requests commit.
    assert not state._progress_existing_engine()
    state._commit_new_dp_group.assert_not_called()
    state._execute_async.assert_not_called()


def test_new_engine_capture_failure_does_not_publish_readiness():
    from vllm.distributed.elastic_ep.elastic_state import ScaleUpNewEngineState

    state = object.__new__(ElasticEPScalingState)
    state.state = ScaleUpNewEngineState.PREPARE
    state.new_dp_group = state.new_dp_store = object()
    state.reconfig_request = SimpleNamespace(operation_id="restore")
    state._collective_rpc = MagicMock(side_effect=RuntimeError("capture failed"))
    state._mark_ready_for_switch = MagicMock()
    with pytest.raises(RuntimeError, match="capture failed"):
        state._progress_new_engine()
    state._collective_rpc.assert_called_once_with(
        "elastic_ep_execute", args=("warmup_new_worker",)
    )
    state._mark_ready_for_switch.assert_not_called()


def _commit_state_after_fault(old_rank, new_rank, new_size=4):
    from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    parallel_config = SimpleNamespace(
        data_parallel_rank=old_rank,
        data_parallel_size=4,
        data_parallel_master_ip="127.0.0.1",
        _data_parallel_master_rank=0,
        fault_tolerance_config=SimpleNamespace(engine_recovery_timeout_sec=300),
    )
    engine = SimpleNamespace(
        engine_index=old_rank,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        output_queue=Queue(),
        dp_store=MagicMock(),
    )
    sentinel = engine.ft_sentinel = EngineCoreSentinel(engine, parallel_config)
    sentinel._recover_and_vote = MagicMock()
    sentinel.scale_down(
        FaultToleranceRequest(
            "scale_down", {"removed_dp_ranks": [2]}, request_id="shrink-1"
        )
    )
    engine.output_queue.get_nowait()

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
        existing_dp_ranks=[0, 1, 3],
        data_parallel_master_rank=0,
    )
    state._commit_new_dp_group = MagicMock()
    state._collective_rpc = MagicMock()
    state._send_reconfigure_finished = MagicMock()
    return state, sentinel


@pytest.mark.parametrize(
    "old_rank,new_rank,new_size,removed_rank",
    [(3, 3, 4, 2), (0, 0, 4, 2), (3, 3, 5, 4)],
)
def test_scale_up_rebases_fault_state_for_the_next_failure(
    old_rank, new_rank, new_size, removed_rank
):
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    state, sentinel = _commit_state_after_fault(old_rank, new_rank, new_size)
    assert state._progress_existing_engine()
    request = FaultToleranceRequest(
        "scale_down", {"removed_dp_ranks": [removed_rank]}, request_id="shrink-2"
    )
    sentinel.scale_down(request)

    # A replacement may reuse a previously dead rank. Only this cycle's
    # failure belongs in the worker recovery request.
    assert request.params["dead_dp_ranks"] == [removed_rank]
    _, output = sentinel.engine.output_queue.get_nowait()
    assert output.engine_index == old_rank
    assert output.utility_output.result.result == {
        "id": old_rank,
        "status": "healthy",
    }


def test_failed_scale_up_keeps_fault_state():
    state, sentinel = _commit_state_after_fault(3, 3)
    state._collective_rpc.side_effect = RuntimeError("worker commit failed")
    with pytest.raises(RuntimeError, match="worker commit failed"):
        state._progress_existing_engine()
    assert sentinel._dead_dp_ranks == {2}
    assert sentinel.parallel_config.data_parallel_rank == 3
    assert sentinel.engine.output_queue.empty()
