# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from vllm.config import CompilationMode
from vllm.distributed.elastic_ep import elastic_execute
from vllm.distributed.eplb.eplb_state import EplbState, compute_logical_maps
from vllm.v1.engine import ReconfigureRankType


def _executor(monkeypatch, dead_ranks, new_dp_size, tp_size=1):
    old_dp_size, local_experts = 4, 2
    width = old_dp_size * tp_size * local_experts
    mapping = torch.arange(width).remainder(local_experts).repeat(2, 1)
    for rank in dead_ranks:
        start = rank * tp_size * local_experts
        mapping[:, start : start + tp_size * local_experts] = -1
    inverse, counts = compute_logical_maps(mapping, local_experts)
    model = MagicMock()
    model.num_logical_experts = local_experts
    model.num_physical_experts = width
    moe = MagicMock()
    moe.moe_config.num_local_experts = local_experts
    moe._quant_method = SimpleNamespace(wraps_legacy_quant_method=False)
    model.modules.return_value = [moe]
    model_state = SimpleNamespace(
        model=model,
        physical_to_logical_map=mapping,
        logical_to_physical_map=inverse,
        logical_replica_count=counts,
        expert_load_pass=torch.arange(2 * width).reshape(2, width),
        expert_load_window=torch.arange(6 * width).reshape(3, 2, width),
        num_unpadded_tokens_tensors=[],
    )
    capacity = max(old_dp_size, new_dp_size) * tp_size * local_experts
    model_state.physical_to_logical_map_buffer = F.pad(
        mapping, (0, capacity - width), value=-1
    )
    model_state.physical_to_logical_map = model_state.physical_to_logical_map_buffer[
        :, :width
    ]
    model_state.expert_load_pass_buffer = F.pad(
        model_state.expert_load_pass, (0, capacity - width)
    )
    model_state.expert_load_pass = model_state.expert_load_pass_buffer[:, :width]
    eplb = object.__new__(EplbState)
    eplb.model_states = {"model": model_state}
    eplb.device = torch.device("cpu")
    eplb.drain_async = MagicMock()
    eplb.update_communicator = MagicMock()
    eplb._propagate_shared_tensors = MagicMock()
    config = SimpleNamespace(
        data_parallel_size=old_dp_size - len(dead_ranks),
        data_parallel_rank=0,
        tensor_parallel_size=tp_size,
        all2all_backend="allgather_reducescatter",
        eplb_config=SimpleNamespace(num_redundant_experts=width - local_experts),
    )
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            parallel_config=config,
            compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        ),
        model_runner=SimpleNamespace(
            model=model,
            get_model=lambda: model,
            model_config=SimpleNamespace(compute_hash=lambda: "model"),
            eplb_state=eplb,
        ),
    )
    executor = object.__new__(elastic_execute.ElasticEPScalingExecutor)
    executor.worker_ref = lambda: worker
    executor.reconfig_request = SimpleNamespace(
        new_data_parallel_size=new_dp_size,
        new_data_parallel_rank=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_rank_local=ReconfigureRankType.KEEP_CURRENT_RANK,
        new_data_parallel_master_ip="127.0.0.1",
        new_data_parallel_master_port=29500,
        data_parallel_master_rank=0,
        existing_dp_ranks=[
            rank for rank in range(old_dp_size) if rank not in dead_ranks
        ],
    )
    executor._prepared_eplb_communicator = object()
    executor._release_cuda_graphs = MagicMock()
    executor._make_eep_moe_config = MagicMock()
    executor._commit_staged_moe_quant_methods = MagicMock()
    groups = {
        "dp": SimpleNamespace(world_size=old_dp_size, dead_dp_ranks=dead_ranks),
        "ep": SimpleNamespace(world_size=old_dp_size * tp_size),
    }
    retired = tuple(groups.values())

    def replace_groups(**standby):
        groups.update(standby)
        return retired

    monkeypatch.setattr(elastic_execute, "get_dp_group", lambda: groups["dp"])
    monkeypatch.setattr(elastic_execute, "get_ep_group", lambda: groups["ep"])
    monkeypatch.setattr(elastic_execute, "_replace_active_groups", replace_groups)
    monkeypatch.setattr(
        elastic_execute,
        "pop_standby_groups",
        lambda: {
            "dp": SimpleNamespace(world_size=new_dp_size),
            "ep": SimpleNamespace(world_size=new_dp_size * tp_size),
        },
    )
    monkeypatch.setattr(elastic_execute, "get_ep_all2all_manager", MagicMock())
    monkeypatch.setattr(elastic_execute, "is_moe_layer", lambda module: module is moe)
    monkeypatch.setattr(
        elastic_execute, "set_current_vllm_config", lambda _: nullcontext()
    )
    return executor, model_state, retired


@pytest.mark.parametrize(
    "dead_ranks,new_dp_size,tp_size",
    [({2}, 4, 1), ({1, 3}, 3, 1), ({2}, 5, 1), ({2}, 4, 2), (set(), 5, 1)],
)
def test_scale_up_preserves_surviving_experts_and_load_history(
    monkeypatch, dead_ranks, new_dp_size, tp_size
):
    executor, state, retired = _executor(monkeypatch, dead_ranks, new_dp_size, tp_size)
    # Surviving ranks keep their original slots, including holes before them.
    padding = new_dp_size * tp_size * 2 - state.physical_to_logical_map.shape[-1]
    expected_map = F.pad(state.physical_to_logical_map, (0, padding), value=-1)
    expected_pass = F.pad(state.expert_load_pass, (0, padding))
    expected_window = F.pad(state.expert_load_window, (0, padding))
    original = state.physical_to_logical_map.clone()
    broadcast = MagicMock()
    monkeypatch.setattr(elastic_execute, "get_standby_dp_group", MagicMock())
    monkeypatch.setattr(elastic_execute, "broadcast_expert_mapping", broadcast)
    executor.broadcast_expert_mapping()
    torch.testing.assert_close(
        broadcast.call_args.kwargs["physical_to_logical"],
        F.pad(
            original,
            (0, state.physical_to_logical_map_buffer.shape[1] - original.shape[1]),
            value=-1,
        ),
    )
    torch.testing.assert_close(state.physical_to_logical_map, original)

    assert executor.switch_and_prepare() is retired

    torch.testing.assert_close(state.physical_to_logical_map, expected_map)
    torch.testing.assert_close(state.expert_load_pass, expected_pass)
    torch.testing.assert_close(state.expert_load_window, expected_window)
    config = executor.worker.vllm_config.parallel_config
    assert config.data_parallel_size == new_dp_size
    state.model.update_physical_experts_metadata.assert_called_once_with(
        num_physical_experts=new_dp_size * tp_size * 2,
        num_local_physical_experts=2,
    )


@pytest.mark.parametrize("commit_fails", [False, True])
def test_scale_up_resumes_eplb_only_after_successful_commit(monkeypatch, commit_fails):
    executor, _, _ = _executor(monkeypatch, {2}, 4)
    runner = executor.worker.model_runner
    runner.eep_eplb_suppressed = True
    executor.broadcast_expert_mapping = MagicMock()
    executor.warm_and_capture = MagicMock()
    executor._perform_eplb_reshuffle = MagicMock()
    executor._start_group_cleanup = MagicMock()

    if commit_fails:
        executor._perform_eplb_reshuffle.side_effect = RuntimeError("reshuffle failed")
        with pytest.raises(RuntimeError, match="reshuffle failed"):
            executor.commit_scale_up(is_existing_worker=True)
        assert runner.eep_eplb_suppressed
    else:
        executor.commit_scale_up(is_existing_worker=True)
        assert not runner.eep_eplb_suppressed
