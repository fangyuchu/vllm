# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.v1.worker.dp_utils import should_skip_dp_coordination, skip_dp_coordination
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import (
    configure_dp_collective_state,
    sync_cudagraph_and_dp_padding,
)


@pytest.mark.parametrize("rank", [0, 3])
def test_local_capture_preserves_dp_shape_without_groups_or_serving_epochs(rank):
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.PIECEWISE, num_tokens=8, num_reqs=2
    )
    manager = MagicMock()
    manager.dispatch.return_value = descriptor
    state = torch.tensor([0, 0], dtype=torch.int64)
    configure_dp_collective_state(state)
    state.fill_(17)
    try:
        with (
            skip_dp_coordination(),
            patch(
                "vllm.v1.worker.gpu.dp_utils.get_dp_group",
                side_effect=AssertionError("DP group accessed"),
            ),
            patch(
                "vllm.v1.worker.gpu.dp_utils.get_tp_group",
                side_effect=AssertionError("TP group accessed"),
            ),
            patch(
                "vllm.v1.worker.gpu.dp_utils.dist.all_reduce",
                side_effect=AssertionError("DP all-reduce called"),
            ),
        ):
            desc, sync = sync_cudagraph_and_dp_padding(
                manager,
                descriptor,
                num_tokens=7,
                num_reqs=2,
                uniform_token_count=None,
                dp_size=4,
                dp_rank=rank,
                parallel_config=SimpleNamespace(
                    enable_fault_tolerance=True, tensor_parallel_size=2
                ),
            )
            assert desc == descriptor
            assert sync.num_tokens_across_dp.tolist() == [8, 8, 8, 8]
            assert not sync.eager
            assert state.tolist() == [17, 17]
    finally:
        configure_dp_collective_state(None)


def test_local_capture_scope_restores_normal_serving_after_failure():
    normal_group = object()
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=1
    )
    state = torch.zeros(2, dtype=torch.int64)
    configure_dp_collective_state(state)
    try:
        with (
            pytest.raises(RuntimeError, match="capture failed"),
            skip_dp_coordination(),
        ):
            assert should_skip_dp_coordination()
            # A background capture context must not disable a serving thread.
            with ThreadPoolExecutor(max_workers=1) as pool:
                assert not pool.submit(should_skip_dp_coordination).result()
            raise RuntimeError("capture failed")
        assert not should_skip_dp_coordination()
        with (
            patch(
                "vllm.v1.worker.gpu.dp_utils.get_dp_group",
                return_value=SimpleNamespace(cpu_group=normal_group),
            ),
            patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce") as all_reduce,
        ):
            sync_cudagraph_and_dp_padding(
                None,
                descriptor,
                num_tokens=1,
                num_reqs=1,
                uniform_token_count=None,
                dp_size=2,
                dp_rank=0,
                parallel_config=SimpleNamespace(enable_fault_tolerance=False),
            )
        assert all_reduce.call_args.kwargs["group"] is normal_group
        assert state.tolist() == [1, 1]
    finally:
        configure_dp_collective_state(None)


def test_fresh_elastic_group_supports_fault_tolerant_dp_sync():
    # Elastic groups bypass GroupCoordinator.__init__, but must expose the
    # same FT topology state before the first profiling/model step.
    with (
        patch("vllm.distributed.parallel_state._WORLD", None),
        patch(
            "vllm.distributed.stateless_coordinator.get_ip", return_value="127.0.0.1"
        ),
        patch(
            "vllm.distributed.stateless_coordinator._allocate_group_ports",
            return_value=([12301, 12302, 12303], []),
        ),
        patch(
            "vllm.distributed.stateless_coordinator."
            "stateless_init_torch_distributed_process_group"
        ),
        patch("vllm.distributed.stateless_coordinator.StatelessProcessGroup.create"),
    ):
        group = StatelessGroupCoordinator(
            [[0, 1]], 0, "gloo", False, MagicMock(), group_name="dp"
        )

    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=1
    )
    with (
        patch("vllm.v1.worker.gpu.dp_utils.get_dp_group", return_value=group),
        patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce"),
    ):
        sync_cudagraph_and_dp_padding(
            None,
            descriptor,
            num_tokens=1,
            num_reqs=1,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            parallel_config=SimpleNamespace(
                enable_fault_tolerance=True, tensor_parallel_size=1
            ),
        )
