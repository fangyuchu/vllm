# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import (
    override_dp_sync_group,
    sync_cudagraph_and_dp_padding,
)


def test_dp_sync_override_is_scoped_to_context():
    normal_group = object()
    capture_group = object()
    before_all_reduce = MagicMock()
    parallel_config = SimpleNamespace(enable_fault_tolerance=False)
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=1,
        num_reqs=1,
    )

    with (
        patch(
            "vllm.v1.worker.gpu.dp_utils.get_dp_group",
            return_value=SimpleNamespace(cpu_group=normal_group),
        ),
        patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce") as all_reduce,
    ):
        with override_dp_sync_group(capture_group, before_all_reduce):
            sync_cudagraph_and_dp_padding(
                None,
                descriptor,
                num_tokens=1,
                num_reqs=1,
                uniform_token_count=None,
                dp_size=2,
                dp_rank=0,
                parallel_config=parallel_config,
            )
        sync_cudagraph_and_dp_padding(
            None,
            descriptor,
            num_tokens=1,
            num_reqs=1,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            parallel_config=parallel_config,
        )

    before_all_reduce.assert_called_once_with()
    assert all_reduce.call_args_list[0].kwargs["group"] is capture_group
    assert all_reduce.call_args_list[1].kwargs["group"] is normal_group


def test_capture_override_does_not_consult_faulted_serving_groups():
    descriptor = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=1, num_reqs=1
    )
    config = SimpleNamespace(enable_fault_tolerance=True, tensor_parallel_size=2)
    with (
        override_dp_sync_group(object()),
        patch(
            "vllm.v1.worker.gpu.dp_utils.get_dp_group",
            side_effect=AssertionError("serving group accessed"),
        ),
        patch("vllm.v1.worker.gpu.dp_utils.dist.all_reduce"),
        patch("vllm.v1.worker.gpu.dp_utils.dist.barrier") as barrier,
    ):
        sync_cudagraph_and_dp_padding(
            None,
            descriptor,
            num_tokens=1,
            num_reqs=1,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            parallel_config=config,
        )
    barrier.assert_not_called()
