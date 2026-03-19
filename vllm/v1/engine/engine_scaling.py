# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.config import VllmConfig


def _calculate_exclude_ep_ranks(
    exclude_dp_ranks: list[int], vllm_config: VllmConfig
) -> list[int]:
    """Calculate excluded EP ranks from excluded DP ranks.

    Each DP rank maps to a range of EP ranks based on tensor parallel size.

    Args:
         exclude_dp_ranks: List of DP ranks to exclude.
         vllm_config: Vllm configuration object; the tensor parallel size is read
             from ``vllm_config.parallel_config.tensor_parallel_size``.

    Returns:
        List[int]: Sorted, deduplicated list of excluded EP ranks.
    """
    tensor_model_parallel_size = vllm_config.parallel_config.tensor_parallel_size
    exclude_ep_ranks: list[int] = []
    for dp_rank in exclude_dp_ranks:
        start = dp_rank * tensor_model_parallel_size
        end = (dp_rank + 1) * tensor_model_parallel_size
        exclude_ep_ranks.extend(range(start, end))

    exclude_ep_ranks = sorted(list(set(exclude_ep_ranks)))
    return exclude_ep_ranks


def parse_exclude_ep_ranks(vllm_config: VllmConfig, exclude_ep_ranks_list: list[int]):
    """Parse excluded DP ranks to calculate scaled-down EP/DP sizes and DP rank mapping.

    Core functionality: When pipeline parallelism is disabled (PP=1), calculate the
    scaled-down Expert Parallel (EP) size, Data Parallel (DP) size based on the list of
    excluded DP ranks, and generate a mapping from old DP ranks to new DP ranks for
    scaling down the cluster.

    Args:
        vllm_config:
            Vllm configuration object containing parallelism settings
        exclude_ep_ranks_list:
            List of DP ranks to exclude/remove from the original DP group

    Raises:
        NotImplementedError: Raised when pipeline parallel size > 1, as scaling down
            does not support pipeline parallelism

    Returns:
        tuple: Contains three elements:
            - new_ep_size: Scaled-down EP parallel size (new_dp_size * tp_size)
            - new_dp_size: Scaled-down DP parallel size (original DP size minus
              number of excluded DP ranks)
            - dp_rank_mapping:
                Dictionary mapping old DP ranks (keys) to new DP ranks (values)
    """
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            "Pipeline parallel is not supported for scaling down."
        )
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    old_dp_size = vllm_config.parallel_config.data_parallel_size

    new_dp_size = old_dp_size - len(exclude_ep_ranks_list)
    new_ep_size = new_dp_size * tp_size
    exclude_dp_ranks = set(exclude_ep_ranks_list)

    dp_rank_mapping = {}
    rank_left = [i for i in range(old_dp_size) if i not in exclude_dp_ranks]
    for i in range(new_dp_size):
        dp_rank_mapping[rank_left[i]] = i
    return new_ep_size, new_dp_size, dp_rank_mapping


def _build_vllm_config_update_dict(
    parallel_config: Any,
    new_ep_size: int,
    data_parallel_size: int,
    rank_mapping: Any,
) -> dict[str, Any]:
    """Build dictionary of VLLM config updates for downstream workers.

    Args:
        parallel_config: Current parallel configuration object
        new_ep_size: New expert parallelism size
        data_parallel_size: New data parallelism size
        rank_mapping: New rank mapping after exclusion

    Returns:
        Dict[str, Any]: VLLM configuration update parameters
    """
    return {
        "ep_world_size": new_ep_size,
        "rank_mapping": rank_mapping,
        "data_parallel_size": data_parallel_size,
        "data_parallel_rank": parallel_config.data_parallel_rank,
        "data_parallel_size_local": parallel_config.data_parallel_size_local,
        "expert_parallel_size": (
            data_parallel_size
            * parallel_config.pipeline_parallel_size
            * parallel_config.tensor_parallel_size
        ),
        "data_parallel_master_port": parallel_config.data_parallel_master_port,
    }
