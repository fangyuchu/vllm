# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

def get_mapping(original_list, to_remove) -> tuple[dict, list]:
    remaining = [num for num in original_list if num not in to_remove]
    original_to_new_dp_rank = {
        original_num: new_index for new_index, original_num in enumerate(remaining)
    }
    new_list = list(original_to_new_dp_rank.values())

    return original_to_new_dp_rank, new_list