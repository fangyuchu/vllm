# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel scale_down coordinate handling.

removed_dp_ranks uses current (densified) coordinates; the engine translates
them to original coordinates via _alive_dp_ranks and passes workers the
cumulative dead set so masks survive across recovery rounds.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest


def _make_sentinel(dp_size: int, dp_rank: int) -> EngineCoreSentinel:
    parallel_config = SimpleNamespace(
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        data_parallel_master_ip="127.0.0.1",
        tensor_parallel_size=1,
        enable_eplb=True,
        eplb_config=SimpleNamespace(num_redundant_experts=8),
        fault_tolerance_config=SimpleNamespace(
            engine_recovery_timeout_sec=120, auto_recovery=False
        ),
    )
    engine = SimpleNamespace(
        engine_index=dp_rank,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
    )
    return EngineCoreSentinel(engine, parallel_config)


def _run_command(
    sentinel: EngineCoreSentinel, instruction: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Run an FT instruction with the reinit/dispatch step stubbed out;
    returns the params that would have been dispatched to workers."""
    captured = {}

    def fake_reinit(ft_request, **kwargs):
        captured.update(ft_request.params)
        return None

    sentinel._reinit_dp_and_dispatch_command = fake_reinit
    getattr(sentinel, instruction)(
        FaultToleranceRequest(instruction=instruction, params=params)
    )
    return captured


def _commit(sentinel: EngineCoreSentinel, dp_size: int, dp_rank: int):
    """Mirror the topology commit done by the real reinit path."""
    pc = sentinel.parallel_config
    pc.data_parallel_size = dp_size
    pc.data_parallel_rank = dp_rank


def test_scale_down_densifies_across_rounds():
    """Second-round removal where current rank != original rank."""
    # Engine on original rank 3 of [0,1,2,3].
    sentinel = _make_sentinel(dp_size=4, dp_rank=3)

    # Round 1: remove original rank 2 (current == original in round 1).
    params = _run_command(sentinel, "scale_down", {"removed_dp_ranks": [2]})
    assert params["new_dp_size"] == 3
    assert params["new_dp_rank"] == 2
    assert params["dead_dp_ranks"] == [2]
    assert sentinel._alive_dp_ranks == [0, 1, 3]
    _commit(sentinel, dp_size=3, dp_rank=2)

    # Round 2: remove original rank 1, whose current rank is also 1.
    params = _run_command(sentinel, "scale_down", {"removed_dp_ranks": [1]})
    assert params["new_dp_size"] == 2
    assert params["new_dp_rank"] == 1
    # Workers get the cumulative dead set in original coordinates.
    assert params["dead_dp_ranks"] == [1, 2]
    assert sentinel._alive_dp_ranks == [0, 3]


def test_scale_down_rejects_invalid_removed_ranks():
    sentinel = _make_sentinel(dp_size=4, dp_rank=1)
    for removed in ([], [1], [4], [0, 1, 2, 3]):
        with pytest.raises(ValueError, match="Invalid removed_dp_ranks"):
            _run_command(sentinel, "scale_down", {"removed_dp_ranks": removed})
    # Failed attempts leave the mapping untouched.
    assert sentinel._alive_dp_ranks == [0, 1, 2, 3]


def test_dead_dp_ranks_excludes_already_removed():
    sentinel = _make_sentinel(dp_size=4, dp_rank=3)
    sentinel._alive_dp_ranks = [0, 1, 3]  # original rank 2 removed

    # Mask still flags the previously removed EP rank 2 plus newly dead 1.
    assert sentinel._dead_dp_ranks([0, 1, 1, 0]) == [1]
    # Mask with only the already-removed rank means nothing new died.
    assert sentinel._dead_dp_ranks([0, 0, 1, 0]) == []


def test_retry_passes_cumulative_dead_ranks():
    sentinel = _make_sentinel(dp_size=4, dp_rank=3)
    sentinel._alive_dp_ranks = [0, 3]

    params = _run_command(sentinel, "retry", {})
    assert params["dead_dp_ranks"] == [1, 2]


class _FakeStore:
    """Minimal TCPStore stand-in keyed like the mask-exchange protocol."""

    def __init__(self, data: dict[str, bytes]):
        self._data = data

    def set(self, key: str, value: bytes):
        self._data[key] = value

    def get(self, key: str) -> bytes:
        if key not in self._data:
            raise RuntimeError("store timeout")
        return self._data[key]


def test_exchange_masks_uses_original_ep_slots():
    """Rank 0's dead-skip check must map current rank -> original EP slots.

    After densify, current rank 1 is original rank 2, so checking EP slot 1
    (already masked from the earlier round) would wrongly skip a live rank's
    mask and lose the newly dead rank.
    """
    import json

    sentinel = _make_sentinel(dp_size=3, dp_rank=0)
    sentinel._alive_dp_ranks = [0, 2, 3]  # original rank 1 removed
    # Current rank 1 (original 2) reports EP rank 3 newly masked alongside the
    # replayed EP rank 1; current rank 2 (original 3) is dead, never writes.
    sentinel.engine.dp_store = _FakeStore(
        {"ft_mask_0_1": json.dumps([0, 1, 0, 1]).encode()}
    )

    combined = sentinel._exchange_masks([0, 1, 0, 0])
    assert combined == [0, 1, 0, 1]
    assert sentinel._dead_dp_ranks(combined) == [2]
