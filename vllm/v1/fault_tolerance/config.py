# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultToleranceConfig:
    """Configuration for the Fault Tolerance framework."""

    enabled: bool = False
    """Enable fault tolerance for DPEngineCoreProc."""
    engine_recovery_timeout_sec: int = 600
    """Maximum time to wait for engine recovery before giving up.
    Must be longer than DeepEP dispatch timeout (~100s) + time for all
    ranks to detect the fault."""
    ft_rpc_timeout_sec: int = 30
    """Maximum seconds to wait for a Worker RPC response when the caller
    uses ``non_block=True``. A healthy Worker responds in milliseconds;
    this timeout catches the case where the Worker is dead so the
    EngineCore can fail fast via ``@fault_tolerant_wrapper``."""
    ft_rpc_timeout_sync_sec: int = 120
    """Like ``ft_rpc_timeout_sec`` but for synchronous RPC calls
    (``non_block=False``, ``timeout=None``)."""
