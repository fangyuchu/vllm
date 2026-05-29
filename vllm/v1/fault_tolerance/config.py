# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaultToleranceConfig:
    """Configuration for the Fault Tolerance framework."""

    enabled: bool = False
    """Enable fault tolerance for DPEngineCoreProc."""
    engine_recovery_timeout_sec: int = 120
    """Maximum time to wait for engine recovery before giving up."""
