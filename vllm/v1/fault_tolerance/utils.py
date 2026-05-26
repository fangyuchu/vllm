# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


class EngineStatus(enum.Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class FaultToleranceRequest:
    request_id: str
    instruction: str  # "retry" | "scale_down"
    params: dict[str, Any] | None = None
