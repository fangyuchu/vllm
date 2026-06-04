# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Lazily import engine_core_sentinel to break circular import chain:
#   vllm.config -> vllm.config.parallel -> vllm.v1.fault_tolerance.config (OK)
#   -> vllm.v1.fault_tolerance.__init__ -> engine_core_sentinel -> vllm.v1.engine
#   -> vllm.pooling_params -> vllm.config (CIRCULAR)
# By deferring engine_core_sentinel load, vllm.config finishes initializing first.

from .config import FaultToleranceConfig  # noqa: F401 — used by parallel.py
from .utils import FaultToleranceRequest  # noqa: F401 — used by engine/core.py


def __getattr__(name: str):
    import importlib

    if name in ("EngineCoreSentinel", "fault_tolerant_wrapper"):
        mod = importlib.import_module("vllm.v1.fault_tolerance.engine_core_sentinel")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EngineCoreSentinel",
    "FaultToleranceRequest",
    "FaultToleranceConfig",
    "fault_tolerant_wrapper",
]
