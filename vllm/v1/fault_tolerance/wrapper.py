# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import EngineStatus

logger = init_logger(__name__)


def fault_tolerance_wrapper(enabled: bool = True):
    """Decorator for DPEngineCoreProc.run_busy_loop.

    Catches exceptions from the busy loop, signals the sentinel, and
    blocks on ft_event until recovery completes or times out.
    """

    def decorator(busy_loop_fn):

        @functools.wraps(busy_loop_fn)
        def wrapper(self, *args, **kwargs):
            if not enabled or not hasattr(self, "_ft_sentinel"):
                return busy_loop_fn(self, *args, **kwargs)

            sentinel = self._ft_sentinel
            timeout = self.vllm_config.parallel_config \
                .fault_tolerance_config.engine_recovery_timeout_sec \
                if hasattr(self, 'vllm_config') else 120

            while True:
                try:
                    busy_loop_fn(self, *args, **kwargs)
                except SystemExit:
                    raise
                except Exception:
                    logger.exception("EngineCore busy loop fault")
                    sentinel.on_fault()
                    if not sentinel.ft_event.wait(timeout=timeout):
                        logger.error(
                            "Engine recovery timed out after %ds", timeout
                        )
                        raise
                    if sentinel.ft_status != EngineStatus.HEALTHY:
                        logger.error(
                            "Engine recovery failed, status=%s",
                            sentinel.ft_status,
                        )
                        raise
                    # status == HEALTHY → continue loop
            raise SystemExit

        return wrapper

    return decorator
