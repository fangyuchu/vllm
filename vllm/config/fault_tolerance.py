# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from vllm.config.utils import config


@config
class FaultToleranceConfig:
    """Configuration for fault tolerance."""

    engine_recovery_timeout_sec: int = 120
    """Timeout (in seconds) to wait for error handling instructions
    before raising an exception. If the EngineCore encounters an
    error, it waits up to this many seconds for vLLM to receive 
    instructions on how to handle the error and then recover from the fault.
    If vLLM does not recover during this time, the original error is raised.
    """

    resume_requests_after_recovery: bool = False
    """Whether in-flight requests are parked instead of aborted on fault.

    If False (default), running requests are aborted immediately when a fault
    occurs. If True, they are preempted back to the waiting queue and resumed
    after a ``retry`` recovery, unless that instruction opts out for this round
    with ``params={"resume": false}``.
    """
