# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import threading

from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import (
    EngineStatus,
    FaultToleranceRequest,
)

logger = init_logger(__name__)


class DPEngineCoreSentinel:
    """FT proxy embedded in DPEngineCoreProc.

    Manages FT state and coordinates recovery between the IO thread
    (where handle_ft_method is called) and the busy loop thread
    (where on_fault and ft_event.wait are called).

    Lifecycle:
      busy_loop exception → decorator.on_fault()
        → _ft_status = UNHEALTHY, ft_event.clear()
        → ft_event.wait() blocks

      IO thread receives UTILITY("handle_fault")
        → handle_ft_method() → _do_recovery()
        → _ft_status = HEALTHY, ft_event.set()

      decorator wakes → _ft_status == HEALTHY → continue loop
    """

    # Method names intercepted in process_input_sockets (bypass input_queue)
    FT_METHODS = frozenset({"handle_fault", "get_ft_status"})

    def __init__(self, engine_core: "DPEngineCoreProc"):
        self._engine_core = engine_core
        self._ft_status = EngineStatus.HEALTHY
        # Bridges IO thread (set) and busy loop thread (wait)
        self.ft_event = threading.Event()
        self.ft_event.set()  # initially ready
        # TCPStore reference for DP group reinit; created once and reused
        self._recovery_store: object = None

    @property
    def ft_status(self) -> EngineStatus:
        return self._ft_status

    def on_fault(self) -> None:
        """Called by @fault_tolerance_wrapper on exception."""
        self._ft_status = EngineStatus.UNHEALTHY
        self.ft_event.clear()

    def handle_ft_method(self, method: str, args: tuple) -> object:
        """Called from IO thread (process_input_sockets) for FT methods."""
        if method == "get_ft_status":
            return self._ft_status.value
        elif method == "handle_fault":
            raw = args[0]
            if isinstance(raw, dict):
                ft_request = FaultToleranceRequest(
                    request_id=raw.get("request_id", ""),
                    instruction=raw.get("instruction", ""),
                    params=raw.get("params"),
                )
            else:
                ft_request = raw  # type: FaultToleranceRequest
            self._do_recovery(ft_request)
            return None
        else:
            raise ValueError(f"Unknown FT method: {method}")

    def _do_recovery(self, ft_request: FaultToleranceRequest) -> None:
        """DP-aware recovery executed in IO thread — fast path only.

        Recovery sequence for retry:
        1. Reset DP sync state (break all-reduce hang)
        2. Reinit DP process group (break all-reduce deadlock)
        3. Signal busy loop + respond to client immediately
        4. Reset EngineCore state (scheduler, in-flight requests)
        5. Spawn a daemon thread for best-effort worker recovery via
           collective_rpc("retry"). This is non-blocking because
           retry shares the FIFO MessageQueue with execute_model
           and can be delayed ~minutes — blocking the IO thread here
           would cause HTTP timeouts on handle_fault.
        """
        try:
            engine = self._engine_core

            # 1. Reset DP sync state
            self._clean_dp_state()

            # 2. Reinit DP process group
            self._reinit_dp_group(ft_request.params)

            # 3. Signal busy loop + respond to client immediately.
            #    The DP group reinit is the critical step that breaks
            #    the all-reduce deadlock. Everything else is deferred.
            self._ft_status = EngineStatus.HEALTHY
            self.ft_event.set()

            # 4. Reset EngineCore state
            self._clean_engine_state()

            # 5. Best-effort worker recovery in background thread.
            #    Daemon thread so it does not prevent process exit.
            params = ft_request.params
            threading.Thread(
                target=self._retry_workers,
                args=(params,),
                daemon=True,
            ).start()

        except Exception:
            logger.exception("FT recovery failed")
            self._ft_status = EngineStatus.UNHEALTHY
            self.ft_event.set()

    def _retry_workers(self, params: dict | None) -> None:
        """Best-effort worker recovery via collective_rpc in background thread."""
        try:
            self._engine_core.model_executor.collective_rpc(
                "retry", timeout=15, args=(params,)
            )
            logger.info("Worker recovery via retry succeeded.")
        except Exception:
            logger.warning(
                "collective_rpc retry failed (non-fatal), "
                "worker runtime state will be reset on next step"
            )

    def _clean_dp_state(self) -> None:
        """Reset DP-specific state for a clean restart.

        IMPORTANT: Do NOT call dp_group.shutdown() or any collective
        operation on the stale dp_group. When a peer crashes:
          - sync_dp_state() all-reduce times out
          - Gloo's pg.shutdown() is also a collective barrier → would hang

        The safest approach is to abandon the reference. The old
        ProcessGroup will be garbage collected. dp_store (TCPStore)
        is pure key-value so dropping it is safe.
        """
        engine = self._engine_core
        engine.pending_pause = False
        engine.ignore_start_dp_wave = False
        engine.engines_running = False
        engine.step_counter = 0
        # DO NOT call shutdown() — Gloo shutdown barrier would hang
        engine.dp_group = None
        engine.dp_store = None

    def _reinit_dp_group(self, params: dict | None) -> None:
        """Reinitialize the DP process group via TCPStore coordination.

        Also allocates a separate port for worker cpu_group reinit via the
        same recovery store, following the pattern from
        StatelessGroupCoordinator: rank 0 binds a random port, publishes it,
        all ranks read it so workers can all rendezvous on the same port.
        """
        engine = self._engine_core
        pc = engine.vllm_config.parallel_config
        if params is not None:
            pc.data_parallel_master_ip = params.get(
                "coord_store_host", pc.data_parallel_master_ip
            )
            coord_store_port = params.get("coord_store_port")
            if coord_store_port is not None:
                store_port = int(coord_store_port)
                if self._recovery_store is None:
                    from torch.distributed import TCPStore
                    self._recovery_store = TCPStore(
                        pc.data_parallel_master_ip,
                        store_port,
                        world_size=pc.data_parallel_size,
                        is_master=pc.data_parallel_rank == 0,
                        multi_tenant=True,
                    )
                pc._coord_store_port = store_port

                # Pass recovery store connection info to workers so they
                # can allocate a dedicated port for cpu_group reinit.
                # The actual port is coordinated among workers (not here)
                # via this store, following the StatelessGroupCoordinator
                # pattern: rank 0 binds a socket and passes listen_socket
                # directly to avoid TOCTOU races.
                params["recv_store_host"] = pc.data_parallel_master_ip
                params["recv_store_port"] = store_port

        dp_group, dp_store = pc.stateless_init_dp_group(return_store=True)
        engine.dp_group = dp_group
        engine.dp_store = dp_store

    def _clean_engine_state(self) -> None:
        """Reset EngineCore scheduler and in-flight request state."""
        engine = self._engine_core
        # Mark all in-flight requests as finished (error)
        # The scheduler is reset on next step
        # Drop all pending input_queue items
        try:
            while True:
                engine.input_queue.get_nowait()
        except Exception:
            pass
        # Reset scheduling state
        if hasattr(engine.scheduler, "reset_state"):
            engine.scheduler.reset_state()
