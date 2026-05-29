# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import socket

import torch

from vllm.distributed.parallel_state import get_dp_group
from vllm.distributed.utils import (
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class WorkerSentinel:
    """FT proxy attached to WorkerBase.

    Only two public FT methods: retry, ft_scale_down (deferred).
    WorkerBase code is NOT modified.
    """

    def __init__(self, worker: "WorkerBase"):
        self._worker = worker

    def retry(self, params: dict | None) -> str:
        """Full retry recovery: reset state, rebuild comm, clean all2all."""
        self._reset_state()
        self._reinit_comm(params)
        self._clean_all2all()
        return f"worker_{self._worker.rank}_recovered"

    def ft_scale_down(self, params: dict | None) -> str:
        """scale_down recovery (deferred to future implementation)."""
        raise NotImplementedError("scale_down not yet implemented")

    def _reset_state(self) -> None:
        """Clear model runner runtime state without destroying the model.

        This resets KV cache state, multimodal cache, and other runtime
        buffers. The model itself is preserved.
        """
        worker = self._worker

        # Reset multimodal caches
        if hasattr(worker.model_runner, "reset_mm_cache"):
            worker.model_runner.reset_mm_cache()
        if hasattr(worker.model_runner, "reset_encoder_cache"):
            worker.model_runner.reset_encoder_cache()

        # Reset forward context
        from vllm.forward_context import override_forward_context

        override_forward_context(None)

        logger.info(
            "Worker rank=%d: reset runtime state", worker.rank
        )

    def _reinit_comm(self, params: dict | None) -> None:
        """Rebuild DP group's Gloo cpu_group via recovery store coordination.

        Follows the StatelessGroupCoordinator pattern: workers connect to the
        recovery store (shared TCPStore), rank 0 in the cpu_group binds a
        random port and publishes it for its group mates via a per-rank key.

        This avoids port collision with the EngineCore's DP group reinit
        which uses a different port on the same coord_store_port.
        """
        if params is None:
            raise ValueError("params required for comm reinit")
        worker = self._worker
        dp_group = get_dp_group()

        store_host = params.get("recv_store_host")
        store_port = params.get("recv_store_port")
        if store_host is not None and store_port is not None:
            store_port = int(store_port)
            from torch.distributed import TCPStore
            store = TCPStore(
                store_host,
                store_port,
                world_size=worker.vllm_config.parallel_config.data_parallel_size,
                is_master=False,
                multi_tenant=True,
            )
            # Single shared key across all DP workers so rank 0 (in the
            # cpu_group) publishes and rank 1 reads the same port.
            _CPU_KEY = "ft_worker_cpu_port"
            if dp_group.rank_in_group == 0:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind((store_host, 0))
                cpu_port = s.getsockname()[1]
                store.set(_CPU_KEY, str(cpu_port).encode())
                listen_sock = s
            else:
                cpu_port = int(store.get(_CPU_KEY).decode())
                listen_sock = None
            host = store_host
            port = cpu_port
        else:
            # Legacy fallback — no recovery store available
            host = params.get("coord_store_host", "127.0.0.1")
            port = int(params.get("coord_store_port", 0))
            listen_sock = None

        # Destroy old cpu_group
        if hasattr(dp_group, "cpu_group") and dp_group.cpu_group is not None:
            try:
                torch.distributed.destroy_process_group(dp_group.cpu_group)
            except Exception:
                logger.warning(
                    "Worker rank=%d: failed to destroy old cpu_group, "
                    "ignoring",
                    worker.rank,
                )

        kwargs = dict(
            host=host,
            port=port,
            rank=dp_group.rank_in_group,
            world_size=dp_group.world_size,
            backend="gloo",
        )
        if listen_sock is not None:
            kwargs["listen_socket"] = listen_sock
        new_cpu_group = stateless_init_torch_distributed_process_group(**kwargs)
        dp_group.cpu_group = new_cpu_group

        logger.info(
            "Worker rank=%d: rebuilt DP cpu_group via %s:%d",
            worker.rank,
            host,
            port,
        )

    def _clean_all2all(self) -> None:
        """Reset MoE all2all mask so the recovered rank can rejoin.

        No-op if not MoE or no all2all manager.
        """
        worker = self._worker
        if not worker.vllm_config.model_config.is_moe:
            return

        try:
            from vllm.distributed import get_ep_group

            ep_group = get_ep_group()
            if (
                ep_group.device_communicator is not None
                and hasattr(ep_group.device_communicator, "all2all_manager")
                and ep_group.device_communicator.all2all_manager is not None
            ):
                ep_group.device_communicator.all2all_manager.clear_mask()
                logger.info(
                    "Worker rank=%d: cleared all2all mask",
                    worker.rank,
                )
        except Exception:
            logger.exception(
                "Worker rank=%d: failed to clean all2all mask",
                worker.rank,
            )
