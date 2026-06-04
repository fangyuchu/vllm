# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.v1.fault_tolerance.utils import FaultToleranceRequest
from vllm.v1.serial_utils import run_method

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_FT_BACKEND_SET = {"deepep_low_latency", "nixl_ep"}


class WorkerSentinel:
    """Holds FT state for a single worker (mask tensors, DP config).

    Methods are called via collective_rpc from EngineCoreSentinel.
    """

    def __init__(self, worker: "Worker", device: torch.device):
        self.worker = worker
        self.device = device
        dp_rank = worker.parallel_config.data_parallel_rank
        self.dp_rank = dp_rank
        self.dp_size = worker.parallel_config.data_parallel_size
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip

        backend = worker.parallel_config.all2all_backend
        self.use_ft_backend = backend in _FT_BACKEND_SET and self.dp_size > 1
        logger.info(
            "dp_rank=%d dp_size=%d backend=%s use_ft_backend=%s ep_world_size=%d",
            dp_rank,
            self.dp_size,
            backend,
            self.use_ft_backend,
            get_ep_group().world_size if self.use_ft_backend else -1,
        )
        if self.use_ft_backend:
            world_size = get_ep_group().world_size
            self.mask = torch.zeros(world_size, device=device, dtype=torch.int)
            self.last_mask = torch.zeros_like(self.mask)
            logger.info(
                "dp_rank=%d world_size=%d device=%s",
                dp_rank,
                world_size,
                device,
            )

    def handle_command(self, ft_request: FaultToleranceRequest):
        """Dispatch an FT command by instruction name."""
        logger.info(
            "dp_rank=%d instruction=%s request_id=%s",
            self.dp_rank,
            ft_request.instruction,
            ft_request.request_id,
        )
        with set_current_vllm_config(self.worker.vllm_config):
            return run_method(self, ft_request.instruction, (ft_request,), {})

    def retry(self, ft_request: FaultToleranceRequest):
        logger.info(
            "dp_rank=%d params_keys=%s",
            self.dp_rank,
            list(ft_request.params.keys()) if ft_request.params else [],
        )
        params = ft_request.params
        self._clean_worker_state()
        if self.dp_size > 1:
            port = params["new_stateless_dp_group_port"]
            get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
                self.data_parallel_master_ip,
                port,
                self.dp_rank,
                self.dp_size,
                backend="gloo",
            )
            if self.use_ft_backend:
                comm = get_ep_group().device_communicator
                assert comm and comm.all2all_manager
                comm.all2all_manager.clean_mask()
                # Reset last_mask to zeros so the next mask comparison
                # doesn't detect a false positive from the pre-recovery state.
                self.last_mask.zero_()

    def _clean_worker_state(self):
        self.worker.model_runner.execute_model_state = None
        self.worker.model_runner.kv_connector_output = None
        input_batch = self.worker.model_runner.input_batch
        cached_req_ids = input_batch.req_id_to_index.keys()
        for req_id in list(cached_req_ids):
            input_batch.remove_request(req_id)
        input_batch.condense()
        input_batch.refresh_metadata()
        input_batch.req_prompt_embeds.clear()
