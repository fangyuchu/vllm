# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from typing import TYPE_CHECKING

import msgspec
import torch
import zmq

from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pp_group,
    get_tp_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import close_sockets, make_zmq_socket
from vllm.v1.fault_tolerance.utils import FaultToleranceCommand

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_GLOBAL_PAUSE_EVENT = threading.Event()

# Currently, only deepep_ll and nixl_ep backends support fault tolerance.
_FT_BACKEND_SET = {"deepep_low_latency", "nixl_ep"}


def get_pause_event() -> threading.Event:
    global _GLOBAL_PAUSE_EVENT
    return _GLOBAL_PAUSE_EVENT


class WorkerSentinel:
    def __init__(
        self,
        worker: "Worker",
        device: torch.device,
        worker_cmd_addr: str,
    ):
        self.worker = worker
        self.sentinel_dead = False
        self.dp_rank = worker.parallel_config.data_parallel_rank
        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        identity = f"PP{pp_rank}_TP{tp_rank}".encode()
        self.device = device
        self.data_parallel_master_ip = worker.parallel_config.data_parallel_master_ip
        self.dp_size = worker.parallel_config.data_parallel_size

        self.ctx = zmq.Context()
        self.engine_core_cmd_socket = make_zmq_socket(
            self.ctx,
            worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=identity,
        )

        self.use_ft_backend = (
            worker.parallel_config.all2all_backend in _FT_BACKEND_SET
            and worker.parallel_config.data_parallel_size > 1
        )
        if self.use_ft_backend:
            world_size = get_ep_group().world_size
            self.mask = torch.zeros((world_size,), device=self.device, dtype=torch.int)
            self.last_mask = torch.zeros_like(self.mask)

        threading.Thread(
            target=self.run, daemon=True, name="WorkerSentinelThread"
        ).start()

    def run(self):
        """Receive and execute commands from EngineCoreSentinel."""
        torch.accelerator.set_device_index(self.device)
        while not self.sentinel_dead:
            try:
                _, msg = self.engine_core_cmd_socket.recv_multipart()
                cmd = msgspec.msgpack.decode(msg, type=FaultToleranceCommand)
                self._execute_cmd(cmd)
                self.engine_core_cmd_socket.send_multipart([b"", b"ok"])
            except zmq.ZMQError:
                logger.info("Socket closed, terminating.")
                self.sentinel_dead = True

    def _execute_cmd(self, cmd: FaultToleranceCommand):
        instruction = cmd.instruction
        if instruction == "pause":
            self.pause()
        elif instruction == "retry":
            self.retry(cmd.params or {})
        else:
            logger.warning("Unknown FT command: %s", instruction)

    def pause(self):
        get_pause_event().set()

    def retry(self, params: dict):
        self.clean_worker_state()
        get_pause_event().clear()
        if self.dp_size > 1:
            get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
                self.data_parallel_master_ip,
                params["new_stateless_dp_group_port"],
                self.dp_rank,
                self.dp_size,
                backend="gloo",
                gloo_timeout_seconds=self.worker.parallel_config.gloo_timeout_seconds,
            )
            if self.worker.vllm_config.model_config.is_moe:
                comm = get_ep_group().device_communicator
                assert comm and comm.all2all_manager
                if self.worker.parallel_config.all2all_backend not in _FT_BACKEND_SET:
                    raise RuntimeError(
                        "Only support {} backends".format(_FT_BACKEND_SET)
                    )
                comm.all2all_manager.clean_mask()

    def clean_worker_state(self):
        self.worker.model_runner.execute_model_state = None
        self.worker.model_runner.kv_connector_output = None
        input_batch = self.worker.model_runner.input_batch
        cached_req_ids = input_batch.req_id_to_index.keys()
        for req_id in list(cached_req_ids):
            input_batch.remove_request(req_id)
        input_batch.condense()
        input_batch.refresh_metadata()
        input_batch.req_prompt_embeds.clear()

    def shutdown(self):
        self.sentinel_dead = True
        close_sockets([self.engine_core_cmd_socket])
        self.ctx.term()
