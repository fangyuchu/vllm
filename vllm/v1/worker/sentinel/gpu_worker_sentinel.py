# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable

import msgspec
import torch
import zmq

from vllm.config import ParallelConfig
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

logger = init_logger(__name__)

_GLOBAL_PAUSE_EVENT = threading.Event()

# Currently, only deepep_ll and nixl_ep backends support fault tolerance.
_FT_BACKEND_SET = {"deepep_low_latency", "nixl_ep"}


def get_pause_event() -> threading.Event:
    global _GLOBAL_PAUSE_EVENT
    return _GLOBAL_PAUSE_EVENT


class WorkerSentinel:
    """Per-GPU worker sentinel thread.

    Listens for out-of-band pause/retry commands from the engine core via ZMQ.
    This is necessary because when a fault occurs, healthy workers may be stuck
    in a collective operation (NCCL/Gloo) and cannot be reached via
    collective_rpc. The sentinel thread provides an independent control channel.
    """

    def __init__(
        self,
        parallel_config: ParallelConfig,
        device: torch.device,
        worker_cmd_addr: str,
        clear_input_batch_callback: Callable,
        reset_async_stream_callback: Callable | None = None,
        restart_async_output_thread_callback: Callable | None = None,
    ):
        self.parallel_config = parallel_config
        self.dp_rank = parallel_config.data_parallel_rank
        self.device = device
        self.data_parallel_master_ip = parallel_config.data_parallel_master_ip
        self.data_parallel_master_port = parallel_config.data_parallel_master_port
        self.dp_size = parallel_config.data_parallel_size

        tp_rank = get_tp_group().rank_in_group
        pp_rank = get_pp_group().rank_in_group
        self.identity = f"PP{pp_rank}_TP{tp_rank}".encode()

        self.ctx = zmq.Context()
        self.cmd_socket = make_zmq_socket(
            self.ctx,
            worker_cmd_addr,
            zmq.DEALER,
            bind=False,
            identity=self.identity,
        )

        self.use_ft_backend = (
            parallel_config.all2all_backend in _FT_BACKEND_SET
            and parallel_config.data_parallel_size > 1
        )
        if self.use_ft_backend:
            world_size = get_ep_group().world_size
            self.mask = torch.zeros(
                (world_size,), device=self.device, dtype=torch.int
            )

        self.clear_input_batch_callback = clear_input_batch_callback
        self.reset_async_stream_callback = reset_async_stream_callback
        self.restart_async_output_thread_callback = (
            restart_async_output_thread_callback
        )

        self._dead = False
        torch.accelerator.set_device_index(self.device)
        threading.Thread(
            target=self._run, daemon=True, name="WorkerSentinelThread"
        ).start()

    def _run(self):
        torch.accelerator.set_device_index(self.device)
        while not self._dead:
            try:
                _, msg = self.cmd_socket.recv_multipart()
                cmd = msgspec.msgpack.decode(msg, type=FaultToleranceCommand)
                result = self._execute(cmd)
                self.cmd_socket.send_multipart(
                    [b"", msgspec.msgpack.encode(result)]
                )
            except zmq.ZMQError:
                logger.info("WorkerSentinel socket closed, terminating.")
                self._dead = True

    def _execute(self, cmd: FaultToleranceCommand) -> bool:
        """Execute a command. Returns True on success."""
        handler = getattr(self, f"_handle_{cmd.instruction}", None)
        if handler is None:
            logger.error("Unknown FT command: %s", cmd.instruction)
            return False
        try:
            handler(cmd)
            return True
        except Exception:
            logger.exception("FT command '%s' failed", cmd.instruction)
            return False

    def _handle_pause(self, cmd: FaultToleranceCommand):
        get_pause_event().set()

    def _handle_retry(self, cmd: FaultToleranceCommand):
        self.clear_input_batch_callback()
        get_pause_event().clear()

        if self.use_ft_backend:
            comm = get_ep_group().device_communicator
            assert comm and comm.all2all_manager
            comm.all2all_manager.clean_mask()

        get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
            self.data_parallel_master_ip,
            cmd.params["new_stateless_dp_group_port"],
            self.dp_rank,
            self.dp_size,
            backend="gloo",
        )
        if self.reset_async_stream_callback is not None:
            self.reset_async_stream_callback()
        if self.restart_async_output_thread_callback is not None:
            self.restart_async_output_thread_callback()

    def shutdown(self):
        self._dead = True
        close_sockets([self.cmd_socket])
        self.ctx.term()
