# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from vllm.logger import init_logger

logger = init_logger(__name__)

"""
    Custom HTTP request handler for port coordination service.
    This handler processes registration requests from distributed training processes
    and provides health/register endpoints for coordination monitoring.
"""


class CoordinationHandler(BaseHTTPRequestHandler):
    def __init__(self, coordinator, *args, **kwargs):
        self.coordinator = coordinator
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "ok",
                "coordinator": True,
                "port": self.coordinator.selected_port,
                "registered_ranks": list(self.coordinator.registered_ranks),
                "world_size": self.coordinator.world_size,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/register":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            rank = data.get("rank")
            if rank is not None:
                self.coordinator.register_rank(rank)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "registered",
                "rank": rank,
                "port": self.coordinator.selected_port,
                "available_port": self.coordinator.available_port,
                "registered_count": len(self.coordinator.registered_ranks),
                "world_size": self.coordinator.world_size,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def log_request(self, code="-", size="-"):
        pass

    def log_message(self, format, *args):
        pass

    def log_error(self, format, *args):
        error_msg = format % args
        logger.error("HTTP Error: %s", error_msg)


"""
    Manages port coordination for distributed training processes.
    The coordinator runs an HTTP server that allows worker processes to
    register and agree on a communication port for collective operations.
"""


class Coordinator:
    def __init__(self, host: str, port: int, world_size: int):
        """
        Initialize a new coordinator instance.

        Args:
            host: Hostname or IP address to bind the server to
            port: Port number for the coordination server
            world_size: Total number of processes expected to register
        """
        self.host = host
        self.selected_port = port
        self.world_size = world_size
        self.registered_ranks = set()
        self.server = None
        self.lock = threading.Lock()
        self.all_registered_event = threading.Event()
        self.available_port = get_random_port()

    def register_rank(self, rank: int):
        with self.lock:
            self.registered_ranks.add(rank)
            logger.info(
                "Rank %s registered, total: %s/%s",
                rank,
                len(self.registered_ranks),
                self.world_size,
            )

            if len(self.registered_ranks) >= self.world_size:
                self.all_registered_event.set()
                logger.info("All %s ranks registered!", self.world_size)

    def start_server(self) -> bool:
        try:
            handler_factory = lambda *args, **kwargs: CoordinationHandler(
                self, *args, **kwargs
            )

            self.server = ThreadingHTTPServer(
                (self.host, self.selected_port), handler_factory
            )

            server_thread = threading.Thread(
                target=self.server.serve_forever, daemon=True
            )
            server_thread.start()

            logger.info(
                "Coordination server started on %s:%s", self.host, self.selected_port
            )
            return True

        except Exception as e:
            logger.error("Failed to start coordination server: %s", e)
            return False

    def stop_server(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Coordination server stopped")

    def wait_for_all_ranks(self, timeout: int = 30) -> bool:
        return self.all_registered_event.wait(timeout)


def is_port_available(host: str, port: int, timeout: float = 0.1) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result != 0
    except Exception:
        return False


def get_random_port():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def check_coordination_server(host: str, port: int, timeout: float = 5.0) -> bool:
    for _ in range(20):
        try:
            url = "http://{}:{}/health".format(host, port)
            response = urlopen(url, timeout=timeout)
            if response.getcode() != 200:
                time.sleep(1)
                continue
            data = json.loads(response.read())
            return data.get("status") == "ok"
        except Exception as e:
            logger.debug("Coordination server check failed: %s", e)
            time.sleep(1)
    return False


def negotiation_port(
    rank: int,
    start_port: int,
    host: str,
    world_size: int,
    max_attempts: int = 10,
    timeout_seconds: int = 60,
) -> int | None:
    """
    Coordinate port selection for distributed training processes.
    Rank 0 acts as the coordinator, starting a server and waiting for
    all other ranks to register. Other ranks search for the coordinator
    and register with it. Once all ranks are registered, a common port
    is returned for distributed communication.

    Args:
        rank: Current process rank (0 for coordinator)
        start_port: Starting port for coordination attempts
        host: Hostname or IP address for coordination
        world_size: Total number of processes in the group
        max_attempts: Maximum number of port attempts
        timeout_seconds: Overall timeout for coordination

    Returns:
        Selected port for distributed communication

    Raises:
        ValueError: For invalid world_size or rank
        TimeoutError: If coordination fails within timeout
    """
    if world_size <= 0:
        logger.error("Invalid world_size")
        raise ValueError("world_size must be positive")

    if rank < 0 or rank >= world_size:
        logger.error("Invalid rank %s for world_size %s", rank, world_size)
        raise ValueError("rank must be between 0 and world_size-1")

    logger.info(
        "Rank %s/%s starting port negotiation from port %s",
        rank,
        world_size,
        start_port,
    )

    start_time = time.time()

    if rank == 0:
        for attempt in range(max_attempts):
            current_port = start_port + attempt

            if not is_port_available(host, current_port):
                logger.warning(
                    "Rank 0: Port %s is in use, trying next...", current_port
                )
                continue

            coordinator = Coordinator(host, current_port, world_size)

            if not coordinator.start_server():
                continue

            time.sleep(0.5)

            logger.info("Rank 0: Coordination server running on port %s", current_port)

            coordinator.register_rank(0)

            logger.info(
                "Rank 0: Waiting for %s other ranks to register...", (world_size - 1)
            )
            if coordinator.wait_for_all_ranks(
                timeout=timeout_seconds - (time.time() - start_time)
            ):
                logger.info("Rank 0: All %s ranks registered successfully", world_size)

                coordinator.stop_server()

                logger.info(
                    "Rank 0: Port negotiation completed, using port %s",
                    coordinator.available_port,
                )
                return coordinator.available_port
            else:
                logger.warning("Rank 0: Timeout waiting for all ranks to register")
                coordinator.stop_server()
                # try next

    else:
        for attempt in range(max_attempts):
            current_port = start_port + attempt

            if check_coordination_server(host, current_port):
                logger.info(
                    "Rank %s: Found coordination server on port %s", rank, current_port
                )
                try:
                    url = "http://{}:{}/register".format(host, current_port)
                    data = json.dumps({"rank": rank}).encode("utf-8")
                    request = Request(
                        url, data=data, headers={"Content-Type": "application/json"}
                    )
                    response = urlopen(request, timeout=2)
                    if response.getcode() == 200:
                        resp_data = json.loads(response.read())
                        if resp_data.get("status") == "registered":
                            available_port = resp_data.get("available_port")
                            return available_port
                except Exception as e:
                    logger.error("Rank %s: Failed to register: %s", rank, e)

    logger.error(
        "Rank %s: Port negotiation timeout after %s seconds", rank, timeout_seconds
    )
    raise TimeoutError("Port negotiation failed for rank %s", rank)
