# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.serve.elastic_ep.middleware import (
    get_scaling_elastic_ep,
    set_scaling_elastic_ep,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post(
    "/scale_elastic_ep",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"model": dict},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.REQUEST_TIMEOUT.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
async def scale_elastic_ep(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    new_data_parallel_size = body.get("new_data_parallel_size")
    drain_timeout = body.get("drain_timeout", 120)  # Default 2 minutes

    if new_data_parallel_size is None:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size is required"
        )

    if not isinstance(new_data_parallel_size, int) or new_data_parallel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="new_data_parallel_size must be a positive integer",
        )

    if not isinstance(drain_timeout, int) or drain_timeout <= 0:
        raise HTTPException(
            status_code=400, detail="drain_timeout must be a positive integer"
        )

    # Set scaling flag to prevent new requests
    set_scaling_elastic_ep(True)
    client = engine_client(raw_request)
    try:
        await client.scale_elastic_ep(new_data_parallel_size, drain_timeout)
        return JSONResponse(
            {
                "message": f"Scaled to {new_data_parallel_size} data parallel engines",
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail="Scale failed") from e
    finally:
        set_scaling_elastic_ep(False)


@router.post("/is_scaling_elastic_ep")
async def is_scaling_elastic_ep(raw_request: Request):
    return JSONResponse({"is_scaling_elastic_ep": get_scaling_elastic_ep()})


def attach_router(app: FastAPI):
    app.include_router(router)

@router.post("/scale_elastic_ep_external")
async def scale_elastic_ep_external(raw_request: Request):
    """Handle reconfiguration from an external orchestrator (External LB mode)."""
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format") from e

    new_data_parallel_size = body.get("new_data_parallel_size")
    new_data_parallel_master_port = body.get("new_data_parallel_master_port")
    drain_timeout = body.get("drain_timeout", 300)

    if new_data_parallel_size is None:
        raise HTTPException(
            status_code=400, detail="new_data_parallel_size is required"
        )

    if not isinstance(new_data_parallel_size, int) or new_data_parallel_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="new_data_parallel_size must be a positive integer",
        )

    if new_data_parallel_master_port is not None:
        if not isinstance(new_data_parallel_master_port, int) or new_data_parallel_master_port <= 0:
            raise HTTPException(
                status_code=400,
                detail="new_data_parallel_master_port must be a positive integer",
            )

    if not isinstance(drain_timeout, int) or drain_timeout <= 0:
        raise HTTPException(
            status_code=400, detail="drain_timeout must be a positive integer"
        )

    client = engine_client(raw_request)

    # Require External LB mode.
    if not client.vllm_config.parallel_config.data_parallel_external_lb:
        raise HTTPException(
            status_code=400,
            detail="This endpoint is only available in External LB mode. "
            "Use /scale_elastic_ep for Internal LB mode.",
        )

    # Block new traffic while waiting for in-flight requests to drain and
    # reconfiguring local engines. Without this, External LB scale-up can keep
    # admitting new work and stretch the drain path until timeout.
    set_scaling_elastic_ep(True)
    try:
        # engine_core is DPAsyncMPClient in External LB mode.
        engine_core = client.engine_core
        
        # Must be DPAsyncMPClient for this endpoint.
        from vllm.v1.engine.core_client import DPAsyncMPClient
        if not isinstance(engine_core, DPAsyncMPClient):
            raise HTTPException(
                status_code=500,
                detail="This endpoint requires DPAsyncMPClient (External LB mode)",
            )

        # Rank 0 on scale-up: start handshake listener if not already running.
        if (
            client.vllm_config.parallel_config.data_parallel_rank == 0
            and new_data_parallel_size
            > client.vllm_config.parallel_config.data_parallel_size
        ):
            await engine_core.start_handshake_listener_if_needed(new_data_parallel_size)

        # Wait for in-flight requests to drain.
        await client.wait_for_requests_to_drain(drain_timeout)

        # Reconfigure local engine for the new DP size.
        await engine_core.reconfigure_for_external_scale(
            new_data_parallel_size,
            new_data_parallel_master_port=new_data_parallel_master_port,
        )

        return JSONResponse(
            {
                "message": f"Scaled to {new_data_parallel_size} data parallel engines",
            }
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=408,
            detail="Scale failed due to request drain timeout "
            f"after {drain_timeout} seconds",
        ) from e
    except Exception as e:
        logger.error("Scale failed: %s", e)
        raise HTTPException(status_code=500, detail="Scale failed") from e
    finally:
        set_scaling_elastic_ep(False)
