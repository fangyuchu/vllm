# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, Request

router = APIRouter(prefix="/fault_tolerance", tags=["fault_tolerance"])


@router.post("/apply")
async def apply_fault_tolerance(request: Request):
    """Trigger fault tolerance recovery on the engine."""
    body = await request.json()
    client = request.app.state.engine_client
    from vllm.v1.fault_tolerance.utils import FaultToleranceRequest

    ft_request = FaultToleranceRequest(
        request_id=body.get("request_id", "ft-manual"),
        instruction=body.get("instruction", "retry"),
        params=body.get("params") or {},
    )
    try:
        result = await client.handle_fault(ft_request)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/status")
async def get_fault_tolerance_status(request: Request):
    """Get current fault tolerance status."""
    client = request.app.state.engine_client
    try:
        status = await client.get_ft_status_async()
        # Normalize: if status is a dict with an inner "status" key, extract it
        if isinstance(status, dict) and "status" in status:
            return {"status": status["status"]}
        return {"status": status}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}
