"""GET /health — liveness check."""

from fastapi import APIRouter, Request
from retrieval import vector_store

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check(request: Request) -> dict:
    """
    Liveness check endpoint.
    Returns HTTP 200 with {"status": "ok"} when the service is running.
    """
    catalog_size = len(vector_store._catalog_metadata)
    llm_ready = hasattr(request.app.state, "llm_client")
    return {"status": "ok", "catalog_size": catalog_size, "llm_ready": llm_ready}
