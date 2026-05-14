"""GET /health — liveness check."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Liveness check endpoint.
    Returns HTTP 200 with {"status": "ok"} when the service is running.
    """
    return {"status": "ok"}
