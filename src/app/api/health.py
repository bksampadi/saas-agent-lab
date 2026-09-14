from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: the process is up and serving requests.

    Deliberately does not touch the database; a readiness check comes later.
    """
    return HealthResponse(status="ok")
