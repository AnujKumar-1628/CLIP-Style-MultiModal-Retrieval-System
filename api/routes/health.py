"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas import APIHealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=APIHealthResponse)
def health_check() -> APIHealthResponse:
    return APIHealthResponse(status="ok", service="clip-retrieval-api")
