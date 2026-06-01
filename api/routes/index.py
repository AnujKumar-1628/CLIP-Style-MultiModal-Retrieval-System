"""Root/index endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import APISettings, get_api_settings
from api.schemas import APIRootResponse

router = APIRouter(tags=["index"])


@router.get("/", response_model=APIRootResponse)
def index(settings: APISettings = Depends(get_api_settings)) -> APIRootResponse:
    return APIRootResponse(
        service=settings.title,
        status="ok",
        version=settings.version,
        docs_url=settings.docs_url,
    )
