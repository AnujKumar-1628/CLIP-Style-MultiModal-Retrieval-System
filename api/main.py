"""FastAPI application entrypoint for retrieval serving."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.requests import Request

from api.dependencies import get_api_settings
from api.routes.health import router as health_router
from api.routes.index import router as index_router
from api.routes.search import router as search_router
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="api_main",
    level="INFO",
    use_console=True,
    use_file=True,
)

settings = get_api_settings()

app = FastAPI(
    title=settings.title,
    version=settings.version,
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    root_path=settings.root_path,
)

app.include_router(index_router)
app.include_router(health_router)
app.include_router(search_router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )
