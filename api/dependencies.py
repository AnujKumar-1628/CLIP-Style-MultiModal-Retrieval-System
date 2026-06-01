"""FastAPI dependency providers for API runtime objects."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

from src.retrieval.serving import RetrievalServer
from src.utils.config import load_api_config
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="api_dependencies",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class APISettings:
    title: str
    version: str
    host: str
    port: int
    root_path: str
    docs_url: str
    redoc_url: str
    retrieval_config_path: Path
    model_config_path: Path
    data_config_path: Path
    image_index_name: str
    text_index_name: str

@lru_cache(maxsize=1)
def get_api_settings() -> APISettings:
    api_cfg = load_api_config()

    settings = APISettings(
        title=api_cfg.app.title,
        version=api_cfg.app.version,
        host=api_cfg.server.host,
        port=api_cfg.server.port,
        root_path=api_cfg.server.root_path,
        docs_url=api_cfg.server.docs_url,
        redoc_url=api_cfg.server.redoc_url,
        retrieval_config_path=api_cfg.paths.retrieval_config,
        model_config_path=api_cfg.paths.model_config,
        data_config_path=api_cfg.paths.data_config,
        image_index_name=api_cfg.index.image_index_name,
        text_index_name=api_cfg.index.text_index_name,
    )
    LOGGER.info("Loaded API settings via src.utils.config.load_api_config().")
    return settings


@lru_cache(maxsize=1)
def get_retrieval_server() -> RetrievalServer:
    settings = get_api_settings()
    try:
        return RetrievalServer(
            retrieval_config_path=settings.retrieval_config_path,
            model_config_path=settings.model_config_path,
            data_config_path=settings.data_config_path,
            image_index_name=settings.image_index_name,
            text_index_name=settings.text_index_name,
        )
    except Exception as exc:
        LOGGER.exception("Failed to initialize RetrievalServer: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize retrieval server: {exc}",
        ) from exc
