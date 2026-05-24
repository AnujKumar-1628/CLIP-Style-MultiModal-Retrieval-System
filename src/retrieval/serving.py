"""Serving helpers for retrieval search APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.retrieval.pipeline import load_retrieval_searcher
from src.retrieval.search import RetrievalSearcher, SearchRequest
from src.utils.config import RetrievalConfig, load_retrieval_config
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_serving",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class SearchResponse:
    hits: list[dict[str, Any]]


class RetrievalServer:
    """Lightweight gateway that keeps model/index loaded in memory."""

    def __init__(
        self,
        *,
        retrieval_config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
        image_index_name: str = "image_index",
        text_index_name: str = "text_index",
    ) -> None:
        self.retrieval_cfg: RetrievalConfig = load_retrieval_config(retrieval_config_path)
        self.searcher: RetrievalSearcher = load_retrieval_searcher(
            retrieval_config_path=retrieval_config_path,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            image_index_name=image_index_name,
            text_index_name=text_index_name,
        )
        LOGGER.info("RetrievalServer initialized.")

    @staticmethod
    def _hits_to_dict(rows) -> list[dict[str, Any]]:
        if not rows:
            return []
        return [
            {
                "rank": hit.rank,
                "item_id": hit.item_id,
                "score": hit.score,
                "metadata": hit.metadata,
            }
            for hit in rows[0]
        ]

    def search_text_to_image(self, *, query: str, top_k: int | None = None) -> SearchResponse:
        rows = self.searcher.search(
            SearchRequest(
                query_type="text",
                target_modality="image",
                top_k=top_k,
                text=query,
            )
        )
        return SearchResponse(hits=self._hits_to_dict(rows))

    def search_image_to_image(
        self, *, image: torch.Tensor, top_k: int | None = None
    ) -> SearchResponse:
        rows = self.searcher.search(
            SearchRequest(
                query_type="image",
                target_modality="image",
                top_k=top_k,
                image=image,
            )
        )
        return SearchResponse(hits=self._hits_to_dict(rows))

    def search_image_to_text(
        self, *, image: torch.Tensor, top_k: int | None = None
    ) -> SearchResponse:
        rows = self.searcher.search(
            SearchRequest(
                query_type="image",
                target_modality="text",
                top_k=top_k,
                image=image,
            )
        )
        return SearchResponse(hits=self._hits_to_dict(rows))

    def search_text_to_text(self, *, query: str, top_k: int | None = None) -> SearchResponse:
        rows = self.searcher.search(
            SearchRequest(
                query_type="text",
                target_modality="text",
                top_k=top_k,
                text=query,
            )
        )
        return SearchResponse(hits=self._hits_to_dict(rows))

    def build_fastapi_app(self):
        """Optional FastAPI wrapper for serving REST search."""
        try:
            from fastapi import FastAPI, HTTPException, Query
        except Exception as exc:  # pragma: no cover
            raise ImportError("FastAPI is required to build API app.") from exc

        app = FastAPI(title="Retrieval Server", version="0.1.0")

        @app.get(self.retrieval_cfg.serving.route)
        def search(
            query: str = Query(..., min_length=1),
            query_type: str = Query("text"),
            target: str = Query("image"),
            top_k: int = Query(self.retrieval_cfg.search.default_top_k, ge=1),
        ):
            try:
                rows = self.searcher.search(
                    SearchRequest(
                        query_type=query_type,  # type: ignore[arg-type]
                        target_modality=target,  # type: ignore[arg-type]
                        top_k=top_k,
                        text=query if query_type == "text" else None,
                    )
                )
                return {"hits": self._hits_to_dict(rows)}
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return app


__all__ = [
    "RetrievalServer",
    "SearchResponse",
]
