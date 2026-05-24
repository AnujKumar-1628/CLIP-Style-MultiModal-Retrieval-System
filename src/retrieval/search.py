"""Search and query routing for multimodal retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models.clip_model import CLIPModel
from src.retrieval.index import NumpyFlatIndex, RetrievalIndexBundle
from src.retrieval.rerankers import BaseRetrievalReranker
from src.utils.config import (
    RetrievalQueryType,
    RetrievalSearchConfig,
    RetrievalTargetModality,
)
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_search",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class SearchRequest:
    query_type: RetrievalQueryType
    target_modality: RetrievalTargetModality
    top_k: int | None = None
    text: str | None = None
    image: torch.Tensor | None = None
    embedding: np.ndarray | torch.Tensor | None = None


@dataclass(frozen=True)
class SearchHit:
    rank: int
    item_id: str
    score: float
    metadata: dict[str, Any] | None


def _to_numpy_embedding(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected query embeddings shape [B, D], got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(
        f"Expected image query with shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}."
    )


class RetrievalSearcher:
    """Router-backed multimodal search over prebuilt indices."""

    def __init__(
        self,
        *,
        model: CLIPModel,
        image_index: RetrievalIndexBundle,
        text_index: RetrievalIndexBundle | None,
        cfg: RetrievalSearchConfig,
        reranker: BaseRetrievalReranker | None = None,
        image_metadata: dict[str, dict[str, Any]] | None = None,
        text_metadata: dict[str, dict[str, Any]] | None = None,
        image_candidate_embeddings: np.ndarray | None = None,
        text_candidate_embeddings: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.image_index = image_index
        self.text_index = text_index
        self.cfg = cfg
        self.reranker = reranker
        self.image_metadata = image_metadata or {}
        self.text_metadata = text_metadata or {}
        self.image_candidate_embeddings = image_candidate_embeddings
        self.text_candidate_embeddings = text_candidate_embeddings

    def _resolve_top_k(self, requested: int | None, max_available: int) -> int:
        if requested is None:
            requested = self.cfg.default_top_k
        k = int(requested)
        if k <= 0:
            raise ValueError("top_k must be > 0.")
        k = min(k, self.cfg.max_top_k)
        k = min(k, max_available)
        return k

    def _resolve_candidate_k(self, top_k: int, max_available: int) -> int:
        if self.reranker is None:
            return top_k
        return self.reranker.resolve_candidate_k(
            requested_top_k=top_k,
            max_available=max_available,
        )

    def _encode_query(self, request: SearchRequest) -> np.ndarray:
        qtype = request.query_type
        if qtype == "text":
            if not request.text:
                raise ValueError("Text query requires `text` field.")
            emb = self.model.encode_texts([request.text], normalize=True)
            return emb.detach().cpu().numpy()
        if qtype == "image":
            if request.image is None:
                raise ValueError("Image query requires `image` field.")
            image_batch = _ensure_image_batch(request.image)
            emb = self.model.encode_images(image_batch, normalize=True)
            return emb.detach().cpu().numpy()
        if qtype in {"text_embedding", "image_embedding"}:
            if request.embedding is None:
                raise ValueError(f"{qtype} query requires `embedding` field.")
            return _to_numpy_embedding(request.embedding)
        raise ValueError(f"Unsupported query_type '{qtype}'.")

    def _select_target_index(
        self,
        target: RetrievalTargetModality,
    ) -> tuple[RetrievalIndexBundle, dict[str, dict[str, Any]]]:
        if target == "image":
            return self.image_index, self.image_metadata
        if self.text_index is None:
            raise RuntimeError("Text index is not loaded; cannot run text-target searches.")
        return self.text_index, self.text_metadata

    def _candidate_embedding_table(
        self,
        *,
        target: RetrievalTargetModality,
        target_index: RetrievalIndexBundle,
    ) -> np.ndarray | None:
        if target == "image":
            if self.image_candidate_embeddings is not None:
                return self.image_candidate_embeddings
        else:
            if self.text_candidate_embeddings is not None:
                return self.text_candidate_embeddings

        if isinstance(target_index.index, NumpyFlatIndex):
            return target_index.index.embeddings
        return None

    @staticmethod
    def _gather_candidate_embeddings(
        *,
        embedding_table: np.ndarray | None,
        indices: np.ndarray,
    ) -> np.ndarray | None:
        if embedding_table is None:
            return None
        table = np.asarray(embedding_table, dtype=np.float32)
        if table.ndim != 2:
            return None
        safe = np.where(indices >= 0, indices, 0).astype(np.int64, copy=False)
        gathered = table[safe]
        invalid = indices < 0
        if np.any(invalid):
            gathered[invalid] = 0.0
        return gathered.astype(np.float32, copy=False)

    def search(self, request: SearchRequest) -> list[list[SearchHit]]:
        query = self._encode_query(request)
        target_index, meta_lookup = self._select_target_index(request.target_modality)
        top_k = self._resolve_top_k(request.top_k, len(target_index.ids))
        candidate_k = self._resolve_candidate_k(top_k, len(target_index.ids))

        scores, indices = target_index.index.search(query, candidate_k)
        if self.reranker is not None and self.reranker.enabled:
            candidate_table = self._candidate_embedding_table(
                target=request.target_modality,
                target_index=target_index,
            )
            candidate_embeddings = self._gather_candidate_embeddings(
                embedding_table=candidate_table,
                indices=indices,
            )
            scores, indices = self.reranker.rerank(
                query_embeddings=query,
                candidate_scores=scores,
                candidate_indices=indices,
                candidate_embeddings=candidate_embeddings,
                metric=target_index.index.metric,  # type: ignore[arg-type]
                top_k=top_k,
            )
        else:
            scores = scores[:, :top_k]
            indices = indices[:, :top_k]

        results: list[list[SearchHit]] = []
        for row_idx in range(scores.shape[0]):
            row_hits: list[SearchHit] = []
            for rank_idx in range(scores.shape[1]):
                col_idx = int(indices[row_idx, rank_idx])
                if col_idx < 0 or col_idx >= len(target_index.ids):
                    continue
                item_id = target_index.ids[col_idx]
                metadata = meta_lookup.get(item_id) if self.cfg.return_metadata else None
                row_hits.append(
                    SearchHit(
                        rank=rank_idx + 1,
                        item_id=item_id,
                        score=float(scores[row_idx, rank_idx]),
                        metadata=metadata,
                    )
                )
            results.append(row_hits)
        return results


__all__ = [
    "RetrievalSearcher",
    "SearchHit",
    "SearchRequest",
]
