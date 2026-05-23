"""Second-stage reranking strategies for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.config import RetrievalDistanceMetric, RetrievalRerankerConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_rerankers",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _to_float32_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [B, D] query embeddings, got {arr.shape}.")
    return arr


def _to_float32_3d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected [B, K, D] candidate embeddings, got {arr.shape}.")
    return arr


def _l2_normalize_2d(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return x / norms


def _l2_normalize_3d(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=2, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return x / norms


def _rowwise_minmax(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    bsz = int(values.shape[0])
    for i in range(bsz):
        mask = valid_mask[i]
        if not np.any(mask):
            continue
        row = values[i, mask]
        vmin = float(np.min(row))
        vmax = float(np.max(row))
        denom = vmax - vmin
        if denom < 1e-12:
            out[i, mask] = 1.0
        else:
            out[i, mask] = (row - vmin) / denom
    return out


def _gather_ranked(
    *,
    candidate_scores: np.ndarray,
    candidate_indices: np.ndarray,
    ranking: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_ids = np.arange(ranking.shape[0])[:, None]
    ordered_scores = candidate_scores[row_ids, ranking]
    ordered_indices = candidate_indices[row_ids, ranking]
    return ordered_scores[:, :top_k], ordered_indices[:, :top_k]


def _query_candidate_similarity(
    *,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    metric: RetrievalDistanceMetric,
) -> np.ndarray:
    q = _to_float32_2d(query_embeddings)
    c = _to_float32_3d(candidate_embeddings)
    if metric == "cosine":
        q = _l2_normalize_2d(q)
        c = _l2_normalize_3d(c)
        return np.einsum("bd,bkd->bk", q, c, optimize=True).astype(np.float32, copy=False)
    if metric == "ip":
        return np.einsum("bd,bkd->bk", q, c, optimize=True).astype(np.float32, copy=False)
    diff = c - q[:, None, :]
    dist_sq = np.sum(diff * diff, axis=2)
    return (-dist_sq).astype(np.float32, copy=False)


@dataclass(frozen=True)
class BaseRetrievalReranker:
    cfg: RetrievalRerankerConfig

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.cfg.name != "none")

    def resolve_candidate_k(self, *, requested_top_k: int, max_available: int) -> int:
        k = int(requested_top_k)
        if k <= 0:
            raise ValueError("requested_top_k must be > 0.")
        if not self.enabled:
            return min(k, max_available)
        expanded = max(k, k * int(self.cfg.candidate_multiplier))
        expanded = min(expanded, int(self.cfg.max_candidates))
        return min(expanded, max_available)

    def rerank(
        self,
        *,
        query_embeddings: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_embeddings: np.ndarray | None,
        metric: RetrievalDistanceMetric,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(int(top_k), int(candidate_scores.shape[1]))
        return candidate_scores[:, :k], candidate_indices[:, :k]


@dataclass(frozen=True)
class NoOpReranker(BaseRetrievalReranker):
    """No second-stage reranking."""


@dataclass(frozen=True)
class EmbeddingBlendReranker(BaseRetrievalReranker):
    """Blend ANN score with exact query-candidate similarity from embeddings."""

    def rerank(
        self,
        *,
        query_embeddings: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_embeddings: np.ndarray | None,
        metric: RetrievalDistanceMetric,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(int(top_k), int(candidate_scores.shape[1]))
        if candidate_embeddings is None:
            LOGGER.warning("Reranker enabled but candidate embeddings are unavailable; using ANN order.")
            return candidate_scores[:, :k], candidate_indices[:, :k]

        scores = np.asarray(candidate_scores, dtype=np.float32, copy=False)
        indices = np.asarray(candidate_indices, dtype=np.int64, copy=False)
        valid = indices >= 0

        exact = _query_candidate_similarity(
            query_embeddings=query_embeddings,
            candidate_embeddings=candidate_embeddings,
            metric=metric,
        )
        exact[~valid] = -np.inf
        approx = scores.copy()
        approx[~valid] = -np.inf

        approx_norm = _rowwise_minmax(approx, valid)
        exact_norm = _rowwise_minmax(exact, valid)

        alpha = float(self.cfg.blend_alpha)
        fused = alpha * approx_norm + (1.0 - alpha) * exact_norm
        fused[~valid] = -np.inf
        ranking = np.argsort(-fused, axis=1)
        fused_scores, fused_indices = _gather_ranked(
            candidate_scores=fused,
            candidate_indices=indices,
            ranking=ranking,
            top_k=k,
        )
        return fused_scores, fused_indices


@dataclass(frozen=True)
class MMRReranker(EmbeddingBlendReranker):
    """Greedy MMR reranking for relevance/diversity balance."""

    def rerank(
        self,
        *,
        query_embeddings: np.ndarray,
        candidate_scores: np.ndarray,
        candidate_indices: np.ndarray,
        candidate_embeddings: np.ndarray | None,
        metric: RetrievalDistanceMetric,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        k = min(int(top_k), int(candidate_scores.shape[1]))
        if candidate_embeddings is None:
            LOGGER.warning("MMR reranker enabled but candidate embeddings are unavailable; using ANN order.")
            return candidate_scores[:, :k], candidate_indices[:, :k]

        # Start from blended relevance scores.
        relevance_scores, ranked_indices = super().rerank(
            query_embeddings=query_embeddings,
            candidate_scores=candidate_scores,
            candidate_indices=candidate_indices,
            candidate_embeddings=candidate_embeddings,
            metric=metric,
            top_k=int(candidate_scores.shape[1]),
        )
        bsz, cand_k = relevance_scores.shape
        if cand_k == 0:
            return relevance_scores, ranked_indices

        # Reorder embeddings to match the blended ranking.
        row_ids = np.arange(bsz)[:, None]
        rank_positions = np.argsort(-relevance_scores, axis=1)
        ordered_indices = ranked_indices[row_ids, rank_positions]
        ordered_relevance = relevance_scores[row_ids, rank_positions]

        original_indices = np.asarray(candidate_indices, dtype=np.int64, copy=False)
        id_to_pos: list[dict[int, int]] = []
        for i in range(bsz):
            mapping: dict[int, int] = {}
            for pos, idx in enumerate(original_indices[i]):
                mapping[int(idx)] = pos
            id_to_pos.append(mapping)

        ordered_embeddings = np.zeros_like(candidate_embeddings, dtype=np.float32)
        for i in range(bsz):
            for j in range(cand_k):
                idx = int(ordered_indices[i, j])
                pos = id_to_pos[i].get(idx)
                if pos is None or idx < 0:
                    continue
                ordered_embeddings[i, j, :] = candidate_embeddings[i, pos, :]

        ordered_embeddings = _l2_normalize_3d(ordered_embeddings)
        lambda_weight = float(self.cfg.mmr_lambda)

        final_scores = np.full((bsz, k), -np.inf, dtype=np.float32)
        final_indices = np.full((bsz, k), -1, dtype=np.int64)

        for b in range(bsz):
            valid = ordered_indices[b] >= 0
            if not np.any(valid):
                continue
            rel = ordered_relevance[b]
            emb = ordered_embeddings[b]
            selected: list[int] = []
            remaining = set(int(i) for i in np.where(valid)[0])

            while remaining and len(selected) < k:
                if not selected:
                    best = max(remaining, key=lambda i: float(rel[i]))
                else:
                    sel_emb = emb[selected]  # [S, D]
                    mmr_best = None
                    mmr_best_val = -np.inf
                    for i in remaining:
                        div = float(np.max(emb[i] @ sel_emb.T))
                        mmr_val = lambda_weight * float(rel[i]) - (1.0 - lambda_weight) * div
                        if mmr_val > mmr_best_val:
                            mmr_best_val = mmr_val
                            mmr_best = i
                    best = int(mmr_best) if mmr_best is not None else max(
                        remaining, key=lambda i: float(rel[i])
                    )
                selected.append(best)
                remaining.remove(best)

            take = min(k, len(selected))
            for out_pos in range(take):
                in_pos = selected[out_pos]
                final_indices[b, out_pos] = int(ordered_indices[b, in_pos])
                final_scores[b, out_pos] = float(rel[in_pos])

        return final_scores, final_indices


def create_reranker(cfg: RetrievalRerankerConfig) -> BaseRetrievalReranker:
    name = str(cfg.name).strip().lower()
    if not cfg.enabled or name == "none":
        return NoOpReranker(cfg=cfg)
    if name == "blend":
        return EmbeddingBlendReranker(cfg=cfg)
    if name == "mmr":
        return MMRReranker(cfg=cfg)
    LOGGER.warning("Unknown reranker '%s'; falling back to no-op.", name)
    return NoOpReranker(cfg=cfg)


__all__ = [
    "BaseRetrievalReranker",
    "EmbeddingBlendReranker",
    "MMRReranker",
    "NoOpReranker",
    "create_reranker",
]

