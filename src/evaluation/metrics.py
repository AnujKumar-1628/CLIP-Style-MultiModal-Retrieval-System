"""Retrieval metrics for CLIP-style evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class RetrievalDirectionMetrics:
    """Metrics for a single retrieval direction."""

    recall_at_k: dict[int, float]
    mrr: float
    mean_rank: float
    median_rank: float
    query_count: int

    def to_dict(self, *, prefix: str) -> dict[str, float]:
        payload: dict[str, float] = {
            f"{prefix}/mrr": self.mrr,
            f"{prefix}/mean_rank": self.mean_rank,
            f"{prefix}/median_rank": self.median_rank,
            f"{prefix}/query_count": float(self.query_count),
        }
        for k, value in sorted(self.recall_at_k.items()):
            payload[f"{prefix}/r@{k}"] = value
        return payload


def _to_python_str_list(values: Iterable[object]) -> list[str]:
    return [str(value) for value in values]


def compute_match_ranks(
    *,
    similarity: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> torch.Tensor:
    """Compute 1-based rank of first correct match for each query."""
    if similarity.ndim != 2:
        raise ValueError(
            f"Expected similarity matrix shape [Q, G], got {tuple(similarity.shape)}."
        )
    q_size, g_size = similarity.shape
    if len(query_ids) != q_size:
        raise ValueError(
            "Length of query_ids does not match similarity rows: "
            f"{len(query_ids)} vs {q_size}."
        )
    if len(gallery_ids) != g_size:
        raise ValueError(
            "Length of gallery_ids does not match similarity cols: "
            f"{len(gallery_ids)} vs {g_size}."
        )

    gallery = _to_python_str_list(gallery_ids)
    query = _to_python_str_list(query_ids)

    ranks = torch.full((q_size,), g_size + 1, dtype=torch.long)
    order = torch.argsort(similarity, dim=1, descending=True)
    for q_idx in range(q_size):
        query_id = query[q_idx]
        ordered_cols = order[q_idx].tolist()
        for rank_idx, col_idx in enumerate(ordered_cols, start=1):
            if gallery[col_idx] == query_id:
                ranks[q_idx] = rank_idx
                break
    return ranks


def compute_retrieval_metrics_from_ranks(
    *,
    ranks: torch.Tensor,
    ks: list[int],
) -> RetrievalDirectionMetrics:
    """Aggregate retrieval metrics from per-query ranks."""
    if ranks.ndim != 1:
        raise ValueError(f"Expected 1D ranks tensor, got {tuple(ranks.shape)}.")
    if ranks.numel() == 0:
        raise ValueError("Ranks tensor is empty.")
    if any(k <= 0 for k in ks):
        raise ValueError("All k values must be > 0.")

    ranks_f = ranks.float()
    query_count = int(ranks.numel())
    recall_at_k = {
        int(k): float((ranks <= int(k)).float().mean().item())
        for k in sorted(set(ks))
    }
    mrr = float((1.0 / ranks_f).mean().item())
    mean_rank = float(ranks_f.mean().item())
    median_rank = float(ranks_f.median().item())

    return RetrievalDirectionMetrics(
        recall_at_k=recall_at_k,
        mrr=mrr,
        mean_rank=mean_rank,
        median_rank=median_rank,
        query_count=query_count,
    )


def evaluate_retrieval_direction(
    *,
    similarity: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
    ks: list[int],
) -> tuple[RetrievalDirectionMetrics, torch.Tensor]:
    """Compute full metrics and ranks for one retrieval direction."""
    ranks = compute_match_ranks(
        similarity=similarity,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
    )
    metrics = compute_retrieval_metrics_from_ranks(ranks=ranks, ks=ks)
    return metrics, ranks


__all__ = [
    "RetrievalDirectionMetrics",
    "compute_match_ranks",
    "compute_retrieval_metrics_from_ranks",
    "evaluate_retrieval_direction",
]
