"""Tests for retrieval metrics computation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.evaluation.metrics import (
    compute_match_ranks,
    compute_retrieval_metrics_from_ranks,
    evaluate_retrieval_direction,
)


def test_compute_match_ranks_finds_diagonal_matches() -> None:
    sim = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    ranks = compute_match_ranks(
        similarity=sim,
        query_ids=["a", "b"],
        gallery_ids=["a", "b"],
    )
    assert ranks.tolist() == [1, 1]


def test_compute_retrieval_metrics_from_ranks_values() -> None:
    ranks = torch.tensor([1, 2, 4], dtype=torch.long)
    metrics = compute_retrieval_metrics_from_ranks(ranks=ranks, ks=[1, 2, 4])
    assert metrics.recall_at_k[1] == pytest.approx(1.0 / 3.0)
    assert metrics.recall_at_k[2] == pytest.approx(2.0 / 3.0)
    assert metrics.recall_at_k[4] == pytest.approx(1.0)


def test_evaluate_retrieval_direction_rejects_bad_shape() -> None:
    sim = torch.zeros((2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected similarity matrix shape"):
        evaluate_retrieval_direction(
            similarity=sim,
            query_ids=["a", "b"],
            gallery_ids=["a", "b"],
            ks=[1, 5],
        )
