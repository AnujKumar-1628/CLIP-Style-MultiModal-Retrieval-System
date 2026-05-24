"""Evaluation package exports."""

from src.evaluation.evaluate import (
    EmbeddingBatch,
    RetrievalEvalResult,
    compute_similarity_matrix,
    evaluate_retrieval,
    save_retrieval_report,
)
from src.evaluation.metrics import (
    RetrievalDirectionMetrics,
    compute_match_ranks,
    compute_retrieval_metrics_from_ranks,
    evaluate_retrieval_direction,
)

__all__ = [
    "EmbeddingBatch",
    "RetrievalDirectionMetrics",
    "RetrievalEvalResult",
    "compute_match_ranks",
    "compute_retrieval_metrics_from_ranks",
    "compute_similarity_matrix",
    "evaluate_retrieval",
    "evaluate_retrieval_direction",
    "save_retrieval_report",
]
