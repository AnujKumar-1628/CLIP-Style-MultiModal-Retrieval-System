"""Vector index builders for retrieval search.

Backends:
- `flat`: exact brute-force search (numpy)
- `ivfpq`: FAISS IVF + Product Quantization
- `ivfsq`: FAISS IVF + Scalar Quantization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.retrieval.storage import (
    build_index_artifact_paths,
    deserialize_pickle,
    load_faiss_index,
    load_index_binary,
    save_faiss_index,
    save_index_binary,
    serialize_pickle,
)
from src.utils.config import (
    RetrievalDistanceMetric,
    RetrievalIndexBackend,
    RetrievalIndexConfig,
    RetrievalStorageConfig,
)
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_index",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _as_float32_2d(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings [N, D], got shape={arr.shape}.")
    return arr


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms


def _prepare_for_metric(matrix: np.ndarray, metric: RetrievalDistanceMetric) -> np.ndarray:
    arr = _as_float32_2d(matrix)
    if metric == "cosine":
        return _l2_normalize(arr)
    return arr


class RetrievalIndex(Protocol):
    backend: RetrievalIndexBackend
    metric: RetrievalDistanceMetric
    dim: int

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        ...


@dataclass
class NumpyFlatIndex:
    """Exact brute-force index implemented with NumPy."""

    metric: RetrievalDistanceMetric
    embeddings: np.ndarray
    backend: RetrievalIndexBackend = "flat"

    def __post_init__(self) -> None:
        self.embeddings = _prepare_for_metric(self.embeddings, self.metric)
        self.dim = int(self.embeddings.shape[1])

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _prepare_for_metric(queries, self.metric)
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim {q.shape[1]} does not match index dim {self.dim}.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")
        k = min(top_k, int(self.embeddings.shape[0]))

        if self.metric in {"cosine", "ip"}:
            scores = q @ self.embeddings.T
            order = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(scores.shape[0])[:, None]
            top_scores = scores[row_idx, order]
            sorted_order = np.argsort(-top_scores, axis=1)
            top_indices = order[row_idx, sorted_order]
            top_scores = top_scores[row_idx, sorted_order]
            return top_scores, top_indices

        # L2 distance: smaller is better. For consistency, return negative distance as score.
        q2 = np.sum(q * q, axis=1, keepdims=True)
        x2 = np.sum(self.embeddings * self.embeddings, axis=1)[None, :]
        dist = q2 + x2 - 2.0 * (q @ self.embeddings.T)
        order = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(dist.shape[0])[:, None]
        top_dist = dist[row_idx, order]
        sorted_order = np.argsort(top_dist, axis=1)
        top_indices = order[row_idx, sorted_order]
        top_dist = top_dist[row_idx, sorted_order]
        return -top_dist, top_indices


@dataclass
class FaissApproxIndex:
    """FAISS-backed ANN index supporting IVFPQ/IVFSQ."""

    metric: RetrievalDistanceMetric
    backend: RetrievalIndexBackend
    index: Any
    dim: int

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        q = _prepare_for_metric(queries, self.metric)
        scores, indices = self.index.search(q.astype(np.float32), int(top_k))
        return scores, indices


@dataclass
class RetrievalIndexBundle:
    """Index plus ID mapping and metadata."""

    index: RetrievalIndex
    ids: list[str]
    metadata: dict[str, Any]


def _faiss_metric(metric: RetrievalDistanceMetric) -> int:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError("faiss is required for IVFPQ/IVFSQ backends.") from exc

    if metric in {"cosine", "ip"}:
        return faiss.METRIC_INNER_PRODUCT
    return faiss.METRIC_L2


def _build_faiss_ann_index(
    embeddings: np.ndarray,
    cfg: RetrievalIndexConfig,
) -> FaissApproxIndex:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError("faiss is required for backend != 'flat'.") from exc

    x = _prepare_for_metric(embeddings, cfg.metric).astype(np.float32)
    n, dim = x.shape
    metric = _faiss_metric(cfg.metric)

    quantizer = (
        faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
    )
    if cfg.backend == "ivfpq":
        index = faiss.IndexIVFPQ(
            quantizer,
            dim,
            int(cfg.nlist),
            int(cfg.pq_m),
            int(cfg.pq_nbits),
            metric,
        )
    else:
        # Scalar quantization backend.
        sq_qtype = str(cfg.sq_qtype).strip().lower()
        qtype_map = {
            "8bit": faiss.ScalarQuantizer.QT_8bit,
            "4bit": faiss.ScalarQuantizer.QT_4bit,
            "fp16": faiss.ScalarQuantizer.QT_fp16,
        }
        if sq_qtype not in qtype_map:
            valid = ", ".join(sorted(qtype_map))
            raise ValueError(f"Unsupported retrieval.index.sq_qtype '{cfg.sq_qtype}'. Valid: {valid}")
        index = faiss.IndexIVFScalarQuantizer(
            quantizer,
            dim,
            int(cfg.nlist),
            qtype_map[sq_qtype],
            metric,
        )

    train_n = min(int(cfg.train_sample_size), n)
    if train_n <= 0:
        raise ValueError("No vectors available to train FAISS index.")
    train_x = x[:train_n]
    index.train(train_x)
    index.add(x)
    index.nprobe = int(cfg.nprobe)

    LOGGER.info(
        "Built FAISS index backend=%s vectors=%d dim=%d nlist=%d nprobe=%d",
        cfg.backend,
        n,
        dim,
        cfg.nlist,
        cfg.nprobe,
    )
    return FaissApproxIndex(
        metric=cfg.metric,
        backend=cfg.backend,
        index=index,
        dim=dim,
    )


def build_retrieval_index(
    *,
    embeddings: np.ndarray,
    ids: list[str],
    cfg: RetrievalIndexConfig,
) -> RetrievalIndexBundle:
    x = _as_float32_2d(embeddings)
    if len(ids) != int(x.shape[0]):
        raise ValueError(f"IDs length ({len(ids)}) must match vectors rows ({x.shape[0]}).")
    if cfg.backend == "flat":
        index: RetrievalIndex = NumpyFlatIndex(metric=cfg.metric, embeddings=x)
    else:
        index = _build_faiss_ann_index(x, cfg)

    metadata = {
        "backend": cfg.backend,
        "metric": cfg.metric,
        "count": int(x.shape[0]),
        "dim": int(x.shape[1]),
    }
    return RetrievalIndexBundle(index=index, ids=[str(v) for v in ids], metadata=metadata)


def save_retrieval_index(
    *,
    bundle: RetrievalIndexBundle,
    storage_cfg: RetrievalStorageConfig,
    name: str,
) -> None:
    payload_meta = dict(bundle.metadata)
    payload_meta["ids"] = bundle.ids

    if isinstance(bundle.index, FaissApproxIndex):
        save_faiss_index(
            cfg=storage_cfg,
            name=name,
            faiss_index=bundle.index.index,
            metadata=payload_meta,
        )
        return

    if isinstance(bundle.index, NumpyFlatIndex):
        payload = {
            "backend": bundle.index.backend,
            "metric": bundle.index.metric,
            "embeddings": bundle.index.embeddings,
            "dim": bundle.index.dim,
        }
        binary = serialize_pickle(payload)
        save_index_binary(
            cfg=storage_cfg,
            name=name,
            binary=binary,
            metadata=payload_meta,
        )
        return

    raise TypeError(f"Unsupported index type for serialization: {type(bundle.index).__name__}")


def load_retrieval_index(
    *,
    storage_cfg: RetrievalStorageConfig,
    name: str,
) -> RetrievalIndexBundle:
    def _load_backend_hint() -> str | None:
        paths = build_index_artifact_paths(cfg=storage_cfg, name=name)
        if not paths.metadata.exists():
            return None
        import json

        with paths.metadata.open("r", encoding="utf-8") as handle:
            raw_meta = dict(json.load(handle) or {})
        backend_raw = raw_meta.get("backend")
        if backend_raw is None:
            return None
        return str(backend_raw).strip().lower()

    def _load_faiss_bundle() -> RetrievalIndexBundle:
        faiss_idx, meta = load_faiss_index(cfg=storage_cfg, name=name)
        backend = str(meta.get("backend", "ivfpq")).strip().lower()
        metric = str(meta.get("metric", "cosine")).strip().lower()
        ids = [str(v) for v in meta.get("ids", [])]
        if not ids:
            raise ValueError("Saved FAISS index metadata missing `ids`.")
        dim = int(faiss_idx.d)
        index = FaissApproxIndex(
            metric=metric,  # type: ignore[arg-type]
            backend=backend,  # type: ignore[arg-type]
            index=faiss_idx,
            dim=dim,
        )
        meta.pop("ids", None)
        return RetrievalIndexBundle(index=index, ids=ids, metadata=meta)

    def _load_flat_bundle() -> RetrievalIndexBundle:
        binary, meta = load_index_binary(cfg=storage_cfg, name=name)
        payload = deserialize_pickle(binary)
        backend = str(payload.get("backend", "flat")).strip().lower()
        metric = str(payload.get("metric", "cosine")).strip().lower()
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        ids = [str(v) for v in meta.get("ids", [])]
        if len(ids) != int(embeddings.shape[0]):
            raise ValueError(
                "Loaded flat index IDs length does not match vector rows: "
                f"{len(ids)} vs {embeddings.shape[0]}."
            )
        index = NumpyFlatIndex(
            metric=metric,  # type: ignore[arg-type]
            embeddings=embeddings,
            backend=backend,  # type: ignore[arg-type]
        )
        meta.pop("ids", None)
        return RetrievalIndexBundle(index=index, ids=ids, metadata=meta)

    backend_hint = _load_backend_hint()
    if backend_hint == "flat":
        return _load_flat_bundle()
    if backend_hint in {"ivfpq", "ivfsq"}:
        try:
            return _load_faiss_bundle()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS index '{name}' (backend={backend_hint})."
            ) from exc
    if backend_hint is not None:
        raise ValueError(
            f"Unsupported saved index backend '{backend_hint}' for index '{name}'."
        )

    # Legacy fallback for artifacts that don't record backend in metadata.
    faiss_error: Exception | None = None
    try:
        return _load_faiss_bundle()
    except Exception as exc:
        faiss_error = exc

    try:
        return _load_flat_bundle()
    except Exception as flat_exc:
        raise RuntimeError(
            f"Failed to load index '{name}' as either FAISS or flat pickle. "
            f"FAISS error: {faiss_error!r} | Flat error: {flat_exc!r}"
        ) from flat_exc


__all__ = [
    "FaissApproxIndex",
    "NumpyFlatIndex",
    "RetrievalIndex",
    "RetrievalIndexBundle",
    "build_retrieval_index",
    "load_retrieval_index",
    "save_retrieval_index",
]
