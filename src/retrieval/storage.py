"""Storage utilities for retrieval artifacts.

Supports:
- embedding/id/metadata persistence
- memory-mapped embedding loading (numpy mmap)
- optional FAISS index read/write helpers
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.config import RetrievalStorageConfig
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_storage",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class EmbeddingArtifactPaths:
    embeddings: Path
    ids: Path
    metadata: Path


@dataclass(frozen=True)
class IndexArtifactPaths:
    index: Path
    metadata: Path


def _dtype_from_name(name: str) -> np.dtype:
    key = name.strip().lower()
    if key in {"float32", "fp32"}:
        return np.float32
    if key in {"float16", "fp16"}:
        return np.float16
    if key in {"float64", "fp64"}:
        return np.float64
    raise ValueError(f"Unsupported embedding dtype '{name}'.")


def ensure_retrieval_dirs(cfg: RetrievalStorageConfig) -> None:
    cfg.embeddings_dir.mkdir(parents=True, exist_ok=True)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)


def build_embedding_artifact_paths(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
) -> EmbeddingArtifactPaths:
    ensure_retrieval_dirs(cfg)
    key = str(name).strip()
    return EmbeddingArtifactPaths(
        embeddings=cfg.embeddings_dir / f"{key}.npy",
        ids=cfg.metadata_dir / f"{key}_ids.json",
        metadata=cfg.metadata_dir / f"{key}_meta.json",
    )


def build_index_artifact_paths(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
) -> IndexArtifactPaths:
    ensure_retrieval_dirs(cfg)
    key = str(name).strip()
    return IndexArtifactPaths(
        index=cfg.index_dir / f"{key}.bin",
        metadata=cfg.metadata_dir / f"{key}_index_meta.json",
    )


def _to_numpy(embeddings: torch.Tensor | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        arr = embeddings.detach().cpu().numpy()
    else:
        arr = np.asarray(embeddings)
    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be rank-2 [N, D], got shape {arr.shape}.")
    return arr.astype(dtype, copy=False)


def save_embedding_artifacts(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
    embeddings: torch.Tensor | np.ndarray,
    ids: list[str],
    metadata: dict[str, Any] | None = None,
) -> EmbeddingArtifactPaths:
    paths = build_embedding_artifact_paths(cfg=cfg, name=name)
    dtype = _dtype_from_name(cfg.embedding_dtype)
    arr = _to_numpy(embeddings, dtype=dtype)
    if len(ids) != int(arr.shape[0]):
        raise ValueError(
            f"IDs length ({len(ids)}) must match embeddings rows ({arr.shape[0]})."
        )

    np.save(paths.embeddings, arr)
    with paths.ids.open("w", encoding="utf-8") as handle:
        json.dump([str(v) for v in ids], handle, indent=2)
    with paths.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)
    LOGGER.info("Saved embedding artifacts: %s", paths)
    return paths


def load_embedding_artifacts(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    paths = build_embedding_artifact_paths(cfg=cfg, name=name)
    mmap_mode = "r" if cfg.use_mmap else None
    embeddings = np.load(paths.embeddings, mmap_mode=mmap_mode)
    with paths.ids.open("r", encoding="utf-8") as handle:
        ids = [str(v) for v in json.load(handle)]
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = dict(json.load(handle) or {})

    if len(ids) != int(embeddings.shape[0]):
        raise ValueError(
            f"Loaded ids length ({len(ids)}) != embeddings rows ({embeddings.shape[0]})."
        )
    return embeddings, ids, metadata


def save_index_binary(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
    binary: bytes,
    metadata: dict[str, Any] | None = None,
) -> IndexArtifactPaths:
    paths = build_index_artifact_paths(cfg=cfg, name=name)
    paths.index.write_bytes(binary)
    with paths.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)
    LOGGER.info("Saved index artifact: %s", paths.index)
    return paths


def load_index_binary(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
) -> tuple[bytes, dict[str, Any]]:
    paths = build_index_artifact_paths(cfg=cfg, name=name)
    binary = paths.index.read_bytes()
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = dict(json.load(handle) or {})
    return binary, metadata


def serialize_pickle(payload: Any) -> bytes:
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_pickle(binary: bytes) -> Any:
    return pickle.loads(binary)


def save_faiss_index(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
    faiss_index: Any,
    metadata: dict[str, Any] | None = None,
) -> IndexArtifactPaths:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError("faiss is required to save FAISS indices.") from exc

    paths = build_index_artifact_paths(cfg=cfg, name=name)
    faiss.write_index(faiss_index, str(paths.index))
    with paths.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata or {}, handle, indent=2)
    LOGGER.info("Saved FAISS index: %s", paths.index)
    return paths


def load_faiss_index(
    *,
    cfg: RetrievalStorageConfig,
    name: str,
) -> tuple[Any, dict[str, Any]]:
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise ImportError("faiss is required to load FAISS indices.") from exc

    paths = build_index_artifact_paths(cfg=cfg, name=name)
    index = faiss.read_index(str(paths.index))
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = dict(json.load(handle) or {})
    return index, metadata


__all__ = [
    "EmbeddingArtifactPaths",
    "IndexArtifactPaths",
    "build_embedding_artifact_paths",
    "build_index_artifact_paths",
    "deserialize_pickle",
    "ensure_retrieval_dirs",
    "load_embedding_artifacts",
    "load_faiss_index",
    "load_index_binary",
    "save_embedding_artifacts",
    "save_faiss_index",
    "save_index_binary",
    "serialize_pickle",
]
