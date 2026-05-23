"""High-level retrieval orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.clip_model import CLIPModel, create_clip_model
from src.retrieval.embed import build_and_save_retrieval_embeddings
from src.retrieval.index import (
    RetrievalIndexBundle,
    build_retrieval_index,
    load_retrieval_index,
    save_retrieval_index,
)
from src.retrieval.rerankers import create_reranker
from src.retrieval.search import RetrievalSearcher, SearchRequest
from src.retrieval.storage import load_embedding_artifacts
from src.utils.config import (
    RetrievalConfig,
    RetrievalEvalConfig,
    load_retrieval_config,
    load_retrieval_eval_config,
)
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_pipeline",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _load_model_checkpoint_if_configured(
    *,
    model: CLIPModel,
    eval_cfg: RetrievalEvalConfig,
) -> None:
    if eval_cfg.checkpoint.path is None:
        LOGGER.warning("No checkpoint configured; retrieval searcher will use fresh model weights.")
        return
    payload = torch.load(eval_cfg.checkpoint.path, map_location=model.device_obj)
    if isinstance(payload, dict):
        state_dict = payload.get(eval_cfg.checkpoint.state_dict_key, payload)
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint payload does not contain a valid state_dict mapping.")
    model.load_state_dict(state_dict, strict=eval_cfg.checkpoint.strict)
    LOGGER.info("Loaded checkpoint for retrieval searcher: %s", eval_cfg.checkpoint.path)


def _load_candidate_embeddings_for_bundle(
    *,
    retrieval_cfg: RetrievalConfig,
    bundle: RetrievalIndexBundle,
) -> np.ndarray | None:
    source_artifact = str(bundle.metadata.get("source_artifact", "")).strip()
    if not source_artifact:
        return None
    try:
        embeddings, ids, _ = load_embedding_artifacts(
            cfg=retrieval_cfg.storage,
            name=source_artifact,
        )
    except Exception as exc:
        LOGGER.warning(
            "Failed to load candidate embeddings for reranking from '%s': %s",
            source_artifact,
            exc,
        )
        return None

    if len(ids) != len(bundle.ids):
        LOGGER.warning(
            "Reranker candidate embeddings row-count mismatch for '%s': %d vs %d.",
            source_artifact,
            len(ids),
            len(bundle.ids),
        )
        return None
    if list(ids) != list(bundle.ids):
        LOGGER.warning(
            "Reranker candidate IDs mismatch for '%s'; reranking disabled for this index.",
            source_artifact,
        )
        return None
    return np.asarray(embeddings, dtype=np.float32)


def build_index_from_embedding_artifact(
    *,
    retrieval_cfg: RetrievalConfig,
    artifact_name: str,
    index_name: str,
) -> RetrievalIndexBundle:
    embeddings, ids, emb_meta = load_embedding_artifacts(
        cfg=retrieval_cfg.storage,
        name=artifact_name,
    )
    bundle = build_retrieval_index(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        ids=ids,
        cfg=retrieval_cfg.index,
    )
    merged_meta: dict[str, Any] = dict(bundle.metadata)
    merged_meta.update({"source_artifact": artifact_name})
    merged_meta.update({f"embedding_{k}": v for k, v in emb_meta.items()})
    bundle = RetrievalIndexBundle(index=bundle.index, ids=bundle.ids, metadata=merged_meta)
    save_retrieval_index(
        bundle=bundle,
        storage_cfg=retrieval_cfg.storage,
        name=index_name,
    )
    LOGGER.info("Built index '%s' from artifact '%s'.", index_name, artifact_name)
    return bundle


def build_retrieval_indices(
    *,
    retrieval_config_path: str | Path | None = None,
    image_embedding_artifact: str = "image_corpus",
    text_embedding_artifact: str = "text_corpus",
    image_index_name: str = "image_index",
    text_index_name: str = "text_index",
) -> dict[str, RetrievalIndexBundle]:
    cfg = load_retrieval_config(retrieval_config_path)
    image_bundle = build_index_from_embedding_artifact(
        retrieval_cfg=cfg,
        artifact_name=image_embedding_artifact,
        index_name=image_index_name,
    )
    text_bundle = build_index_from_embedding_artifact(
        retrieval_cfg=cfg,
        artifact_name=text_embedding_artifact,
        index_name=text_index_name,
    )
    return {"image": image_bundle, "text": text_bundle}


def load_retrieval_searcher(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    image_index_name: str = "image_index",
    text_index_name: str = "text_index",
) -> RetrievalSearcher:
    cfg = load_retrieval_config(retrieval_config_path)
    eval_cfg = load_retrieval_eval_config(retrieval_config_path)
    model: CLIPModel = create_clip_model(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    _load_model_checkpoint_if_configured(model=model, eval_cfg=eval_cfg)
    model.eval()

    image_bundle = load_retrieval_index(
        storage_cfg=cfg.storage,
        name=image_index_name,
    )
    try:
        text_bundle = load_retrieval_index(
            storage_cfg=cfg.storage,
            name=text_index_name,
        )
    except Exception:
        text_bundle = None

    reranker = create_reranker(cfg.reranker)
    image_candidate_embeddings = _load_candidate_embeddings_for_bundle(
        retrieval_cfg=cfg,
        bundle=image_bundle,
    )
    text_candidate_embeddings = (
        _load_candidate_embeddings_for_bundle(
            retrieval_cfg=cfg,
            bundle=text_bundle,
        )
        if text_bundle is not None
        else None
    )

    return RetrievalSearcher(
        model=model,
        image_index=image_bundle,
        text_index=text_bundle,
        cfg=cfg.search,
        reranker=reranker,
        image_candidate_embeddings=image_candidate_embeddings,
        text_candidate_embeddings=text_candidate_embeddings,
    )


def search_text_to_image(
    *,
    query: str,
    top_k: int = 10,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    searcher = load_retrieval_searcher(
        retrieval_config_path=retrieval_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    rows = searcher.search(
        SearchRequest(
            query_type="text",
            target_modality="image",
            top_k=top_k,
            text=query,
        )
    )
    return [
        {
            "rank": hit.rank,
            "item_id": hit.item_id,
            "score": hit.score,
            "metadata": hit.metadata,
        }
        for hit in rows[0]
    ]


def run_retrieval_build_pipeline(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    build_and_save_retrieval_embeddings(
        retrieval_config_path=retrieval_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        split=split,
    )
    bundles = build_retrieval_indices(retrieval_config_path=retrieval_config_path)
    return {
        "image_index_size": len(bundles["image"].ids),
        "text_index_size": len(bundles["text"].ids),
        "image_index_backend": bundles["image"].metadata.get("backend"),
        "text_index_backend": bundles["text"].metadata.get("backend"),
    }


__all__ = [
    "build_index_from_embedding_artifact",
    "build_retrieval_indices",
    "load_retrieval_searcher",
    "run_retrieval_build_pipeline",
    "search_text_to_image",
]
