"""Evaluation pipeline for CLIP-style retrieval."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from src.data_logic.eval_datamodule import create_eval_datamodule
from src.evaluation.metrics import (
    RetrievalDirectionMetrics,
    evaluate_retrieval_direction,
)
from src.models.clip_model import CLIPModel, create_clip_model
from src.utils.config import RetrievalEvalConfig, load_retrieval_eval_config
from src.utils.logger import setup_logger
from src.utils.paths import ensure_base_dirs

LOGGER = setup_logger(
    name="evaluation_pipeline",
    level="INFO",
    use_console=True,
    use_file=True,
)

@dataclass(frozen=True)
class EmbeddingBatch:
    embeddings: torch.Tensor
    image_ids: list[str]


@dataclass(frozen=True)
class RetrievalEvalResult:
    config: RetrievalEvalConfig
    text_to_image: RetrievalDirectionMetrics | None
    image_to_text: RetrievalDirectionMetrics | None
    text_to_image_ranks: list[int] | None
    image_to_text_ranks: list[int] | None

    def to_report_dict(self) -> dict[str, object]:
        def _to_json_safe(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {str(k): _to_json_safe(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_to_json_safe(v) for v in value]
            return value

        payload: dict[str, object] = {
            "config": _to_json_safe(asdict(self.config)),
            "metrics": {},
        }
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        if self.text_to_image is not None:
            metrics["text_to_image"] = {
                "recall_at_k": self.text_to_image.recall_at_k,
                "mrr": self.text_to_image.mrr,
                "mean_rank": self.text_to_image.mean_rank,
                "median_rank": self.text_to_image.median_rank,
                "query_count": self.text_to_image.query_count,
            }
        if self.image_to_text is not None:
            metrics["image_to_text"] = {
                "recall_at_k": self.image_to_text.recall_at_k,
                "mrr": self.image_to_text.mrr,
                "mean_rank": self.image_to_text.mean_rank,
                "median_rank": self.image_to_text.median_rank,
                "query_count": self.image_to_text.query_count,
            }
        if self.text_to_image_ranks is not None:
            payload["text_to_image_ranks"] = self.text_to_image_ranks
        if self.image_to_text_ranks is not None:
            payload["image_to_text_ranks"] = self.image_to_text_ranks
        return payload


def _load_model_checkpoint(
    model: CLIPModel,
    cfg: RetrievalEvalConfig,
) -> None:
    if cfg.checkpoint.path is None:
        LOGGER.warning(
            "No checkpoint configured for evaluation; metrics will reflect a freshly "
            "initialized model. Set retrieval.checkpoint.path or pass a checkpoint "
            "path to scripts/evaluate.sh."
        )
        return
    payload = torch.load(cfg.checkpoint.path, map_location=model.device_obj)
    if isinstance(payload, dict):
        state_dict = payload.get(cfg.checkpoint.state_dict_key, payload)
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint payload does not contain a valid state_dict mapping.")
    model.load_state_dict(state_dict, strict=cfg.checkpoint.strict)
    LOGGER.info("Loaded checkpoint: %s", cfg.checkpoint.path)


def _autocast_context(*, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _encode_query_embeddings(
    model: CLIPModel,
    loader,
    *,
    use_amp: bool,
) -> EmbeddingBatch:
    embeddings: list[torch.Tensor] = []
    image_ids: list[str] = []
    amp_enabled = bool(use_amp and model.device_obj.type == "cuda")
    for batch in loader:
        with _autocast_context(enabled=amp_enabled):
            if "text_tokens" in batch and isinstance(batch["text_tokens"], dict):
                token_bundle = batch["text_tokens"]
                encoded = model.encode_text_tokens(
                    input_ids=token_bundle["input_ids"],
                    attention_mask=token_bundle.get("attention_mask"),
                    normalize=True,
                )
            else:
                texts = batch.get("texts")
                if texts is None:
                    raise ValueError("Query batch must contain `texts` or `text_tokens`.")
                encoded = model.encode_texts(texts, normalize=True)
        embeddings.append(encoded.detach().cpu())
        image_ids.extend([str(v) for v in batch["image_names"]])

    if not embeddings:
        raise ValueError("Query dataloader produced no batches.")
    return EmbeddingBatch(
        embeddings=torch.cat(embeddings, dim=0),
        image_ids=image_ids,
    )


def _encode_gallery_embeddings(
    model: CLIPModel,
    loader,
    *,
    use_amp: bool,
) -> EmbeddingBatch:
    embeddings: list[torch.Tensor] = []
    image_ids: list[str] = []
    amp_enabled = bool(use_amp and model.device_obj.type == "cuda")
    for batch in loader:
        images = batch["images"]
        if not isinstance(images, torch.Tensor):
            raise TypeError("Gallery batch key `images` must be a torch.Tensor.")
        with _autocast_context(enabled=amp_enabled):
            encoded = model.encode_images(images, normalize=True)
        embeddings.append(encoded.detach().cpu())
        image_ids.extend([str(v) for v in batch["image_names"]])

    if not embeddings:
        raise ValueError("Gallery dataloader produced no batches.")
    return EmbeddingBatch(
        embeddings=torch.cat(embeddings, dim=0),
        image_ids=image_ids,
    )


def compute_similarity_matrix(
    *,
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Compute [Q, G] cosine-similarity matrix in query chunks."""
    if query_embeddings.ndim != 2 or gallery_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape [N, D].")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"{query_embeddings.shape[1]} vs {gallery_embeddings.shape[1]}."
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    gallery_t = gallery_embeddings.t().contiguous()
    rows: list[torch.Tensor] = []
    total = int(query_embeddings.shape[0])
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        query_chunk = query_embeddings[start:stop]
        rows.append(query_chunk @ gallery_t)
    return torch.cat(rows, dim=0)


def evaluate_retrieval(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> RetrievalEvalResult:
    ensure_base_dirs()
    cfg = load_retrieval_eval_config(retrieval_config_path)
    if checkpoint_path is not None:
        cfg = replace(
            cfg,
            checkpoint=replace(cfg.checkpoint, path=Path(checkpoint_path)),
        )
    model = create_clip_model(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    _load_model_checkpoint(model, cfg)
    model.eval()

    datamodule = create_eval_datamodule(config_path=data_config_path)
    datamodule.prepare_data()
    datamodule.setup(stage=None, split=cfg.runtime.split)
    loaders = datamodule.retrieval_dataloaders(cfg.runtime.split)

    query_batch = _encode_query_embeddings(
        model,
        loaders["query"],
        use_amp=cfg.runtime.use_amp,
    )
    gallery_batch = _encode_gallery_embeddings(
        model,
        loaders["gallery"],
        use_amp=cfg.runtime.use_amp,
    )

    similarity = compute_similarity_matrix(
        query_embeddings=query_batch.embeddings,
        gallery_embeddings=gallery_batch.embeddings,
        chunk_size=cfg.runtime.similarity_chunk_size,
    )

    text_to_image_metrics: RetrievalDirectionMetrics | None = None
    image_to_text_metrics: RetrievalDirectionMetrics | None = None
    text_to_image_ranks: list[int] | None = None
    image_to_text_ranks: list[int] | None = None

    if "text_to_image" in cfg.runtime.directions:
        metrics, ranks = evaluate_retrieval_direction(
            similarity=similarity,
            query_ids=query_batch.image_ids,
            gallery_ids=gallery_batch.image_ids,
            ks=cfg.runtime.ks,
        )
        text_to_image_metrics = metrics
        text_to_image_ranks = ranks.tolist() if cfg.output.include_per_query else None

    if "image_to_text" in cfg.runtime.directions:
        metrics, ranks = evaluate_retrieval_direction(
            similarity=similarity.t(),
            query_ids=gallery_batch.image_ids,
            gallery_ids=query_batch.image_ids,
            ks=cfg.runtime.ks,
        )
        image_to_text_metrics = metrics
        image_to_text_ranks = ranks.tolist() if cfg.output.include_per_query else None

    result = RetrievalEvalResult(
        config=cfg,
        text_to_image=text_to_image_metrics,
        image_to_text=image_to_text_metrics,
        text_to_image_ranks=text_to_image_ranks,
        image_to_text_ranks=image_to_text_ranks,
    )
    if cfg.output.save_json:
        save_retrieval_report(result)
    return result


def save_retrieval_report(result: RetrievalEvalResult) -> Path:
    output_dir = result.config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / result.config.output.filename
    payload = result.to_report_dict()
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Saved retrieval report: %s", target)
    return target


__all__ = [
    "EmbeddingBatch",
    "RetrievalEvalResult",
    "compute_similarity_matrix",
    "evaluate_retrieval",
    "save_retrieval_report",
]
