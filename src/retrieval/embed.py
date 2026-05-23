"""Embedding jobs for retrieval indexing."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data_logic.eval_datamodule import create_eval_datamodule
from src.models.clip_model import CLIPModel, create_clip_model
from src.retrieval.storage import save_embedding_artifacts
from src.utils.config import RetrievalConfig, load_retrieval_config
from src.utils.config import load_retrieval_eval_config
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="retrieval_embed",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class EmbeddedCorpus:
    embeddings: np.ndarray
    ids: list[str]
    modality: str
    split: str


def _autocast_context(*, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def embed_gallery_images(
    *,
    model: CLIPModel,
    loader,
    use_amp: bool,
    normalize: bool,
    split: str,
) -> EmbeddedCorpus:
    amp_enabled = bool(use_amp and model.device_obj.type == "cuda")
    chunks: list[np.ndarray] = []
    ids: list[str] = []
    for batch in loader:
        images = batch["images"]
        if not isinstance(images, torch.Tensor):
            raise TypeError("Expected batch['images'] to be a torch.Tensor.")
        with _autocast_context(enabled=amp_enabled):
            emb = model.encode_images(images, normalize=normalize)
        chunks.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
        ids.extend([str(v) for v in batch["image_names"]])

    if not chunks:
        raise ValueError("Gallery loader produced no samples.")
    matrix = np.concatenate(chunks, axis=0)
    return EmbeddedCorpus(
        embeddings=matrix,
        ids=ids,
        modality="image",
        split=str(split),
    )


def embed_query_texts(
    *,
    model: CLIPModel,
    loader,
    use_amp: bool,
    normalize: bool,
    split: str,
) -> EmbeddedCorpus:
    amp_enabled = bool(use_amp and model.device_obj.type == "cuda")
    chunks: list[np.ndarray] = []
    ids: list[str] = []
    for batch in loader:
        with _autocast_context(enabled=amp_enabled):
            if "text_tokens" in batch and isinstance(batch["text_tokens"], dict):
                bundle = batch["text_tokens"]
                emb = model.encode_text_tokens(
                    input_ids=bundle["input_ids"],
                    attention_mask=bundle.get("attention_mask"),
                    normalize=normalize,
                )
            else:
                texts = batch.get("texts")
                if texts is None:
                    raise ValueError("Query loader batch missing both `text_tokens` and `texts`.")
                emb = model.encode_texts(texts, normalize=normalize)

        chunks.append(emb.detach().cpu().numpy().astype(np.float32, copy=False))
        ids.extend([str(v) for v in batch["image_names"]])

    if not chunks:
        raise ValueError("Query loader produced no samples.")
    matrix = np.concatenate(chunks, axis=0)
    return EmbeddedCorpus(
        embeddings=matrix,
        ids=ids,
        modality="text",
        split=str(split),
    )


def build_and_save_retrieval_embeddings(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    split: str = "test",
    image_artifact_name: str = "image_corpus",
    text_artifact_name: str = "text_corpus",
) -> dict[str, object]:
    cfg: RetrievalConfig = load_retrieval_config(retrieval_config_path)
    model = create_clip_model(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    eval_cfg = load_retrieval_eval_config(retrieval_config_path)
    if eval_cfg.checkpoint.path is not None:
        payload = torch.load(eval_cfg.checkpoint.path, map_location=model.device_obj)
        if isinstance(payload, dict):
            state_dict = payload.get(eval_cfg.checkpoint.state_dict_key, payload)
        else:
            state_dict = payload
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint payload does not contain a valid state_dict mapping.")
        model.load_state_dict(state_dict, strict=eval_cfg.checkpoint.strict)
        LOGGER.info("Loaded checkpoint for retrieval embeddings: %s", eval_cfg.checkpoint.path)
    else:
        LOGGER.warning("No checkpoint configured; retrieval embeddings will use fresh model weights.")
    model.eval()

    dm = create_eval_datamodule(config_path=data_config_path)
    dm.prepare_data()
    dm.setup(stage=None, split=split)  # type: ignore[arg-type]
    loaders = dm.retrieval_dataloaders(split)  # type: ignore[arg-type]

    gallery = embed_gallery_images(
        model=model,
        loader=loaders["gallery"],
        use_amp=cfg.embed.use_amp,
        normalize=cfg.embed.normalize,
        split=split,
    )
    query = embed_query_texts(
        model=model,
        loader=loaders["query"],
        use_amp=cfg.embed.use_amp,
        normalize=cfg.embed.normalize,
        split=split,
    )

    gallery_meta = {
        "modality": "image",
        "split": split,
        "count": int(gallery.embeddings.shape[0]),
        "dim": int(gallery.embeddings.shape[1]),
    }
    query_meta = {
        "modality": "text",
        "split": split,
        "count": int(query.embeddings.shape[0]),
        "dim": int(query.embeddings.shape[1]),
    }
    image_paths = save_embedding_artifacts(
        cfg=cfg.storage,
        name=image_artifact_name,
        embeddings=gallery.embeddings,
        ids=gallery.ids,
        metadata=gallery_meta,
    )
    text_paths = save_embedding_artifacts(
        cfg=cfg.storage,
        name=text_artifact_name,
        embeddings=query.embeddings,
        ids=query.ids,
        metadata=query_meta,
    )
    LOGGER.info(
        "Saved retrieval embeddings | image=%s (%d) text=%s (%d)",
        image_paths.embeddings,
        gallery.embeddings.shape[0],
        text_paths.embeddings,
        query.embeddings.shape[0],
    )
    return {
        "image_paths": image_paths,
        "text_paths": text_paths,
        "gallery_shape": tuple(gallery.embeddings.shape),
        "query_shape": tuple(query.embeddings.shape),
    }


__all__ = [
    "EmbeddedCorpus",
    "build_and_save_retrieval_embeddings",
    "embed_gallery_images",
    "embed_query_texts",
]
