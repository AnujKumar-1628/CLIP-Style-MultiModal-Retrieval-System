"""CLIP-style multimodal model wrapper.

This module composes:
- image encoder (ResNet-50)
- text encoder (DistilBERT)
- dual projection heads

and exposes a clean training/inference API for retrieval tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.models.encoders.image.resnet50 import ResNet50ImageEncoder
from src.models.projection import (
    DualProjectionHeads,
    build_dual_projection_heads,
)
from src.models.encoders.text.distilbert import DistilBertTextEncoder
from src.models.runtime import resolve_device, resolve_dtype
from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR
from src.utils.registry import Registry

LOGGER = setup_logger(
    name="clip_model",
    level="INFO",
    use_console=True,
    use_file=True,
)

CLIP_MODEL_REGISTRY: Registry[type[nn.Module]] = Registry("clip_model")


@dataclass(frozen=True)
class CLIPModelConfig:
    model_name: str
    device: str
    dtype: str
    normalize_embeddings: bool
    logit_scale_init_value: float
    logit_scale_learnable: bool
    logit_scale_clamp_min: float
    logit_scale_clamp_max: float


def load_clip_model_config(config_path: str | Path | None = None) -> CLIPModelConfig:
    """Load model-level CLIP settings from `configs/model.yaml`."""
    model_cfg_path = (
        Path(config_path) if config_path is not None else (CONFIGS_DIR / "model.yaml")
    )
    raw = load_yaml(model_cfg_path)

    model_raw = raw.get("model", {})
    similarity_raw = raw.get("similarity", {})
    logit_scale_raw = similarity_raw.get("logit_scale", {})

    init_value = float(logit_scale_raw.get("init_value", 0.07))
    if init_value <= 0:
        raise ValueError("similarity.logit_scale.init_value must be > 0.")

    cfg = CLIPModelConfig(
        model_name=str(model_raw.get("name", "clip_style_retrieval")),
        device=str(model_raw.get("device", "cpu")),
        dtype=str(model_raw.get("dtype", "float32")),
        normalize_embeddings=bool(similarity_raw.get("normalize_embeddings", True)),
        logit_scale_init_value=init_value,
        logit_scale_learnable=bool(logit_scale_raw.get("learnable", True)),
        logit_scale_clamp_min=float(logit_scale_raw.get("clamp_min", 0.001)),
        logit_scale_clamp_max=float(logit_scale_raw.get("clamp_max", 100.0)),
    )
    if cfg.logit_scale_clamp_min <= 0 or cfg.logit_scale_clamp_max <= 0:
        raise ValueError("logit scale clamp values must be > 0.")
    if cfg.logit_scale_clamp_min >= cfg.logit_scale_clamp_max:
        raise ValueError("logit_scale.clamp_min must be < logit_scale.clamp_max.")
    return cfg


@CLIP_MODEL_REGISTRY.register("clip_style_retrieval")
class CLIPModel(nn.Module):
    """Unified CLIP-style model for training and retrieval inference."""

    def __init__(
        self,
        *,
        config: CLIPModelConfig | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
        image_encoder: ResNet50ImageEncoder | None = None,
        text_encoder: DistilBertTextEncoder | None = None,
        projection_heads: DualProjectionHeads | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_clip_model_config(model_config_path)
        self.device_obj = resolve_device(self.config.device, logger=LOGGER)
        self.dtype_obj = resolve_dtype(self.config.dtype)

        self.image_encoder = image_encoder or ResNet50ImageEncoder.from_config_paths(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )
        self.text_encoder = text_encoder or DistilBertTextEncoder.from_config_paths(
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )
        self.projection_heads = projection_heads or build_dual_projection_heads(
            model_config_path
        )

        self.embed_dim = self.projection_heads.embed_dim
        self._validate_dimensions()

        # In CLIP, learned parameter is usually log(scale), where scale = 1 / temperature.
        initial_scale = 1.0 / self.config.logit_scale_init_value
        self.logit_scale_log = nn.Parameter(
            torch.tensor(math.log(initial_scale), dtype=torch.float32),
            requires_grad=self.config.logit_scale_learnable,
        )

        self.to(device=self.device_obj, dtype=self.dtype_obj)
        LOGGER.info(
            "Initialized CLIPModel | name=%s device=%s dtype=%s embed_dim=%d "
            "normalize_embeddings=%s logit_scale_learnable=%s",
            self.config.model_name,
            self.device_obj,
            self.dtype_obj,
            self.embed_dim,
            self.config.normalize_embeddings,
            self.config.logit_scale_learnable,
        )

    def _validate_dimensions(self) -> None:
        if (
            self.image_encoder.output_dim
            != self.projection_heads.image.config.input_dim
        ):
            raise ValueError(
                "Image encoder output dim ({}) does not match image projection input dim ({}).".format(
                    self.image_encoder.output_dim,
                    self.projection_heads.image.config.input_dim,
                )
            )
        if self.text_encoder.output_dim != self.projection_heads.text.config.input_dim:
            raise ValueError(
                "Text encoder output dim ({}) does not match text projection input dim ({}).".format(
                    self.text_encoder.output_dim,
                    self.projection_heads.text.config.input_dim,
                )
            )

    def get_logit_scale(self) -> torch.Tensor:
        scale = self.logit_scale_log.exp()
        return torch.clamp(
            scale,
            min=self.config.logit_scale_clamp_min,
            max=self.config.logit_scale_clamp_max,
        )

    def encode_image_features(
        self,
        images: torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        image_batch = images.to(
            self.device_obj, dtype=self.dtype_obj, non_blocking=True
        )
        features = self.image_encoder(image_batch)
        projected = self.projection_heads.image(features)
        if normalize is True:
            return F.normalize(projected, p=2, dim=-1)
        if normalize is False:
            return projected
        if self.config.normalize_embeddings:
            return F.normalize(projected, p=2, dim=-1)
        return projected

    def encode_text_features(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        ids = input_ids.to(self.device_obj, non_blocking=True)
        mask = (
            attention_mask.to(self.device_obj, non_blocking=True)
            if attention_mask is not None
            else None
        )
        features = self.text_encoder(input_ids=ids, attention_mask=mask)
        projected = self.projection_heads.text(features)
        if normalize is True:
            return F.normalize(projected, p=2, dim=-1)
        if normalize is False:
            return projected
        if self.config.normalize_embeddings:
            return F.normalize(projected, p=2, dim=-1)
        return projected

    def compute_similarity_logits(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.get_logit_scale()
        logits_per_image = scale * (image_embeddings @ text_embeddings.t())
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def forward(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        image_embeddings = self.encode_image_features(images)
        text_embeddings = self.encode_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_per_image, logits_per_text = self.compute_similarity_logits(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": self.get_logit_scale(),
        }

    @torch.inference_mode()
    def encode_images(
        self,
        images: torch.Tensor,
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        embeddings = self.encode_image_features(images, normalize=normalize)
        if was_training:
            self.train()
        return embeddings

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        tokens = self.text_encoder.tokenize_texts(texts)
        return self.encode_text_tokens(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
            normalize=normalize,
        )

    @torch.inference_mode()
    def encode_text_tokens(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        embeddings = self.encode_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=normalize,
        )
        if was_training:
            self.train()
        return embeddings

    @torch.inference_mode()
    def similarity_from_tensors(
        self,
        *,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_embeddings = self.encode_images(images, normalize=True)
        text_embeddings = self.encode_text_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            normalize=True,
        )
        logits_per_image, _ = self.compute_similarity_logits(
            image_embeddings, text_embeddings
        )
        return logits_per_image

    @torch.inference_mode()
    def similarity_from_raw(
        self,
        *,
        images: Iterable[Image.Image],
        texts: Iterable[str],
    ) -> torch.Tensor:
        image_batch = self.image_encoder.preprocess_images(images)
        return self.similarity_from_texts_and_images_tensor(
            image_batch=image_batch, texts=texts
        )

    @torch.inference_mode()
    def similarity_from_texts_and_images_tensor(
        self,
        *,
        image_batch: torch.Tensor,
        texts: Iterable[str],
    ) -> torch.Tensor:
        image_embeddings = self.encode_images(image_batch, normalize=True)
        text_embeddings = self.encode_texts(texts, normalize=True)
        logits_per_image, _ = self.compute_similarity_logits(
            image_embeddings, text_embeddings
        )
        return logits_per_image

    @classmethod
    def from_config_paths(
        cls,
        *,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> "CLIPModel":
        cfg = load_clip_model_config(model_config_path)
        return cls(
            config=cfg,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        )


def create_clip_model(
    *,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> CLIPModel:
    return CLIPModel.from_config_paths(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )


__all__ = [
    "CLIPModel",
    "CLIPModelConfig",
    "CLIP_MODEL_REGISTRY",
    "create_clip_model",
    "load_clip_model_config",
]
