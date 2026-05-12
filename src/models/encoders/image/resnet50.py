"""ResNet-50 image encoder for CLIP-style retrieval.

Design goals:
- Config-driven construction from `configs/model.yaml`
- Efficient inference helpers with low CPU overhead
- Seamless reuse of existing data transforms/config utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from src.data_logic.transforms import build_eval_image_transform, get_data_config
from src.models.runtime import resolve_device, resolve_dtype
from src.utils.config import (
    ResNet50EncoderConfig,
    load_resnet50_encoder_config as load_resnet50_encoder_config_from_utils,
)
from src.utils.logger import setup_logger
from src.utils.registry import Registry


LOGGER = setup_logger(
    name="resnet50_encoder",
    level="INFO",
    use_console=True,
    use_file=True,
)

IMAGE_ENCODER_REGISTRY: Registry[type[nn.Module]] = Registry("image_encoder")


def _resolve_resnet_weights(pretrained: bool, weights_name: str | None) -> ResNet50_Weights | None:
    if not pretrained:
        return None
    if weights_name is None:
        return ResNet50_Weights.DEFAULT
    try:
        return ResNet50_Weights[weights_name]
    except KeyError as exc:
        valid = ", ".join(item.name for item in ResNet50_Weights)
        raise ValueError(
            f"Invalid ResNet50 weights '{weights_name}'. Valid options: {valid}"
        ) from exc


def load_resnet50_encoder_config(config_path: str | Path | None = None) -> ResNet50EncoderConfig:
    """Load encoder settings from model config."""
    return load_resnet50_encoder_config_from_utils(config_path)


@IMAGE_ENCODER_REGISTRY.register("resnet50")
class ResNet50ImageEncoder(nn.Module):
    """Config-driven ResNet-50 encoder with efficient inference helpers."""

    def __init__(
        self,
        config: ResNet50EncoderConfig | None = None,
        *,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config or load_resnet50_encoder_config(model_config_path)

        weights = _resolve_resnet_weights(self.config.pretrained, self.config.weights)
        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            if weights is None:
                raise
            if not self.config.allow_random_init_fallback:
                raise RuntimeError(
                    "Failed to load pretrained ResNet-50 weights '{}': {}. "
                    "To allow random-initialized fallback, set "
                    "image_encoder.allow_random_init_fallback=true in model config."
                    .format(self.config.weights, exc)
                ) from exc
            LOGGER.warning(
                "Failed to load pretrained weights '%s' (%s). Falling back to random init "
                "(allow_random_init_fallback=true).",
                self.config.weights,
                exc,
            )
            backbone = resnet50(weights=None)
        in_dim = int(backbone.fc.in_features)
        if self.config.out_dim != in_dim:
            raise ValueError(
                f"image_encoder.out_dim must be {in_dim} for ResNet-50, got {self.config.out_dim}."
            )
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
        self.output_dim = in_dim

        self.device_obj = resolve_device(self.config.device, logger=LOGGER)
        self.dtype_obj = resolve_dtype(self.config.dtype)
        self.use_channels_last = bool(self.config.channels_last)

        self._apply_trainability_controls()
        self.to(device=self.device_obj, dtype=self.dtype_obj)
        if self.use_channels_last:
            self.backbone = self.backbone.to(memory_format=torch.channels_last)

        # Reuse the same deterministic eval transform that data pipeline uses.
        self.eval_image_transform = build_eval_image_transform(get_data_config(data_config_path))

        LOGGER.info(
            "Initialized ResNet50ImageEncoder | device=%s dtype=%s pretrained=%s weights=%s "
            "trainable=%s freeze_backbone=%s freeze_until=%s normalize_output=%s channels_last=%s "
            "allow_random_init_fallback=%s",
            self.device_obj,
            self.dtype_obj,
            self.config.pretrained,
            self.config.weights,
            self.config.trainable,
            self.config.freeze_backbone,
            self.config.freeze_until,
            self.config.normalize_output,
            self.use_channels_last,
            self.config.allow_random_init_fallback,
        )

    def _apply_trainability_controls(self) -> None:
        if not self.config.trainable or self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            return

        if self.config.freeze_until is None:
            return

        reached_marker = False
        for name, module in self.backbone.named_children():
            if not reached_marker:
                for param in module.parameters():
                    param.requires_grad = False
            if name == self.config.freeze_until:
                reached_marker = True

        if not reached_marker:
            valid = ", ".join(name for name, _ in self.backbone.named_children())
            raise ValueError(
                f"freeze_until='{self.config.freeze_until}' not found. Valid layer names: {valid}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        if x.ndim != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(x.shape)}")

        if self.use_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.dropout(features)

        if self.config.normalize_output:
            features = F.normalize(features, p=2, dim=-1)
        return features

    @torch.inference_mode()
    def encode(
        self,
        images: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        """Efficient inference wrapper with minimal CPU-side overhead."""
        target_device = torch.device(device) if device is not None else self.device_obj
        batch = images.to(target_device, dtype=self.dtype_obj, non_blocking=non_blocking)
        was_training = self.training
        self.eval()
        embeddings = self.forward(batch)
        if was_training:
            self.train()
        return embeddings

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply project-consistent eval preprocessing to a single PIL image."""
        return self.eval_image_transform(image)

    def preprocess_images(self, images: Iterable[Image.Image]) -> torch.Tensor:
        """Apply preprocessing and stack PIL images into [B, C, H, W]."""
        tensors = [self.preprocess_image(img) for img in images]
        if not tensors:
            raise ValueError("Expected at least one image in preprocess_images.")
        return torch.stack(tensors, dim=0)

    @classmethod
    def from_config_paths(
        cls,
        *,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> "ResNet50ImageEncoder":
        """Convenience constructor when caller wants explicit config paths."""
        cfg = load_resnet50_encoder_config(model_config_path)
        return cls(config=cfg, model_config_path=model_config_path, data_config_path=data_config_path)


__all__ = [
    "IMAGE_ENCODER_REGISTRY",
    "ResNet50EncoderConfig",
    "ResNet50ImageEncoder",
    "load_resnet50_encoder_config",
]
