"""Projection heads for CLIP-style multimodal retrieval.

This module builds config-driven projection heads that map image/text encoder
features into a shared embedding space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.runtime import resolve_device, resolve_dtype
from src.utils.config import (
    ProjectionConfigBundle,
    ProjectionHeadConfig,
    load_projection_config_bundle as load_projection_config_bundle_from_utils,
)
from src.utils.logger import setup_logger
from src.utils.registry import Registry

LOGGER = setup_logger(
    name="projection_heads",
    level="INFO",
    use_console=True,
    use_file=True,
)

PROJECTION_HEAD_REGISTRY: Registry[type[nn.Module]] = Registry("projection_head")


def _build_activation(name: str) -> nn.Module:
    key = name.strip().lower()
    if key == "relu":
        return nn.ReLU(inplace=True)
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU(inplace=True)
    if key == "tanh":
        return nn.Tanh()
    raise ValueError(
        "Unsupported activation '{}'. Use one of: relu, gelu, silu, tanh.".format(name)
    )


def load_projection_config_bundle(
    config_path: str | Path | None = None,
) -> ProjectionConfigBundle:
    """Load projection-head configuration from `configs/model.yaml`."""
    return load_projection_config_bundle_from_utils(config_path)


@PROJECTION_HEAD_REGISTRY.register("mlp_projection")
class MLPProjectionHead(nn.Module):
    """Two-layer projection head with optional LayerNorm and output normalization."""

    def __init__(self, config: ProjectionHeadConfig) -> None:
        super().__init__()
        if config.input_dim <= 0 or config.hidden_dim <= 0 or config.output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be > 0.")
        if config.dropout < 0.0:
            raise ValueError("dropout must be >= 0.")

        self.config = config
        self.device_obj = resolve_device(config.device, logger=LOGGER)
        self.dtype_obj = resolve_dtype(config.dtype)

        layers: list[nn.Module] = [
            nn.Linear(config.input_dim, config.hidden_dim),
            _build_activation(config.activation),
        ]
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(config.output_dim))

        self.proj = nn.Sequential(*layers)
        self.normalize_output = bool(config.normalize_output)
        self.to(device=self.device_obj, dtype=self.dtype_obj)

        LOGGER.info(
            "Initialized MLPProjectionHead | in=%d hidden=%d out=%d act=%s dropout=%.3f "
            "layer_norm=%s normalize_output=%s device=%s dtype=%s",
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.activation,
            config.dropout,
            config.use_layer_norm,
            config.normalize_output,
            self.device_obj,
            self.dtype_obj,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"Expected feature tensor shape [B, D], got {tuple(features.shape)}"
            )
        out = self.proj(features)
        if self.normalize_output:
            out = F.normalize(out, p=2, dim=-1)
        return out

    @torch.inference_mode()
    def project(
        self,
        features: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        target_device = torch.device(device) if device is not None else self.device_obj
        x = features.to(target_device, dtype=self.dtype_obj, non_blocking=non_blocking)
        was_training = self.training
        self.eval()
        out = self.forward(x)
        if was_training:
            self.train()
        return out


class DualProjectionHeads(nn.Module):
    """Container for image/text projection heads sharing a common embed dimension."""

    def __init__(
        self, *, image_head: MLPProjectionHead, text_head: MLPProjectionHead
    ) -> None:
        super().__init__()
        self.image = image_head
        self.text = text_head
        self.embed_dim = image_head.config.output_dim
        if self.embed_dim != text_head.config.output_dim:
            raise ValueError(
                "Image and text projection heads must have the same output_dim."
            )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(image_features), self.text(text_features)


def build_projection_head(
    config: ProjectionHeadConfig,
    *,
    head_type: Literal["mlp_projection"] = "mlp_projection",
) -> MLPProjectionHead:
    head_cls = PROJECTION_HEAD_REGISTRY.get(head_type)
    head = head_cls(config)  # type: ignore[call-arg]
    return head


def build_dual_projection_heads(
    config_path: str | Path | None = None,
) -> DualProjectionHeads:
    """Build image and text projection heads from model config."""
    bundle = load_projection_config_bundle(config_path=config_path)
    image_head = build_projection_head(bundle.image)
    text_head = build_projection_head(bundle.text)
    return DualProjectionHeads(image_head=image_head, text_head=text_head)


__all__ = [
    "DualProjectionHeads",
    "MLPProjectionHead",
    "PROJECTION_HEAD_REGISTRY",
    "ProjectionConfigBundle",
    "ProjectionHeadConfig",
    "build_dual_projection_heads",
    "build_projection_head",
    "load_projection_config_bundle",
]
