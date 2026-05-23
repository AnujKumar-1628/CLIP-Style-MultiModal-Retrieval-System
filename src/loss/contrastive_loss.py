"""Contrastive losses for CLIP-style image-text retrieval.

This module provides:
1) Config loading for the `loss` section in `configs/model.yaml`
2) A registry for loss modules
3) Symmetric InfoNCE loss over image->text and text->image logits
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import (
    ContrastiveLossConfig,
    load_contrastive_loss_config as load_contrastive_loss_config_from_utils,
)
from src.utils.logger import setup_logger
from src.utils.registry import Registry

LOGGER = setup_logger(
    name="contrastive_loss",
    level="INFO",
    use_console=True,
    use_file=True,
)

LOSS_REGISTRY: Registry[type[nn.Module]] = Registry("loss")

ReductionType = Literal["mean", "sum", "none"]


def load_contrastive_loss_config(
    config_path: str | Path | None = None,
) -> ContrastiveLossConfig:
    """Load contrastive loss config from model config."""
    return load_contrastive_loss_config_from_utils(config_path)


def _validate_logits(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> None:
    if logits_per_image.ndim != 2:
        raise ValueError(
            "Expected logits_per_image shape [B, B], got "
            f"{tuple(logits_per_image.shape)}."
        )
    if logits_per_text.ndim != 2:
        raise ValueError(
            "Expected logits_per_text shape [B, B], got "
            f"{tuple(logits_per_text.shape)}."
        )
    if logits_per_image.shape != logits_per_text.shape:
        raise ValueError(
            "logits_per_image and logits_per_text must have the same shape, got "
            f"{tuple(logits_per_image.shape)} vs {tuple(logits_per_text.shape)}."
        )
    rows, cols = logits_per_image.shape
    if rows != cols:
        raise ValueError(
            "Contrastive InfoNCE expects square logits [B, B], got "
            f"{tuple(logits_per_image.shape)}."
        )


def _build_targets(batch_size: int, device: torch.device) -> torch.Tensor:
    # Targets align matching image/text pairs along the diagonal.
    return torch.arange(batch_size, device=device, dtype=torch.long)


def symmetric_info_nce_loss(
    *,
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
    label_smoothing: float = 0.0,
    reduction: ReductionType = "mean",
) -> torch.Tensor:
    """Compute symmetric CLIP InfoNCE loss.

    The final loss is the average of:
    - image->text classification loss
    - text->image classification loss
    """
    _validate_logits(logits_per_image, logits_per_text)

    batch_size = logits_per_image.shape[0]
    targets = _build_targets(batch_size, logits_per_image.device)

    loss_i2t = F.cross_entropy(
        logits_per_image,
        targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    loss_t2i = F.cross_entropy(
        logits_per_text,
        targets,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )
    return 0.5 * (loss_i2t + loss_t2i)


@LOSS_REGISTRY.register("symmetric_info_nce")
@LOSS_REGISTRY.register("clip_contrastive")
class SymmetricInfoNCELoss(nn.Module):
    """Symmetric InfoNCE module used for CLIP-style training."""

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        reduction: ReductionType = "mean",
    ) -> None:
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must satisfy 0.0 <= value < 1.0.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of: mean, sum, none.")
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(
        self,
        *,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        return symmetric_info_nce_loss(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def build_contrastive_loss(
    config: ContrastiveLossConfig,
) -> nn.Module:
    """Instantiate loss module from typed config."""
    loss_cls = LOSS_REGISTRY.get(config.name)
    return loss_cls(  # type: ignore[call-arg]
        label_smoothing=config.label_smoothing,
        reduction=config.reduction,
    )


def create_contrastive_loss(
    *,
    config_path: str | Path | None = None,
) -> nn.Module:
    """Create contrastive loss from model config path."""
    cfg = load_contrastive_loss_config(config_path)
    loss = build_contrastive_loss(cfg)
    LOGGER.info(
        "Initialized contrastive loss | name=%s label_smoothing=%.4f reduction=%s",
        cfg.name,
        cfg.label_smoothing,
        cfg.reduction,
    )
    return loss


__all__ = [
    "ContrastiveLossConfig",
    "LOSS_REGISTRY",
    "SymmetricInfoNCELoss",
    "build_contrastive_loss",
    "create_contrastive_loss",
    "load_contrastive_loss_config",
    "symmetric_info_nce_loss",
]
