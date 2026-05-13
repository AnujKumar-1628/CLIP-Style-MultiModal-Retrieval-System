"""Two-phase backbone unfreeze callback for CLIP-style training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.training.callbacks import Callback
from src.utils.config import (
    load_resnet50_encoder_config,
    load_distilbert_encoder_config,
)
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="backbone_unfreeze_callback",
    level="INFO",
    use_console=True,
    use_file=True,
)


class BackboneUnfreezeCallback(Callback):
    """Unfreeze selected backbone layers at a target epoch and lower LR."""

    def __init__(
        self,
        *,
        unfreeze_at_epoch: int,
        backbone_lr: float,
        image_unfreeze_layers: tuple[str, ...],
        text_unfreeze_layers: tuple[str, ...],
    ) -> None:
        self.unfreeze_at_epoch = int(unfreeze_at_epoch)
        self.backbone_lr = float(backbone_lr)
        self.image_unfreeze_layers = image_unfreeze_layers
        self.text_unfreeze_layers = text_unfreeze_layers
        self._unfrozen = False

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        if epoch == self.unfreeze_at_epoch and not self._unfrozen:
            self._unfreeze_backbones(trainer=trainer, epoch=epoch)

    def _unfreeze_backbones(self, *, trainer: Any, epoch: int) -> None:
        LOGGER.info("Epoch %d: activating phase-2 backbone fine-tuning.", epoch)
        model = self._get_model(trainer)
        if model is None:
            LOGGER.warning("Model not found on trainer; skipping backbone unfreeze.")
            return

        unfrozen_param_count = 0

        image_encoder = getattr(model, "image_encoder", None)
        if image_encoder is None:
            LOGGER.warning("image_encoder not found on model.")
        else:
            for name, param in image_encoder.named_parameters():
                if any(layer in name for layer in self.image_unfreeze_layers):
                    if not param.requires_grad:
                        param.requires_grad_(True)
                        unfrozen_param_count += param.numel()
            LOGGER.info("Image encoder layers unfrozen: %s", self.image_unfreeze_layers)

        text_encoder = getattr(model, "text_encoder", None)
        if text_encoder is None:
            LOGGER.warning("text_encoder not found on model.")
        else:
            for name, param in text_encoder.named_parameters():
                if any(token in name for token in self.text_unfreeze_layers):
                    if not param.requires_grad:
                        param.requires_grad_(True)
                        unfrozen_param_count += param.numel()
            LOGGER.info(
                "Text encoder layers unfrozen: %s",
                self.text_unfreeze_layers,
            )

        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is None:
            LOGGER.warning("optimizer not found on trainer; learning rate not updated.")
        else:
            old_lrs = [group.get("lr", None) for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = self.backbone_lr
            LOGGER.info(
                "Learning rates updated for all parameter groups: %s -> %.2e",
                old_lrs,
                self.backbone_lr,
            )

        self._unfrozen = True
        LOGGER.info(
            "Epoch %d: phase-2 is active; newly trainable parameters=%s",
            epoch,
            f"{unfrozen_param_count:,}",
        )

    @staticmethod
    def _get_model(trainer: Any) -> Any | None:
        for attr in ("_unwrapped_model", "model", "_model"):
            candidate = getattr(trainer, attr, None)
            if candidate is not None:
                return candidate
        return None

    def __repr__(self) -> str:
        return (
            "BackboneUnfreezeCallback("
            f"unfreeze_at_epoch={self.unfreeze_at_epoch}, "
            f"backbone_lr={self.backbone_lr:.2e})"
        )


def build_backbone_unfreeze_callback(
    *,
    unfreeze_at_epoch: int = 4,
    backbone_lr: float = 1e-5,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> BackboneUnfreezeCallback:
    """Factory helper to keep callback wiring in train.py concise."""
    resnet_cfg = load_resnet50_encoder_config(model_config_path)
    distilbert_cfg = load_distilbert_encoder_config(model_config_path, data_config_path)

    return BackboneUnfreezeCallback(
        unfreeze_at_epoch=unfreeze_at_epoch,
        backbone_lr=backbone_lr,
        image_unfreeze_layers=resnet_cfg.unfreeze_layers,
        text_unfreeze_layers=distilbert_cfg.unfreeze_layers,
    )


__all__ = [
    "BackboneUnfreezeCallback",
    "build_backbone_unfreeze_callback",
]
