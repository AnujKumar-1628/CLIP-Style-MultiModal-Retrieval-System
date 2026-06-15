"""Tests for contrastive loss module."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.loss.contrastive_loss import load_contrastive_loss_config, symmetric_info_nce_loss


def test_symmetric_info_nce_loss_returns_scalar_for_mean_reduction() -> None:
    logits = torch.tensor([[4.0, 1.0], [1.0, 4.0]], dtype=torch.float32)
    loss = symmetric_info_nce_loss(
        logits_per_image=logits,
        logits_per_text=logits.t(),
        label_smoothing=0.0,
        reduction="mean",
    )
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_symmetric_info_nce_loss_rejects_non_square_logits() -> None:
    logits_i = torch.zeros((2, 3), dtype=torch.float32)
    logits_t = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="expects square logits"):
        symmetric_info_nce_loss(logits_per_image=logits_i, logits_per_text=logits_t)


def test_load_contrastive_loss_config_rejects_invalid_reduction(write_yaml) -> None:
    cfg_path = write_yaml(
        "model.yaml",
        {
            "loss": {
                "name": "symmetric_info_nce",
                "label_smoothing": 0.1,
                "reduction": "median",
            }
        },
    )
    with pytest.raises(ValueError, match="loss.reduction"):
        load_contrastive_loss_config(cfg_path)
