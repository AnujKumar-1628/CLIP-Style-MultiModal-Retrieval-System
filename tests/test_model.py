"""Tests for model runtime helpers."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from src.models.runtime import resolve_device, resolve_dtype


def test_resolve_dtype_supports_aliases() -> None:
    assert resolve_dtype("float32") == torch.float32
    assert resolve_dtype("fp16") == torch.float16
    assert resolve_dtype("bf16") == torch.bfloat16


def test_resolve_dtype_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int8")


def test_resolve_device_falls_back_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    logger = logging.getLogger("test_model_runtime")
    device = resolve_device("cuda:0", logger=logger)
    assert str(device) == "cpu"
