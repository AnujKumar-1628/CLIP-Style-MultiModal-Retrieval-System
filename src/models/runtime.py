"""Shared runtime helpers for model modules."""

from __future__ import annotations

import logging

import torch


def resolve_device(device_name: str, *, logger: logging.Logger | None = None) -> torch.device:
    """Resolve a torch device string with safe CUDA fallback."""
    requested = device_name.strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if logger is not None:
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_dtype(dtype_name: str) -> torch.dtype:
    """Resolve dtype aliases used across model components."""
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        supported = ", ".join(sorted(mapping))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Supported: {supported}")
    return mapping[key]

