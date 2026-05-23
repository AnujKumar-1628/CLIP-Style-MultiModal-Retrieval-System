"""Utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """
    Seed Python, NumPy, and PyTorch (CPU/CUDA) for reproducibility.

    Args:
        seed: Global seed value.
        deterministic: If True, enables deterministic torch behavior where possible.

    Returns:
        The seed that was set.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Enforce deterministic algorithms when requested.
    # warn_only keeps training running if an op is non-deterministic on current backend.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    return seed


def seed_worker(worker_id: int) -> None:
    """
    Seed function for PyTorch DataLoader workers.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker, generator=get_torch_generator(seed))
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_torch_generator(seed: Optional[int] = None) -> torch.Generator:
    """Create a torch.Generator seeded for DataLoader reproducibility."""
    generator = torch.Generator()
    if seed is not None:
        if seed < 0:
            raise ValueError("seed must be >= 0")
        generator.manual_seed(seed)
    return generator
