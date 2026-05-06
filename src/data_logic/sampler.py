"""Sampling utilities for CLIP-style training and evaluation.

This module centralizes sampler construction so dataloaders can share
consistent behavior across:
- single-process training/evaluation
- distributed training (DDP)
- optional unique-image batching when multiple captions exist per image
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from src.data_logic.transforms import get_data_config
from src.utils.config import DataConfig
from src.utils.logger import setup_logger


LOGGER = setup_logger(
    name="data_sampler",
    level="INFO",
    use_console=True,
    use_file=True,
)


@dataclass(frozen=True)
class SamplerBundle:
    """Container returned by sampler factory utilities."""

    sampler: Sampler[int] | None
    batch_sampler: BatchSampler | None
    shuffle: bool
    drop_last: bool
    worker_init_fn: Callable[[int], None]


def build_torch_generator(seed: int) -> torch.Generator:
    """Create a deterministic torch generator."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def seed_worker(worker_id: int) -> None:
    """Initialize dataloader worker RNG states deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class UniqueImageBatchSampler(BatchSampler):
    """Build batches with at most one sample per image_name.

    This is useful when a dataset split contains multiple captions per image and
    we want to avoid duplicates in the same mini-batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        records = getattr(dataset, "records", None)
        if records is None:
            raise TypeError(
                "UniqueImageBatchSampler expects dataset.records with image_name fields."
            )

        image_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            image_to_indices[str(record.image_name)].append(idx)

        self._image_keys = list(image_to_indices.keys())
        self._image_to_indices = image_to_indices

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic per-epoch shuffling."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        image_keys = self._image_keys.copy()

        if self.shuffle:
            rng.shuffle(image_keys)

        # One sample per image for this pass. If an image has multiple captions,
        # randomly pick one caption index.
        candidates: list[int] = []
        for image_key in image_keys:
            indices = self._image_to_indices[image_key]
            if len(indices) == 1:
                candidates.append(indices[0])
            else:
                candidates.append(indices[rng.randrange(len(indices))])

        if self.shuffle:
            rng.shuffle(candidates)

        batch: list[int] = []
        for index in candidates:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self._image_keys)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


def _resolve_data_cfg(
    data_cfg: DataConfig | None,
    config_path: str | Path | None,
) -> DataConfig:
    if data_cfg is not None:
        return data_cfg
    return get_data_config(config_path)


def build_train_sampler_bundle(
    dataset: Dataset,
    *,
    data_cfg: DataConfig | None = None,
    config_path: str | Path | None = None,
    distributed: bool | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
    ensure_unique_images_per_batch: bool | None = None,
    seed: int | None = None,
) -> SamplerBundle:
    """Build sampler configuration for train dataloader.

    Returns a bundle that can be plugged into DataLoader construction.
    """
    cfg = _resolve_data_cfg(data_cfg, config_path)
    effective_seed = int(seed if seed is not None else cfg.split.seed)
    effective_distributed = bool(distributed) if distributed is not None else cfg.datamodule.distributed
    if ensure_unique_images_per_batch is None:
        ensure_unique_images_per_batch = cfg.datamodule.use_unique_image_batch_sampler

    if effective_distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            seed=effective_seed,
            drop_last=cfg.loader.drop_last,
        )
        LOGGER.info(
            "Built distributed train sampler | replicas=%s rank=%s seed=%d",
            num_replicas,
            rank,
            effective_seed,
        )
        return SamplerBundle(
            sampler=sampler,
            batch_sampler=None,
            shuffle=False,
            drop_last=cfg.loader.drop_last,
            worker_init_fn=seed_worker,
        )

    if ensure_unique_images_per_batch:
        batch_sampler = UniqueImageBatchSampler(
            dataset=dataset,
            batch_size=cfg.loader.batch_size,
            shuffle=True,
            drop_last=cfg.loader.drop_last,
            seed=effective_seed,
        )
        LOGGER.info(
            "Built unique-image train batch sampler | batch_size=%d seed=%d",
            cfg.loader.batch_size,
            effective_seed,
        )
        return SamplerBundle(
            sampler=None,
            batch_sampler=batch_sampler,
            shuffle=False,
            drop_last=cfg.loader.drop_last,
            worker_init_fn=seed_worker,
        )

    LOGGER.info("Using default random train sampling (DataLoader shuffle=True).")
    return SamplerBundle(
        sampler=None,
        batch_sampler=None,
        shuffle=True,
        drop_last=cfg.loader.drop_last,
        worker_init_fn=seed_worker,
    )


def build_eval_sampler_bundle(
    dataset: Dataset,
    *,
    data_cfg: DataConfig | None = None,
    config_path: str | Path | None = None,
    distributed: bool | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
) -> SamplerBundle:
    """Build sampler configuration for val/test dataloaders."""
    cfg = _resolve_data_cfg(data_cfg, config_path)
    effective_distributed = bool(distributed) if distributed is not None else cfg.datamodule.distributed

    if effective_distributed:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=False,
            seed=cfg.split.seed,
            drop_last=False,
        )
        LOGGER.info(
            "Built distributed eval sampler | replicas=%s rank=%s",
            num_replicas,
            rank,
        )
        return SamplerBundle(
            sampler=sampler,
            batch_sampler=None,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

    LOGGER.info("Using sequential eval sampling (DataLoader shuffle=False).")
    return SamplerBundle(
        sampler=None,
        batch_sampler=None,
        shuffle=False,
        drop_last=False,
        worker_init_fn=seed_worker,
    )


__all__ = [
    "SamplerBundle",
    "UniqueImageBatchSampler",
    "build_eval_sampler_bundle",
    "build_torch_generator",
    "build_train_sampler_bundle",
    "seed_worker",
]
