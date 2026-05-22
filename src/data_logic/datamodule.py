"""Data module for CLIP-style image-text training/evaluation.

This module wires together:
- dataset construction (`src.data_logic.dataset`)
- sampler strategies (`src.data_logic.sampler`)
- batch collation/tokenization

The goal is to keep training scripts minimal and config-driven.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data_logic.base_datamodule import BaseDataModule
from src.data_logic.dataset import CLIPCollator, CLIPDataset, build_or_load_split_tables
from src.data_logic.sampler import (
    SamplerBundle,
    build_eval_sampler_bundle,
    build_train_sampler_bundle,
)
from src.utils.config import DataConfig
from src.utils.logger import setup_logger


LOGGER = setup_logger(
    name="data_datamodule",
    level="INFO",
    use_console=True,
    use_file=True,
)


class CLIPDataModule(BaseDataModule):
    """Config-driven data module with train/val/test dataloaders."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        data_cfg: DataConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        distributed: bool | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            data_cfg=data_cfg,
            tokenizer=tokenizer,
        )

        self.distributed = (
            bool(distributed)
            if distributed is not None
            else self.data_cfg.datamodule.distributed
        )
        self.num_replicas = (
            num_replicas
            if num_replicas is not None
            else self.data_cfg.datamodule.distributed_num_replicas
        )
        self.rank = (
            rank
            if rank is not None
            else self.data_cfg.datamodule.distributed_rank
        )

        self.train_dataset: CLIPDataset | None = None
        self.val_dataset: CLIPDataset | None = None
        self.test_dataset: CLIPDataset | None = None
        self._last_train_bundle: SamplerBundle | None = None

        LOGGER.info(
            "Initialized CLIPDataModule | distributed=%s num_replicas=%s rank=%s",
            self.distributed,
            self.num_replicas,
            self.rank,
        )

    def prepare_data(self) -> None:
        """Prepare deterministic split artifacts (and tokenizer, if requested)."""
        build_or_load_split_tables(
            self.data_cfg,
            force_rebuild=self.data_cfg.datamodule.force_rebuild_splits,
        )
        if self._needs_tokenizer():
            self._get_tokenizer()
        LOGGER.info("Data preparation complete.")

    def setup(self, stage: str | None = None) -> None:
        """Create datasets for the requested stage."""
        stage_normalized = None if stage is None else stage.lower()

        tokenizer = self._get_tokenizer()
        force_rebuild = self.data_cfg.datamodule.force_rebuild_splits
        fail_on_missing_image = self.data_cfg.datamodule.fail_on_missing_image
        tokenize_in_dataset = self.data_cfg.datamodule.tokenize_in_dataset

        if stage_normalized in (None, "fit", "train"):
            self.train_dataset = CLIPDataset(
                split="train",
                data_cfg=self.data_cfg,
                tokenizer=tokenizer,
                tokenize_in_dataset=tokenize_in_dataset,
                fail_on_missing_image=fail_on_missing_image,
                force_rebuild_splits=force_rebuild,
                use_train_augmentations=self.data_cfg.datamodule.use_train_augmentations,
            )
            self.val_dataset = CLIPDataset(
                split="val",
                data_cfg=self.data_cfg,
                tokenizer=tokenizer,
                tokenize_in_dataset=tokenize_in_dataset,
                fail_on_missing_image=fail_on_missing_image,
                force_rebuild_splits=False,
                one_caption_per_image=self.data_cfg.datamodule.val_one_caption_per_image,
            )

        if stage_normalized in (None, "fit", "validate", "val") and self.val_dataset is None:
            self.val_dataset = CLIPDataset(
                split="val",
                data_cfg=self.data_cfg,
                tokenizer=tokenizer,
                tokenize_in_dataset=tokenize_in_dataset,
                fail_on_missing_image=fail_on_missing_image,
                force_rebuild_splits=force_rebuild,
                one_caption_per_image=self.data_cfg.datamodule.val_one_caption_per_image,
            )

        if stage_normalized in (None, "test"):
            self.test_dataset = CLIPDataset(
                split="test",
                data_cfg=self.data_cfg,
                tokenizer=tokenizer,
                tokenize_in_dataset=tokenize_in_dataset,
                fail_on_missing_image=fail_on_missing_image,
                force_rebuild_splits=False,
                one_caption_per_image=self.data_cfg.datamodule.test_one_caption_per_image,
            )

        if stage_normalized in ("predict",) and self.test_dataset is None:
            self.test_dataset = CLIPDataset(
                split="test",
                data_cfg=self.data_cfg,
                tokenizer=tokenizer,
                tokenize_in_dataset=tokenize_in_dataset,
                fail_on_missing_image=fail_on_missing_image,
                force_rebuild_splits=False,
                one_caption_per_image=self.data_cfg.datamodule.test_one_caption_per_image,
            )

        collator_tokenizer = tokenizer if self.data_cfg.datamodule.use_tokenizer_in_collate else None
        self.collator = CLIPCollator(
            tokenizer=collator_tokenizer,
            data_cfg=self.data_cfg,
        )

        LOGGER.info("DataModule setup complete for stage=%s", stage_normalized)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup('fit') first.")

        bundle = build_train_sampler_bundle(
            self.train_dataset,
            data_cfg=self.data_cfg,
            distributed=self.distributed,
            num_replicas=self.num_replicas,
            rank=self.rank,
            ensure_unique_images_per_batch=self.data_cfg.datamodule.use_unique_image_batch_sampler,
            seed=self.data_cfg.split.seed,
        )
        self._last_train_bundle = bundle
        return self._build_loader(dataset=self.train_dataset, sampler_bundle=bundle)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup('fit') or setup('validate').")

        bundle = build_eval_sampler_bundle(
            self.val_dataset,
            data_cfg=self.data_cfg,
            distributed=self.distributed,
            num_replicas=self.num_replicas,
            rank=self.rank,
        )
        return self._build_loader(dataset=self.val_dataset, sampler_bundle=bundle)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Call setup('test') first.")

        bundle = build_eval_sampler_bundle(
            self.test_dataset,
            data_cfg=self.data_cfg,
            distributed=self.distributed,
            num_replicas=self.num_replicas,
            rank=self.rank,
        )
        return self._build_loader(dataset=self.test_dataset, sampler_bundle=bundle)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to epoch-aware train samplers."""
        if self.train_dataset is not None and hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)
        if self._last_train_bundle is None:
            return
        if self._last_train_bundle.sampler is not None and hasattr(
            self._last_train_bundle.sampler, "set_epoch"
        ):
            self._last_train_bundle.sampler.set_epoch(epoch)
        if self._last_train_bundle.batch_sampler is not None and hasattr(
            self._last_train_bundle.batch_sampler, "set_epoch"
        ):
            self._last_train_bundle.batch_sampler.set_epoch(epoch)


def create_datamodule(
    *,
    config_path: str | Path | None = None,
    tokenizer: PreTrainedTokenizerBase | str | None = None,
    distributed: bool | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
) -> CLIPDataModule:
    """Convenience factory for datamodule construction."""
    return CLIPDataModule(
        config_path=config_path,
        tokenizer=tokenizer,
        distributed=distributed,
        num_replicas=num_replicas,
        rank=rank,
    )


__all__ = [
    "CLIPDataModule",
    "create_datamodule",
]
