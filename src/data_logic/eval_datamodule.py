"""Evaluation datamodule for CLIP-style retrieval workflows.

This module provides clean, config-driven dataloaders for:
- query side (typically all captions)
- gallery side (typically one sample per image)

It reuses dataset/sampler/collator utilities from the training stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.data_logic.base_datamodule import BaseDataModule
from src.data_logic.dataset import CLIPCollator, CLIPDataset, build_or_load_split_tables
from src.data_logic.sampler import SamplerBundle, build_eval_sampler_bundle
from src.utils.config import DataConfig
from src.utils.logger import setup_logger


LOGGER = setup_logger(
    name="data_eval_datamodule",
    level="INFO",
    use_console=True,
    use_file=True,
)

EvalSplitName = Literal["val", "test"]


class CLIPEvalDataModule(BaseDataModule):
    """Datamodule specialized for retrieval evaluation."""

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

        self.query_datasets: dict[EvalSplitName, CLIPDataset] = {}
        self.gallery_datasets: dict[EvalSplitName, CLIPDataset] = {}
        self._last_sampler_bundles: dict[tuple[EvalSplitName, str], SamplerBundle] = {}

        LOGGER.info(
            "Initialized CLIPEvalDataModule | distributed=%s num_replicas=%s rank=%s",
            self.distributed,
            self.num_replicas,
            self.rank,
        )

    def _resolve_eval_splits(
        self,
        *,
        stage: str | None = None,
        split: EvalSplitName | None = None,
    ) -> list[EvalSplitName]:
        if split is not None:
            return [split]

        if stage is None:
            default_split = self.data_cfg.datamodule.eval_default_split
            if default_split == "val":
                return ["val"]
            return ["test"]

        stage_normalized = stage.lower().strip()
        if stage_normalized in {"validate", "val"}:
            return ["val"]
        if stage_normalized in {"test", "predict"}:
            return ["test"]

        if stage_normalized in {"fit", "train"}:
            return ["val"]

        raise ValueError(
            f"Unsupported eval stage '{stage}'. Expected one of: "
            "'val', 'validate', 'test', 'predict', 'fit', 'train'."
        )

    def _build_dataset(self, *, split: EvalSplitName, one_caption_per_image: bool) -> CLIPDataset:
        tokenizer = self._get_tokenizer()
        return CLIPDataset(
            split=split,
            data_cfg=self.data_cfg,
            tokenizer=tokenizer,
            tokenize_in_dataset=self.data_cfg.datamodule.tokenize_in_dataset,
            fail_on_missing_image=self.data_cfg.datamodule.fail_on_missing_image,
            force_rebuild_splits=False,
            one_caption_per_image=one_caption_per_image,
        )

    def prepare_data(self) -> None:
        """Prepare split artifacts and tokenizer cache when requested."""
        build_or_load_split_tables(
            self.data_cfg,
            force_rebuild=self.data_cfg.datamodule.force_rebuild_splits,
        )
        if self._needs_tokenizer():
            self._get_tokenizer()
        LOGGER.info("Evaluation data preparation complete.")

    def setup(self, stage: str | None = None, *, split: EvalSplitName | None = None) -> None:
        """Instantiate query/gallery datasets for target eval splits."""
        target_splits = self._resolve_eval_splits(stage=stage, split=split)

        for target_split in target_splits:
            self.query_datasets[target_split] = self._build_dataset(
                split=target_split,
                one_caption_per_image=self.data_cfg.datamodule.eval_query_one_caption_per_image,
            )
            self.gallery_datasets[target_split] = self._build_dataset(
                split=target_split,
                one_caption_per_image=self.data_cfg.datamodule.eval_gallery_one_caption_per_image,
            )

        collator_tokenizer = (
            self._get_tokenizer() if self.data_cfg.datamodule.use_tokenizer_in_collate else None
        )
        self.collator = CLIPCollator(
            tokenizer=collator_tokenizer,
            data_cfg=self.data_cfg,
        )

        LOGGER.info(
            "EvalDataModule setup complete | stage=%s | splits=%s",
            stage,
            target_splits,
        )

    def _resolve_ready_split(self, split: EvalSplitName | None) -> EvalSplitName:
        if split is not None:
            return split

        default_split = self.data_cfg.datamodule.eval_default_split
        if default_split not in {"val", "test"}:
            raise RuntimeError(
                "Invalid configured eval split. Expected 'val' or 'test'."
            )
        if default_split in self.query_datasets:
            return default_split

        if "test" in self.query_datasets:
            return "test"
        if "val" in self.query_datasets:
            return "val"

        raise RuntimeError(
            "No eval datasets are initialized. Call setup('test') or setup('validate') first."
        )

    def _build_eval_loader(
        self,
        *,
        dataset: CLIPDataset,
        split: EvalSplitName,
        loader_kind: Literal["query", "gallery"],
    ) -> DataLoader:
        sampler_bundle = build_eval_sampler_bundle(
            dataset,
            data_cfg=self.data_cfg,
            distributed=self.distributed,
            num_replicas=self.num_replicas,
            rank=self.rank,
        )
        self._last_sampler_bundles[(split, loader_kind)] = sampler_bundle
        return super()._build_loader(dataset=dataset, sampler_bundle=sampler_bundle)

    def query_dataloader(self, split: EvalSplitName | None = None) -> DataLoader:
        """Return caption-query dataloader for the requested split."""
        active_split = self._resolve_ready_split(split)
        if active_split not in self.query_datasets:
            raise RuntimeError(
                f"Query dataset for split '{active_split}' is not initialized. Call setup first."
            )
        return self._build_eval_loader(
            dataset=self.query_datasets[active_split],
            split=active_split,
            loader_kind="query",
        )

    def gallery_dataloader(self, split: EvalSplitName | None = None) -> DataLoader:
        """Return image-gallery dataloader for the requested split."""
        active_split = self._resolve_ready_split(split)
        if active_split not in self.gallery_datasets:
            raise RuntimeError(
                f"Gallery dataset for split '{active_split}' is not initialized. Call setup first."
            )
        return self._build_eval_loader(
            dataset=self.gallery_datasets[active_split],
            split=active_split,
            loader_kind="gallery",
        )

    def retrieval_dataloaders(self, split: EvalSplitName | None = None) -> dict[str, DataLoader]:
        """Return both query and gallery dataloaders for retrieval evaluation."""
        return {
            "query": self.query_dataloader(split),
            "gallery": self.gallery_dataloader(split),
        }

    def val_dataloader(self) -> DataLoader:
        """Framework-friendly alias: returns query dataloader for val split."""
        return self.query_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Framework-friendly alias: returns query dataloader for test split."""
        return self.query_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        """Framework-friendly alias: returns query dataloader for default eval split."""
        return self.query_dataloader(None)

    def set_epoch(self, epoch: int) -> None:
        """Propagate epoch to distributed samplers when available."""
        for sampler_bundle in self._last_sampler_bundles.values():
            if sampler_bundle.sampler is not None and hasattr(sampler_bundle.sampler, "set_epoch"):
                sampler_bundle.sampler.set_epoch(epoch)
            if sampler_bundle.batch_sampler is not None and hasattr(
                sampler_bundle.batch_sampler, "set_epoch"
            ):
                sampler_bundle.batch_sampler.set_epoch(epoch)


def create_eval_datamodule(
    *,
    config_path: str | Path | None = None,
    tokenizer: PreTrainedTokenizerBase | str | None = None,
    distributed: bool | None = None,
    num_replicas: int | None = None,
    rank: int | None = None,
) -> CLIPEvalDataModule:
    """Convenience factory for evaluation datamodule construction."""
    return CLIPEvalDataModule(
        config_path=config_path,
        tokenizer=tokenizer,
        distributed=distributed,
        num_replicas=num_replicas,
        rank=rank,
    )


__all__ = [
    "CLIPEvalDataModule",
    "create_eval_datamodule",
]
