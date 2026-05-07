"""Shared base utilities for data modules.

This class centralizes tokenization and DataLoader construction logic so
train/eval data modules stay consistent and easier to maintain.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data_logic.dataset import CLIPCollator, CLIPDataset
from src.data_logic.sampler import SamplerBundle, build_torch_generator
from src.data_logic.transforms import get_data_config
from src.utils.config import DataConfig


class BaseDataModule:
    """Shared behavior for datamodule implementations."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        data_cfg: DataConfig | None = None,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
    ) -> None:
        self.data_cfg = data_cfg or get_data_config(config_path)
        self.config_path = config_path

        self._tokenizer_override = tokenizer
        self._tokenizer_cache: PreTrainedTokenizerBase | None = None
        self.collator: CLIPCollator | None = None

    def _needs_tokenizer(self) -> bool:
        return (
            self.data_cfg.datamodule.tokenize_in_dataset
            or self.data_cfg.datamodule.use_tokenizer_in_collate
            or self._tokenizer_override is not None
        )

    def _get_tokenizer(self) -> PreTrainedTokenizerBase | None:
        if not self._needs_tokenizer():
            return None

        if self._tokenizer_cache is not None:
            return self._tokenizer_cache

        tokenizer_spec = self._tokenizer_override or self.data_cfg.text.tokenizer_name
        if isinstance(tokenizer_spec, str):
            self._tokenizer_cache = AutoTokenizer.from_pretrained(tokenizer_spec)
        else:
            self._tokenizer_cache = tokenizer_spec
        return self._tokenizer_cache

    def _build_loader(
        self,
        *,
        dataset: CLIPDataset,
        sampler_bundle: SamplerBundle,
    ) -> DataLoader:
        if self.collator is None:
            collator_tokenizer = (
                self._get_tokenizer()
                if self.data_cfg.datamodule.use_tokenizer_in_collate
                else None
            )
            self.collator = CLIPCollator(
                tokenizer=collator_tokenizer,
                data_cfg=self.data_cfg,
            )

        cfg = self.data_cfg.loader
        loader_kwargs: dict[str, object] = {
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "persistent_workers": cfg.persistent_workers if cfg.num_workers > 0 else False,
            "worker_init_fn": sampler_bundle.worker_init_fn,
            "generator": build_torch_generator(self.data_cfg.split.seed),
            "collate_fn": self.collator,
        }

        if sampler_bundle.batch_sampler is not None:
            loader_kwargs["batch_sampler"] = sampler_bundle.batch_sampler
        else:
            loader_kwargs.update(
                {
                    "batch_size": cfg.batch_size,
                    "sampler": sampler_bundle.sampler,
                    "shuffle": sampler_bundle.shuffle,
                    "drop_last": sampler_bundle.drop_last,
                }
            )

        return DataLoader(dataset, **loader_kwargs)
