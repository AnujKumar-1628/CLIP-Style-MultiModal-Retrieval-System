"""Dataset and split utilities for CLIP-style image-text training.

This module provides:
1) Captions manifest loading/validation
2) Leak-safe split generation by image id
3) A PyTorch Dataset that applies project transforms and text preprocessing
4) Optional tokenizer-aware collate function
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data_logic.transforms import (
    build_eval_image_transform,
    build_text_preprocessor,
    build_train_image_transform,
    get_data_config,
)
from src.utils.config import DataConfig
from src.utils.logger import setup_logger


LOGGER = setup_logger(
    name="data_dataset",
    level="INFO",
    use_console=True,
    use_file=True,
)

SplitName = Literal["train", "val", "test"]

_CANONICAL_COLUMNS = ("image_name", "caption_id", "caption")
_COLUMN_ALIASES = {
    "comment_number": "caption_id",
    "comment": "caption",
}


@dataclass(frozen=True)
class SampleRecord:
    """In-memory sample metadata used by the dataset."""

    image_name: str
    caption_id: int
    caption: str
    image_path: Path


def _resolve_split_path(data_cfg: DataConfig, split: SplitName) -> Path:
    split_dir = data_cfg.split.output_dir
    split_dir.mkdir(parents=True, exist_ok=True)

    filenames: dict[SplitName, str] = {
        "train": data_cfg.split.train_filename,
        "val": data_cfg.split.val_filename,
        "test": data_cfg.split.test_filename,
    }
    return split_dir / filenames[split]


def _standardize_columns(df: pd.DataFrame, context_name: str) -> pd.DataFrame:
    renamed = df.rename(columns=_COLUMN_ALIASES).copy()
    missing_cols = [column for column in _CANONICAL_COLUMNS if column not in renamed.columns]
    if missing_cols:
        raise ValueError(
            f"{context_name}: missing required columns {missing_cols}. "
            f"Expected columns include {_CANONICAL_COLUMNS} or aliases {_COLUMN_ALIASES}."
        )

    standardized = renamed.loc[:, _CANONICAL_COLUMNS].copy()

    standardized["image_name"] = standardized["image_name"].astype(str).str.strip()
    standardized["caption"] = standardized["caption"].astype(str).str.strip()
    standardized["caption_id"] = pd.to_numeric(
        standardized["caption_id"], errors="coerce"
    ).astype("Int64")

    standardized = standardized.dropna(subset=["caption_id"])
    standardized["caption_id"] = standardized["caption_id"].astype(int)
    standardized = standardized[
        (standardized["image_name"] != "") & (standardized["caption"] != "")
    ].reset_index(drop=True)

    if standardized.empty:
        raise ValueError(f"{context_name}: no valid rows after validation.")
    return standardized


def load_captions_manifest(
    captions_file: str | Path,
    *,
    validate_images_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw captions CSV and return standardized dataframe."""
    captions_path = Path(captions_file)
    if not captions_path.exists():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    df = pd.read_csv(captions_path)
    df = _standardize_columns(df, context_name=f"captions manifest '{captions_path}'")

    if validate_images_dir is not None:
        images_dir = Path(validate_images_dir)
        image_exists = df["image_name"].map(lambda name: (images_dir / name).exists())
        missing_count = int((~image_exists).sum())
        if missing_count:
            LOGGER.warning(
                "Found %d caption rows whose image file is missing under %s.",
                missing_count,
                images_dir,
            )
        df = df[image_exists].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No valid caption rows remain after image-path validation against {images_dir}."
            )

    LOGGER.info("Loaded captions manifest with %d rows.", len(df))
    return df


def _split_by_image_id(df: pd.DataFrame, data_cfg: DataConfig) -> dict[SplitName, pd.DataFrame]:
    unique_images = df["image_name"].drop_duplicates().to_numpy()
    if unique_images.size == 0:
        raise ValueError("Cannot create splits from an empty captions table.")

    rng = np.random.default_rng(data_cfg.split.seed)
    rng.shuffle(unique_images)

    n_images = unique_images.size
    n_train = int(n_images * data_cfg.split.train_ratio)
    n_val = int(n_images * data_cfg.split.val_ratio)
    n_test = n_images - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            "Split sizes must all be > 0. "
            f"Got train={n_train}, val={n_val}, test={n_test} for {n_images} images."
        )

    train_images = set(unique_images[:n_train])
    val_images = set(unique_images[n_train : n_train + n_val])
    test_images = set(unique_images[n_train + n_val :])

    split_frames: dict[SplitName, pd.DataFrame] = {
        "train": df[df["image_name"].isin(train_images)].reset_index(drop=True),
        "val": df[df["image_name"].isin(val_images)].reset_index(drop=True),
        "test": df[df["image_name"].isin(test_images)].reset_index(drop=True),
    }

    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"Created empty '{split_name}' split; check your dataset/config.")
        LOGGER.info(
            "Split '%s': %d rows, %d unique images.",
            split_name,
            len(split_df),
            split_df["image_name"].nunique(),
        )

    return split_frames


def build_or_load_split_tables(
    data_cfg: DataConfig,
    *,
    force_rebuild: bool = False,
) -> dict[SplitName, pd.DataFrame]:
    """Load split CSVs if available, else build them from caption manifest."""
    split_paths = {
        split: _resolve_split_path(data_cfg, split)
        for split in ("train", "val", "test")
    }

    split_files_exist = all(path.exists() for path in split_paths.values())
    if split_files_exist and not force_rebuild:
        loaded = {
            split: _standardize_columns(
                pd.read_csv(path),
                context_name=f"{split} split '{path}'",
            )
            for split, path in split_paths.items()
        }
        LOGGER.info("Loaded existing split files from %s.", data_cfg.split.output_dir)
        return loaded

    manifest_df = load_captions_manifest(
        data_cfg.dataset.captions_file,
        validate_images_dir=data_cfg.dataset.images_dir,
    )
    splits = _split_by_image_id(manifest_df, data_cfg)

    for split, split_df in splits.items():
        path = split_paths[split]
        split_df.to_csv(path, index=False)
        LOGGER.info("Wrote split file: %s", path)

    return splits


def _one_caption_per_image(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    # We sample one caption per image to avoid overweighting images with many captions.
    sampled = (
        df.groupby("image_name", group_keys=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled


class CLIPDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset for image-text pairs."""

    def __init__(
        self,
        *,
        split: SplitName,
        config_path: str | Path | None = None,
        data_cfg: DataConfig | None = None,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        text_preprocessor: Callable[[str], str] | None = None,
        one_caption_per_image: bool | None = None,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        tokenize_in_dataset: bool | None = None,
        fail_on_missing_image: bool | None = None,
        force_rebuild_splits: bool | None = None,
        use_train_augmentations: bool | None = None,
    ) -> None:
        self.split = split
        self.data_cfg = data_cfg or get_data_config(config_path)
        if fail_on_missing_image is None:
            fail_on_missing_image = self.data_cfg.datamodule.fail_on_missing_image
        if tokenize_in_dataset is None:
            tokenize_in_dataset = self.data_cfg.datamodule.tokenize_in_dataset
        if force_rebuild_splits is None:
            force_rebuild_splits = self.data_cfg.datamodule.force_rebuild_splits

        self.fail_on_missing_image = fail_on_missing_image
        self.tokenize_in_dataset = tokenize_in_dataset
        self._epoch = 0

        split_tables = build_or_load_split_tables(
            self.data_cfg, force_rebuild=force_rebuild_splits
        )
        split_df = split_tables[split]

        if one_caption_per_image is None:
            one_caption_per_image = (
                split == "train" and self.data_cfg.split.random_one_caption_per_image
            )
        self._one_caption_per_image = bool(one_caption_per_image)
        self._base_seed = int(self.data_cfg.split.seed)
        self._split_df_full = split_df.reset_index(drop=True).copy()
        split_df_for_records = (
            _one_caption_per_image(self._split_df_full, seed=self._base_seed)
            if self._one_caption_per_image
            else self._split_df_full
        )

        if use_train_augmentations is None:
            use_train_augmentations = self.data_cfg.datamodule.use_train_augmentations

        self.image_transform = image_transform or (
            build_train_image_transform(
                self.data_cfg,
                use_augmentations=use_train_augmentations,
            )
            if split == "train"
            else build_eval_image_transform(self.data_cfg)
        )
        self.text_preprocessor = text_preprocessor or build_text_preprocessor(self.data_cfg)

        self.tokenizer = self._init_tokenizer(tokenizer)

        images_dir = self.data_cfg.dataset.images_dir
        self.records: list[SampleRecord] = []
        self._set_records_from_dataframe(split_df_for_records, images_dir=images_dir)

        LOGGER.info(
            "Initialized CLIPDataset | split=%s | samples=%d | one_caption_per_image=%s | epoch=%d",
            split,
            len(self.records),
            self._one_caption_per_image,
            self._epoch,
        )

    def _set_records_from_dataframe(self, df: pd.DataFrame, *, images_dir: Path) -> None:
        self.records = [
            SampleRecord(
                image_name=row.image_name,
                caption_id=int(row.caption_id),
                caption=row.caption,
                image_path=images_dir / row.image_name,
            )
            for row in df.itertuples(index=False)
        ]

    def set_epoch(self, epoch: int) -> None:
        """Update epoch state and re-sample captions when enabled."""
        self._epoch = int(epoch)
        if not self._one_caption_per_image:
            return

        sampled_df = _one_caption_per_image(
            self._split_df_full,
            seed=self._base_seed + self._epoch,
        )
        self._set_records_from_dataframe(sampled_df, images_dir=self.data_cfg.dataset.images_dir)
        LOGGER.info(
            "Resampled one-caption-per-image mapping | split=%s | epoch=%d | seed=%d | samples=%d",
            self.split,
            self._epoch,
            self._base_seed + self._epoch,
            len(self.records),
        )

    def _init_tokenizer(
        self, tokenizer: PreTrainedTokenizerBase | str | None
    ) -> PreTrainedTokenizerBase | None:
        if tokenizer is None:
            return None
        if isinstance(tokenizer, str):
            return AutoTokenizer.from_pretrained(tokenizer)
        return tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: Path) -> Image.Image:
        if not image_path.exists():
            if self.fail_on_missing_image:
                raise FileNotFoundError(f"Image file not found: {image_path}")
            LOGGER.warning("Missing image file. Using black fallback image: %s", image_path)
            return Image.new("RGB", (self.data_cfg.image.size, self.data_cfg.image.size))

        try:
            with Image.open(image_path) as image:
                return image.copy()
        except (OSError, UnidentifiedImageError) as exc:
            if self.fail_on_missing_image:
                raise RuntimeError(f"Failed to load image '{image_path}': {exc}") from exc
            LOGGER.warning("Failed to load image. Using black fallback image: %s", image_path)
            return Image.new("RGB", (self.data_cfg.image.size, self.data_cfg.image.size))

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        image = self._load_image(record.image_path)
        image_tensor = self.image_transform(image)

        text = self.text_preprocessor(record.caption)

        sample: dict[str, Any] = {
            "image": image_tensor,
            "text": text,
            "image_name": record.image_name,
            "caption_id": record.caption_id,
            "image_path": str(record.image_path),
        }

        if self.tokenize_in_dataset:
            if self.tokenizer is None:
                raise ValueError(
                    "tokenize_in_dataset=True requires a tokenizer instance or tokenizer name."
                )
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.data_cfg.text.max_length,
                padding=False,
                return_tensors="pt",
            )
            sample["tokenized_text"] = {k: v.squeeze(0) for k, v in tokenized.items()}

        return sample


class CLIPCollator:
    """Batch collator with optional tokenizer support."""

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        data_cfg: DataConfig | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.data_cfg = data_cfg or get_data_config(config_path)
        self.tokenizer = self._init_tokenizer(tokenizer)

    def _init_tokenizer(
        self, tokenizer: PreTrainedTokenizerBase | str | None
    ) -> PreTrainedTokenizerBase | None:
        if tokenizer is None:
            return None
        if isinstance(tokenizer, str):
            return AutoTokenizer.from_pretrained(tokenizer)
        return tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Received empty batch in CLIPCollator.")

        images = torch.stack([item["image"] for item in batch], dim=0)
        texts = [item["text"] for item in batch]

        collated: dict[str, Any] = {
            "images": images,
            "texts": texts,
            "image_names": [item["image_name"] for item in batch],
            "caption_ids": torch.tensor(
                [int(item["caption_id"]) for item in batch], dtype=torch.long
            ),
            "image_paths": [item["image_path"] for item in batch],
        }

        if self.tokenizer is not None:
            padding_mode: str | bool = (
                "longest" if self.data_cfg.text.dynamic_padding else "max_length"
            )
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.data_cfg.text.max_length,
                padding=padding_mode,
                return_tensors="pt",
            )
            collated["text_tokens"] = tokenized

        return collated


def create_dataset(
    split: SplitName,
    *,
    config_path: str | Path | None = None,
    tokenizer: PreTrainedTokenizerBase | str | None = None,
    tokenize_in_dataset: bool | None = None,
    force_rebuild_splits: bool | None = None,
    use_train_augmentations: bool | None = None,
) -> CLIPDataset:
    """Convenience factory for common dataset construction."""
    return CLIPDataset(
        split=split,
        config_path=config_path,
        tokenizer=tokenizer,
        tokenize_in_dataset=tokenize_in_dataset,
        force_rebuild_splits=force_rebuild_splits,
        use_train_augmentations=use_train_augmentations,
    )


__all__ = [
    "CLIPCollator",
    "CLIPDataset",
    "SampleRecord",
    "build_or_load_split_tables",
    "create_dataset",
    "load_captions_manifest",
]
