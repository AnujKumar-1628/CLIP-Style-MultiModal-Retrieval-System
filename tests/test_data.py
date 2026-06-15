"""Tests for data configuration parsing and validation."""

from __future__ import annotations

import pytest

from src.utils.config import parse_data_config


def _base_data_config() -> dict:
    return {
        "dataset": {
            "name": "flickr",
            "captions_file": "data/captions.csv",
            "images_dir": "data/images",
        },
        "split": {
            "strategy": "by_image_id",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 42,
            "output_dir": "artifacts/splits",
        },
        "image": {
            "size": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "text": {
            "tokenizer_name": "distilbert-base-uncased",
            "max_length": 64,
        },
        "loader": {
            "batch_size": 16,
            "num_workers": 0,
        },
    }


def test_parse_data_config_success() -> None:
    cfg = parse_data_config(_base_data_config())
    assert cfg.dataset.name == "flickr"
    assert cfg.split.strategy == "by_image_id"
    assert cfg.loader.batch_size == 16


def test_parse_data_config_rejects_invalid_split_ratio_sum() -> None:
    raw = _base_data_config()
    raw["split"]["test_ratio"] = 0.2
    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        parse_data_config(raw)


def test_parse_data_config_rejects_invalid_eval_split() -> None:
    raw = _base_data_config()
    raw["datamodule"] = {"eval_default_split": "train"}
    with pytest.raises(ValueError, match="eval_default_split"):
        parse_data_config(raw)
