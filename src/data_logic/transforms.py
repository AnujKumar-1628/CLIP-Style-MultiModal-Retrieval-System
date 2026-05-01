"""Data transform builders for training/evaluation pipelines.

This module keeps all image/text preprocessing in one place and reads defaults
from `configs/data.yaml` through `src.utils.config`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.utils.config import DataConfig, load_data_config
from src.utils.logger import setup_logger


LOGGER = setup_logger(
    name="data_transforms",
    level="INFO",
    use_console=True,
    use_file=True,
)


class EnsureRGB:
    """Convert input PIL image to RGB to avoid channel-mismatch at runtime."""

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            return image.convert("RGB")
        return image


def _to_interpolation_mode(value: str) -> InterpolationMode:
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    key = value.strip().lower()
    if key not in mapping:
        supported = ", ".join(sorted(mapping))
        LOGGER.error(
            "Invalid interpolation requested: '%s'. Supported values: %s",
            value,
            supported,
        )
        raise ValueError(f"Unsupported interpolation '{value}'. Supported values: {supported}")
    return mapping[key]


@lru_cache(maxsize=1)
def _cached_default_data_config() -> DataConfig:
    """Load and cache default data config once per process."""
    cfg = load_data_config()
    LOGGER.info("Loaded default data config from configs/data.yaml")
    return cfg


def get_data_config(config_path: str | Path | None = None) -> DataConfig:
    """
    Return data config from path, or cached default config when path is omitted.
    """
    if config_path is None:
        return _cached_default_data_config()
    cfg = load_data_config(config_path)
    LOGGER.info("Loaded data config from: %s", Path(config_path))
    return cfg


def build_train_image_transform(
    data_cfg: DataConfig | None = None,
    use_augmentations: bool = True,
) -> transforms.Compose:
    """
    Build training image transform.

    Default behavior:
    - Always convert to RGB
    - If `use_augmentations=True`: random resized crop + horizontal flip
    - Else: deterministic resize + center crop
    - Normalize with config mean/std (ImageNet by default)
    """
    cfg = data_cfg or get_data_config()
    size = cfg.image.size
    interpolation = _to_interpolation_mode(cfg.image.interpolation)

    ops: list[Callable] = [EnsureRGB()]

    if use_augmentations:
        ops.extend(
            [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=interpolation,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    else:
        # Keep the same final tensor size while avoiding randomness.
        ops.extend(
            [
                transforms.Resize(size=size + 32, interpolation=interpolation),
                transforms.CenterCrop(size=size),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.image.normalize_mean,
                std=cfg.image.normalize_std,
            ),
        ]
    )

    transform = transforms.Compose(ops)
    LOGGER.info(
        "Built train image transform | size=%d | interpolation=%s | augmentations=%s",
        size,
        cfg.image.interpolation,
        use_augmentations,
    )
    return transform


def build_eval_image_transform(data_cfg: DataConfig | None = None) -> transforms.Compose:
    """
    Build deterministic validation/test transform.
    """
    cfg = data_cfg or get_data_config()
    size = cfg.image.size
    interpolation = _to_interpolation_mode(cfg.image.interpolation)

    transform = transforms.Compose(
        [
            EnsureRGB(),
            transforms.Resize(size=size + 32, interpolation=interpolation),
            transforms.CenterCrop(size=size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.image.normalize_mean,
                std=cfg.image.normalize_std,
            ),
        ]
    )
    LOGGER.info(
        "Built eval image transform | size=%d | interpolation=%s",
        size,
        cfg.image.interpolation,
    )
    return transform


def build_text_preprocessor(data_cfg: DataConfig | None = None) -> Callable[[str], str]:
    """
    Build simple text normalization callable from config.
    """
    cfg = data_cfg or get_data_config()
    lowercase = cfg.text.lowercase
    LOGGER.info("Built text preprocessor | lowercase=%s", lowercase)

    def _preprocess(text: str) -> str:
        out = str(text).strip()
        if lowercase:
            out = out.lower()
        return out

    return _preprocess


def get_transform_bundle(
    config_path: str | Path | None = None,
    use_train_augmentations: bool = True,
) -> dict[str, Callable]:
    """
    Return all commonly-used transforms/preprocessors in one dict.
    """
    cfg = get_data_config(config_path)
    bundle = {
        "train_image": build_train_image_transform(cfg, use_augmentations=use_train_augmentations),
        "eval_image": build_eval_image_transform(cfg),
        "text": build_text_preprocessor(cfg),
    }
    LOGGER.info(
        "Built transform bundle | keys=%s | train_augmentations=%s",
        sorted(bundle.keys()),
        use_train_augmentations,
    )
    return bundle
