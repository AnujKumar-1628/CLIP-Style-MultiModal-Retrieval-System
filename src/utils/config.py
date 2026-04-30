"""Configuration loading and validation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for config loading. Install with `pip install pyyaml`."
    ) from exc


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    captions_file: Path
    images_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    output_dir: Path
    train_filename: str
    val_filename: str
    test_filename: str
    random_one_caption_per_image: bool


@dataclass(frozen=True)
class ImageConfig:
    size: int
    interpolation: str
    normalize_mean: list[float]
    normalize_std: list[float]


@dataclass(frozen=True)
class TextConfig:
    tokenizer_name: str
    max_length: int
    dynamic_padding: bool
    lowercase: bool


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool


@dataclass(frozen=True)
class DataModuleConfig:
    force_rebuild_splits: bool
    fail_on_missing_image: bool
    tokenize_in_dataset: bool
    use_tokenizer_in_collate: bool
    use_train_augmentations: bool
    use_unique_image_batch_sampler: bool
    distributed: bool
    distributed_num_replicas: int | None
    distributed_rank: int | None
    eval_default_split: str
    eval_query_one_caption_per_image: bool
    eval_gallery_one_caption_per_image: bool


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetConfig
    split: SplitConfig
    image: ImageConfig
    text: TextConfig
    loader: LoaderConfig
    datamodule: DataModuleConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert typed config to a serializable dictionary."""
        return asdict(self)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _require_section(raw_cfg: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in raw_cfg or raw_cfg[section] is None:
        raise ValueError(f"Missing required config section: '{section}'")
    if not isinstance(raw_cfg[section], dict):
        raise TypeError(f"Config section '{section}' must be a mapping/dict.")
    return raw_cfg[section]


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def _validate_image_stats(mean: list[float], std: list[float]) -> None:
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("image.normalize_mean and image.normalize_std must each contain 3 values.")
    if any(v <= 0 for v in std):
        raise ValueError("image.normalize_std values must be > 0.")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary."""
    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as file:
        raw_cfg = yaml.safe_load(file) or {}

    if not isinstance(raw_cfg, dict):
        raise TypeError(f"Top-level config must be a mapping/dict, got: {type(raw_cfg).__name__}")
    return raw_cfg


def parse_data_config(raw_cfg: dict[str, Any]) -> DataConfig:
    """Parse and validate the data config dictionary into typed objects."""
    dataset_raw = _require_section(raw_cfg, "dataset")
    split_raw = _require_section(raw_cfg, "split")
    image_raw = _require_section(raw_cfg, "image")
    text_raw = _require_section(raw_cfg, "text")
    loader_raw = _require_section(raw_cfg, "loader")
    datamodule_raw = raw_cfg.get("datamodule", {})
    if not isinstance(datamodule_raw, dict):
        raise TypeError("Config section 'datamodule' must be a mapping/dict when provided.")

    dataset_cfg = DatasetConfig(
        name=str(dataset_raw["name"]),
        captions_file=_resolve_path(dataset_raw["captions_file"]),
        images_dir=_resolve_path(dataset_raw["images_dir"]),
    )

    split_cfg = SplitConfig(
        strategy=str(split_raw["strategy"]),
        train_ratio=float(split_raw["train_ratio"]),
        val_ratio=float(split_raw["val_ratio"]),
        test_ratio=float(split_raw["test_ratio"]),
        seed=int(split_raw.get("seed", 42)),
        output_dir=_resolve_path(split_raw["output_dir"]),
        train_filename=str(split_raw.get("train_filename", "train.csv")),
        val_filename=str(split_raw.get("val_filename", "val.csv")),
        test_filename=str(split_raw.get("test_filename", "test.csv")),
        random_one_caption_per_image=bool(split_raw.get("random_one_caption_per_image", True)),
    )

    image_cfg = ImageConfig(
        size=int(image_raw["size"]),
        interpolation=str(image_raw.get("interpolation", "bicubic")),
        normalize_mean=[float(v) for v in image_raw["normalize_mean"]],
        normalize_std=[float(v) for v in image_raw["normalize_std"]],
    )

    text_cfg = TextConfig(
        tokenizer_name=str(text_raw["tokenizer_name"]),
        max_length=int(text_raw["max_length"]),
        dynamic_padding=bool(text_raw.get("dynamic_padding", True)),
        lowercase=bool(text_raw.get("lowercase", True)),
    )

    loader_cfg = LoaderConfig(
        batch_size=int(loader_raw.get("batch_size", 32)),
        num_workers=int(loader_raw.get("num_workers", 4)),
        pin_memory=bool(loader_raw.get("pin_memory", True)),
        persistent_workers=bool(loader_raw.get("persistent_workers", True)),
        drop_last=bool(loader_raw.get("drop_last", False)),
    )

    dm_num_replicas_raw = datamodule_raw.get("distributed_num_replicas")
    dm_rank_raw = datamodule_raw.get("distributed_rank")

    datamodule_cfg = DataModuleConfig(
        force_rebuild_splits=bool(datamodule_raw.get("force_rebuild_splits", False)),
        fail_on_missing_image=bool(datamodule_raw.get("fail_on_missing_image", True)),
        tokenize_in_dataset=bool(datamodule_raw.get("tokenize_in_dataset", False)),
        use_tokenizer_in_collate=bool(datamodule_raw.get("use_tokenizer_in_collate", False)),
        use_train_augmentations=bool(datamodule_raw.get("use_train_augmentations", True)),
        use_unique_image_batch_sampler=bool(
            datamodule_raw.get("use_unique_image_batch_sampler", False)
        ),
        distributed=bool(datamodule_raw.get("distributed", False)),
        distributed_num_replicas=(
            int(dm_num_replicas_raw) if dm_num_replicas_raw is not None else None
        ),
        distributed_rank=(int(dm_rank_raw) if dm_rank_raw is not None else None),
        eval_default_split=str(datamodule_raw.get("eval_default_split", "test")).strip().lower(),
        eval_query_one_caption_per_image=bool(
            datamodule_raw.get("eval_query_one_caption_per_image", False)
        ),
        eval_gallery_one_caption_per_image=bool(
            datamodule_raw.get("eval_gallery_one_caption_per_image", True)
        ),
    )

    if split_cfg.strategy != "by_image_id":
        raise ValueError(
            "split.strategy must be 'by_image_id' to avoid caption/image leakage."
        )

    _validate_split_ratios(split_cfg.train_ratio, split_cfg.val_ratio, split_cfg.test_ratio)

    if image_cfg.size <= 0:
        raise ValueError("image.size must be > 0.")
    _validate_image_stats(image_cfg.normalize_mean, image_cfg.normalize_std)

    if text_cfg.max_length <= 0:
        raise ValueError("text.max_length must be > 0.")

    if loader_cfg.batch_size <= 0:
        raise ValueError("loader.batch_size must be > 0.")
    if loader_cfg.num_workers < 0:
        raise ValueError("loader.num_workers must be >= 0.")
    if datamodule_cfg.distributed_num_replicas is not None and datamodule_cfg.distributed_num_replicas <= 0:
        raise ValueError("datamodule.distributed_num_replicas must be > 0 when provided.")
    if datamodule_cfg.distributed_rank is not None and datamodule_cfg.distributed_rank < 0:
        raise ValueError("datamodule.distributed_rank must be >= 0 when provided.")
    if datamodule_cfg.eval_default_split not in {"val", "test"}:
        raise ValueError("datamodule.eval_default_split must be either 'val' or 'test'.")

    return DataConfig(
        dataset=dataset_cfg,
        split=split_cfg,
        image=image_cfg,
        text=text_cfg,
        loader=loader_cfg,
        datamodule=datamodule_cfg,
    )


def load_data_config(path: str | Path | None = None) -> DataConfig:
    """
    Load and validate `configs/data.yaml`.

    Args:
        path: Optional path override. Uses `<project_root>/configs/data.yaml` if omitted.
    """
    config_path = path or (CONFIGS_DIR / "data.yaml")
    raw_cfg = load_yaml(config_path)
    return parse_data_config(raw_cfg)
