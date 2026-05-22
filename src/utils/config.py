"""Centralized configuration schema, loading, and validation utilities.

File organization:
- Data pipeline config (`configs/data.yaml`)
- Model + loss config (`configs/model.yaml`)
- Training config (`configs/training.yaml`)
- Retrieval config (`configs/retrieval.yaml`)
- API + inference config (`configs/api.yaml`)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from src.utils.paths import CONFIGS_DIR, PROJECT_ROOT

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for config loading. Install with `pip install pyyaml`."
    ) from exc


# ============================================================================
# Data Config (`configs/data.yaml`)
# ============================================================================


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset source files and directories."""

    name: str
    captions_file: Path
    images_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    """Train/val/test split strategy and output files."""

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
    """Image preprocessing settings."""

    size: int
    interpolation: str
    normalize_mean: list[float]
    normalize_std: list[float]


@dataclass(frozen=True)
class TextConfig:
    """Text preprocessing and tokenizer settings."""

    tokenizer_name: str
    max_length: int
    dynamic_padding: bool
    lowercase: bool


@dataclass(frozen=True)
class LoaderConfig:
    """PyTorch DataLoader runtime settings."""

    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool


@dataclass(frozen=True)
class DataModuleConfig:
    """Dataset/DataModule behavior toggles used at runtime."""

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
    val_one_caption_per_image: bool
    test_one_caption_per_image: bool


@dataclass(frozen=True)
class DataConfig:
    """Typed root object for `configs/data.yaml`."""

    dataset: DatasetConfig
    split: SplitConfig
    image: ImageConfig
    text: TextConfig
    loader: LoaderConfig
    datamodule: DataModuleConfig

    def to_dict(self) -> dict[str, Any]:
        """Convert typed config to a serializable dictionary."""
        return asdict(self)


# ----------------------------------------------------------------------------
# Shared parsing helpers (used across all config families)
# ----------------------------------------------------------------------------


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


def _validate_split_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(
            f"split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def _validate_image_stats(mean: list[float], std: list[float]) -> None:
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "image.normalize_mean and image.normalize_std must each contain 3 values."
        )
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
        raise TypeError(
            f"Top-level config must be a mapping/dict, got: {type(raw_cfg).__name__}"
        )
    return raw_cfg


# ----------------------------------------------------------------------------
# Data config parser/loader
# ----------------------------------------------------------------------------


def parse_data_config(raw_cfg: dict[str, Any]) -> DataConfig:
    """Parse and validate the data config dictionary into typed objects."""
    dataset_raw = _require_section(raw_cfg, "dataset")
    split_raw = _require_section(raw_cfg, "split")
    image_raw = _require_section(raw_cfg, "image")
    text_raw = _require_section(raw_cfg, "text")
    loader_raw = _require_section(raw_cfg, "loader")
    datamodule_raw = raw_cfg.get("datamodule", {})
    if not isinstance(datamodule_raw, dict):
        raise TypeError(
            "Config section 'datamodule' must be a mapping/dict when provided."
        )

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
        random_one_caption_per_image=bool(
            split_raw.get("random_one_caption_per_image", True)
        ),
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
        use_tokenizer_in_collate=bool(
            datamodule_raw.get("use_tokenizer_in_collate", False)
        ),
        use_train_augmentations=bool(
            datamodule_raw.get("use_train_augmentations", True)
        ),
        use_unique_image_batch_sampler=bool(
            datamodule_raw.get("use_unique_image_batch_sampler", False)
        ),
        distributed=bool(datamodule_raw.get("distributed", False)),
        distributed_num_replicas=(
            int(dm_num_replicas_raw) if dm_num_replicas_raw is not None else None
        ),
        distributed_rank=(int(dm_rank_raw) if dm_rank_raw is not None else None),
        eval_default_split=str(datamodule_raw.get("eval_default_split", "test"))
        .strip()
        .lower(),
        eval_query_one_caption_per_image=bool(
            datamodule_raw.get("eval_query_one_caption_per_image", False)
        ),
        eval_gallery_one_caption_per_image=bool(
            datamodule_raw.get("eval_gallery_one_caption_per_image", True)
        ),
        val_one_caption_per_image=bool(
            datamodule_raw.get("val_one_caption_per_image", True)
        ),
        test_one_caption_per_image=bool(
            datamodule_raw.get("test_one_caption_per_image", False)
        ),
    )

    if split_cfg.strategy != "by_image_id":
        raise ValueError(
            "split.strategy must be 'by_image_id' to avoid caption/image leakage."
        )

    _validate_split_ratios(
        split_cfg.train_ratio, split_cfg.val_ratio, split_cfg.test_ratio
    )

    if image_cfg.size <= 0:
        raise ValueError("image.size must be > 0.")
    _validate_image_stats(image_cfg.normalize_mean, image_cfg.normalize_std)

    if text_cfg.max_length <= 0:
        raise ValueError("text.max_length must be > 0.")

    if loader_cfg.batch_size <= 0:
        raise ValueError("loader.batch_size must be > 0.")
    if loader_cfg.num_workers < 0:
        raise ValueError("loader.num_workers must be >= 0.")
    if (
        datamodule_cfg.distributed_num_replicas is not None
        and datamodule_cfg.distributed_num_replicas <= 0
    ):
        raise ValueError(
            "datamodule.distributed_num_replicas must be > 0 when provided."
        )
    if (
        datamodule_cfg.distributed_rank is not None
        and datamodule_cfg.distributed_rank < 0
    ):
        raise ValueError("datamodule.distributed_rank must be >= 0 when provided.")
    if datamodule_cfg.eval_default_split not in {"val", "test"}:
        raise ValueError(
            "datamodule.eval_default_split must be either 'val' or 'test'."
        )

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


# ============================================================================
# Model + Loss Config (`configs/model.yaml`)
# ============================================================================


ModelLossReduction = Literal["mean", "sum", "none"]


@dataclass(frozen=True)
class CLIPModelConfig:
    """Global CLIP model runtime and similarity settings."""

    model_name: str
    device: str
    dtype: str
    normalize_embeddings: bool
    logit_scale_init_value: float
    logit_scale_learnable: bool
    logit_scale_clamp_min: float
    logit_scale_clamp_max: float


@dataclass(frozen=True)
class ResNet50EncoderConfig:
    """Image encoder (ResNet50) settings."""

    architecture: str
    pretrained: bool
    weights: str | None
    trainable: bool
    freeze_backbone: bool
    freeze_until: str | None
    out_dim: int
    dropout: float
    device: str
    dtype: str
    normalize_output: bool
    channels_last: bool
    allow_random_init_fallback: bool
    unfreeze_layers: tuple[str, ...]


@dataclass(frozen=True)
class DistilBertEncoderConfig:
    """Text encoder (DistilBERT) settings."""

    architecture: str
    pretrained: bool
    trainable: bool
    freeze_backbone: bool
    freeze_until: str | None
    out_dim: int
    dropout: float
    max_length: int
    dynamic_padding: bool
    lowercase: bool
    device: str
    dtype: str
    normalize_output: bool
    allow_random_init_fallback: bool
    allow_simple_tokenizer_fallback: bool
    unfreeze_layers: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionHeadConfig:
    """Single projection head settings."""

    input_dim: int
    hidden_dim: int
    output_dim: int
    activation: str
    dropout: float
    use_layer_norm: bool
    normalize_output: bool
    dtype: str
    device: str


@dataclass(frozen=True)
class ProjectionConfigBundle:
    """Image + text projection heads with shared embedding dimension."""

    embed_dim: int
    image: ProjectionHeadConfig
    text: ProjectionHeadConfig


@dataclass(frozen=True)
class ContrastiveLossConfig:
    """Contrastive loss settings."""

    name: str
    label_smoothing: float
    reduction: ModelLossReduction


@dataclass(frozen=True)
class BackboneUnfreezeConfig:
    """Backbone unfreeze schedule used by training callbacks."""

    enabled: bool
    at_epoch: int
    lr: float


def _load_model_config_raw(path: str | Path | None = None) -> dict[str, Any]:
    config_path = path or (CONFIGS_DIR / "model.yaml")
    return load_yaml(config_path)


def load_backbone_unfreeze_config(
    path: str | Path | None = None,
) -> BackboneUnfreezeConfig:
    """Load backbone-unfreeze schedule from `configs/model.yaml`."""
    raw = _load_model_config_raw(path)

    image_raw = raw.get("image_encoder", {})
    text_raw = raw.get("text_encoder", {})

    training_raw = raw.get("training", {})
    if not isinstance(training_raw, dict):
        raise TypeError("model.training must be a mapping/dict when provided.")

    unfreeze_raw = training_raw.get("backbone_unfreeze", {})
    if not isinstance(unfreeze_raw, dict):
        raise TypeError(
            "model.training.backbone_unfreeze must be a mapping/dict when provided."
        )

    default_enabled = bool(image_raw.get("freeze_backbone", False)) or bool(
        text_raw.get("freeze_backbone", False)
    )
    cfg = BackboneUnfreezeConfig(
        enabled=bool(unfreeze_raw.get("enabled", default_enabled)),
        at_epoch=int(unfreeze_raw.get("at_epoch", 4)),
        lr=float(unfreeze_raw.get("lr", 1e-5)),
    )
    if cfg.at_epoch <= 0:
        raise ValueError("training.backbone_unfreeze.at_epoch must be > 0.")
    if cfg.lr <= 0:
        raise ValueError("training.backbone_unfreeze.lr must be > 0.")
    return cfg


def load_clip_model_config(path: str | Path | None = None) -> CLIPModelConfig:
    raw = _load_model_config_raw(path)

    model_raw = raw.get("model", {})
    similarity_raw = raw.get("similarity", {})
    logit_scale_raw = similarity_raw.get("logit_scale", {})

    init_value = float(logit_scale_raw.get("init_value", 0.07))
    if init_value <= 0:
        raise ValueError("similarity.logit_scale.init_value must be > 0.")

    cfg = CLIPModelConfig(
        model_name=str(model_raw.get("name", "clip_style_retrieval")),
        device=str(model_raw.get("device", "cpu")),
        dtype=str(model_raw.get("dtype", "float32")),
        normalize_embeddings=bool(similarity_raw.get("normalize_embeddings", True)),
        logit_scale_init_value=init_value,
        logit_scale_learnable=bool(logit_scale_raw.get("learnable", True)),
        logit_scale_clamp_min=float(logit_scale_raw.get("clamp_min", 0.001)),
        logit_scale_clamp_max=float(logit_scale_raw.get("clamp_max", 100.0)),
    )
    if cfg.logit_scale_clamp_min <= 0 or cfg.logit_scale_clamp_max <= 0:
        raise ValueError("logit scale clamp values must be > 0.")
    if cfg.logit_scale_clamp_min >= cfg.logit_scale_clamp_max:
        raise ValueError("logit_scale.clamp_min must be < logit_scale.clamp_max.")
    return cfg


def load_resnet50_encoder_config(
    path: str | Path | None = None,
) -> ResNet50EncoderConfig:
    raw = _load_model_config_raw(path)

    model_raw = raw.get("model", {})
    image_raw = raw.get("image_encoder", {})
    architecture = str(image_raw.get("architecture", "resnet50")).strip().lower()
    if architecture != "resnet50":
        raise ValueError(
            f"This encoder only supports architecture='resnet50'. Got '{architecture}'."
        )

    return ResNet50EncoderConfig(
        architecture=architecture,
        pretrained=bool(image_raw.get("pretrained", True)),
        weights=image_raw.get("weights"),
        trainable=bool(image_raw.get("trainable", True)),
        freeze_backbone=bool(image_raw.get("freeze_backbone", False)),
        freeze_until=(
            str(image_raw["freeze_until"]).strip()
            if image_raw.get("freeze_until")
            else None
        ),
        out_dim=int(image_raw.get("out_dim", 2048)),
        dropout=float(image_raw.get("dropout", 0.0)),
        device=str(model_raw.get("device", "cpu")),
        dtype=str(model_raw.get("dtype", "float32")),
        normalize_output=bool(image_raw.get("normalize_output", False)),
        channels_last=bool(image_raw.get("channels_last", True)),
        allow_random_init_fallback=bool(
            image_raw.get("allow_random_init_fallback", False)
        ),
        unfreeze_layers=tuple(
            str(x) for x in image_raw.get("unfreeze_layers", ["layer4", "avgpool"])
        ),
    )


def load_distilbert_encoder_config(
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> DistilBertEncoderConfig:
    model_raw = _load_model_config_raw(model_config_path)
    data_cfg = load_data_config(data_config_path)

    model_section = model_raw.get("model", {})
    text_raw = model_raw.get("text_encoder", {})

    architecture = str(text_raw.get("architecture", "distilbert-base-uncased")).strip()
    if "distilbert" not in architecture.lower():
        raise ValueError(
            f"This encoder expects a DistilBERT architecture. Got '{architecture}'."
        )

    return DistilBertEncoderConfig(
        architecture=architecture,
        pretrained=bool(text_raw.get("pretrained", True)),
        trainable=bool(text_raw.get("trainable", True)),
        freeze_backbone=bool(text_raw.get("freeze_backbone", False)),
        freeze_until=(
            str(text_raw["freeze_until"]).strip()
            if text_raw.get("freeze_until")
            else None
        ),
        out_dim=int(text_raw.get("out_dim", 768)),
        dropout=float(text_raw.get("dropout", 0.0)),
        max_length=int(data_cfg.text.max_length),
        dynamic_padding=bool(data_cfg.text.dynamic_padding),
        lowercase=bool(data_cfg.text.lowercase),
        device=str(model_section.get("device", "cpu")),
        dtype=str(model_section.get("dtype", "float32")),
        normalize_output=bool(text_raw.get("normalize_output", False)),
        allow_random_init_fallback=bool(
            text_raw.get("allow_random_init_fallback", False)
        ),
        allow_simple_tokenizer_fallback=bool(
            text_raw.get("allow_simple_tokenizer_fallback", False)
        ),
        unfreeze_layers=tuple(
            str(x)
            for x in text_raw.get("unfreeze_layers", ["layer.3", "layer.4", "layer.5"])
        ),
    )


def load_projection_config_bundle(
    path: str | Path | None = None,
) -> ProjectionConfigBundle:
    raw = _load_model_config_raw(path)

    model_raw = raw.get("model", {})
    image_encoder_raw = raw.get("image_encoder", {})
    text_encoder_raw = raw.get("text_encoder", {})
    proj_raw = raw.get("projection_heads", {})
    embed_dim = int(proj_raw.get("embed_dim", 256))
    if embed_dim <= 0:
        raise ValueError("projection_heads.embed_dim must be > 0.")

    dtype_name = str(model_raw.get("dtype", "float32"))
    device_name = str(model_raw.get("device", "cpu"))

    image_raw = proj_raw.get("image", {})
    text_raw = proj_raw.get("text", {})

    image_cfg = ProjectionHeadConfig(
        input_dim=int(image_encoder_raw.get("out_dim", 2048)),
        hidden_dim=int(image_raw.get("hidden_dim", 1024)),
        output_dim=embed_dim,
        activation=str(image_raw.get("activation", "gelu")),
        dropout=float(image_raw.get("dropout", 0.0)),
        use_layer_norm=bool(image_raw.get("use_layer_norm", True)),
        normalize_output=bool(image_raw.get("normalize_output", False)),
        dtype=dtype_name,
        device=device_name,
    )
    text_cfg = ProjectionHeadConfig(
        input_dim=int(text_encoder_raw.get("out_dim", 768)),
        hidden_dim=int(text_raw.get("hidden_dim", 768)),
        output_dim=embed_dim,
        activation=str(text_raw.get("activation", "gelu")),
        dropout=float(text_raw.get("dropout", 0.0)),
        use_layer_norm=bool(text_raw.get("use_layer_norm", True)),
        normalize_output=bool(text_raw.get("normalize_output", False)),
        dtype=dtype_name,
        device=device_name,
    )
    return ProjectionConfigBundle(embed_dim=embed_dim, image=image_cfg, text=text_cfg)


def load_contrastive_loss_config(
    path: str | Path | None = None,
) -> ContrastiveLossConfig:
    raw = _load_model_config_raw(path)
    loss_raw = raw.get("loss", {})

    name = str(loss_raw.get("name", "symmetric_info_nce")).strip().lower()
    label_smoothing = float(loss_raw.get("label_smoothing", 0.0))
    reduction = str(loss_raw.get("reduction", "mean")).strip().lower()

    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError("loss.label_smoothing must satisfy 0.0 <= value < 1.0.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("loss.reduction must be one of: mean, sum, none.")

    return ContrastiveLossConfig(
        name=name,
        label_smoothing=label_smoothing,
        reduction=reduction,  # type: ignore[arg-type]
    )


# ============================================================================
# Training Config (`configs/training.yaml`)
# ============================================================================


TrainingSchedulerName = Literal["none", "cosine", "plateau"]
TrainingSchedulerStepOn = Literal["step", "epoch"]
TrainingMonitorMode = Literal["min", "max"]
TrainingLoggingBackend = Literal["none", "tensorboard"]
RetrievalDirection = Literal["text_to_image", "image_to_text"]
RetrievalSplit = Literal["val", "test"]
RetrievalIndexBackend = Literal["flat", "ivfpq", "ivfsq"]
RetrievalDistanceMetric = Literal["cosine", "ip", "l2"]
RetrievalQueryType = Literal["text", "image", "text_embedding", "image_embedding"]
RetrievalTargetModality = Literal["image", "text"]
RetrievalRerankerName = Literal["none", "blend", "mmr"]


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters."""

    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    name: TrainingSchedulerName
    step_on: TrainingSchedulerStepOn
    warmup_steps: int
    min_lr_ratio: float
    plateau_factor: float
    plateau_patience: int


@dataclass(frozen=True)
class RuntimeConfig:
    """Core training runtime settings."""

    seed: int
    deterministic: bool
    epochs: int
    use_amp: bool
    grad_accumulation_steps: int
    grad_clip_norm: float | None
    log_every_n_steps: int
    resume_from: str | None


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint saving policy."""

    enabled: bool
    save_last: bool
    save_best: bool
    save_every_n_epochs: int
    max_to_keep: int
    monitor: str
    mode: TrainingMonitorMode
    filename_prefix: str
    dir: str


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping criteria."""

    enabled: bool
    monitor: str
    mode: TrainingMonitorMode
    patience: int
    min_delta: float


@dataclass(frozen=True)
class ExperimentLoggingConfig:
    """Experiment tracking/logging settings."""

    backend: TrainingLoggingBackend
    run_name: str


@dataclass(frozen=True)
class TrainingConfig:
    """Typed root object for `configs/training.yaml`."""

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    runtime: RuntimeConfig
    checkpoint: CheckpointConfig
    early_stopping: EarlyStoppingConfig
    experiment_logging: ExperimentLoggingConfig


def _as_tuple2(values: object, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(values, (list, tuple)) and len(values) == 2:
        return (float(values[0]), float(values[1]))
    return default


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load and validate `configs/training.yaml`."""
    config_path = path or (CONFIGS_DIR / "training.yaml")
    raw = load_yaml(config_path)

    optimizer_raw = raw.get("optimizer", {})
    scheduler_raw = raw.get("scheduler", {})
    runtime_raw = raw.get("runtime", {})
    checkpoint_raw = raw.get("checkpoint", {})
    early_raw = raw.get("early_stopping", {})
    logging_raw = raw.get("experiment_logging", {})

    optimizer = OptimizerConfig(
        lr=float(optimizer_raw.get("lr", 5e-5)),
        weight_decay=float(optimizer_raw.get("weight_decay", 0.01)),
        betas=_as_tuple2(optimizer_raw.get("betas"), (0.9, 0.999)),
        eps=float(optimizer_raw.get("eps", 1e-8)),
    )
    scheduler = SchedulerConfig(
        name=str(scheduler_raw.get("name", "cosine")).strip().lower(),  # type: ignore[arg-type]
        step_on=str(scheduler_raw.get("step_on", "step")).strip().lower(),  # type: ignore[arg-type]
        warmup_steps=int(scheduler_raw.get("warmup_steps", 0)),
        min_lr_ratio=float(scheduler_raw.get("min_lr_ratio", 0.1)),
        plateau_factor=float(scheduler_raw.get("plateau_factor", 0.5)),
        plateau_patience=int(scheduler_raw.get("plateau_patience", 2)),
    )
    runtime = RuntimeConfig(
        seed=int(runtime_raw.get("seed", 42)),
        deterministic=bool(runtime_raw.get("deterministic", True)),
        epochs=int(runtime_raw.get("epochs", 10)),
        use_amp=bool(runtime_raw.get("use_amp", True)),
        grad_accumulation_steps=int(runtime_raw.get("grad_accumulation_steps", 1)),
        grad_clip_norm=(
            float(runtime_raw["grad_clip_norm"])
            if runtime_raw.get("grad_clip_norm") is not None
            else None
        ),
        log_every_n_steps=int(runtime_raw.get("log_every_n_steps", 20)),
        resume_from=(
            str(runtime_raw["resume_from"]) if runtime_raw.get("resume_from") else None
        ),
    )
    checkpoint = CheckpointConfig(
        enabled=bool(checkpoint_raw.get("enabled", True)),
        save_last=bool(checkpoint_raw.get("save_last", True)),
        save_best=bool(checkpoint_raw.get("save_best", True)),
        save_every_n_epochs=int(checkpoint_raw.get("save_every_n_epochs", 1)),
        max_to_keep=int(checkpoint_raw.get("max_to_keep", 3)),
        monitor=str(checkpoint_raw.get("monitor", "val/loss")),
        mode=str(checkpoint_raw.get("mode", "min")).strip().lower(),  # type: ignore[arg-type]
        filename_prefix=str(checkpoint_raw.get("filename_prefix", "clip")),
        dir=str(checkpoint_raw.get("dir", "artifacts/checkpoints")),
    )
    early_stopping = EarlyStoppingConfig(
        enabled=bool(early_raw.get("enabled", False)),
        monitor=str(early_raw.get("monitor", "val/loss")),
        mode=str(early_raw.get("mode", "min")).strip().lower(),  # type: ignore[arg-type]
        patience=int(early_raw.get("patience", 5)),
        min_delta=float(early_raw.get("min_delta", 0.0)),
    )
    experiment_logging = ExperimentLoggingConfig(
        backend=str(logging_raw.get("backend", "none")).strip().lower(),  # type: ignore[arg-type]
        run_name=str(logging_raw.get("run_name", "default")),
    )

    if optimizer.lr <= 0:
        raise ValueError("optimizer.lr must be > 0.")
    if optimizer.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be >= 0.")
    if runtime.epochs <= 0:
        raise ValueError("runtime.epochs must be > 0.")
    if runtime.grad_accumulation_steps <= 0:
        raise ValueError("runtime.grad_accumulation_steps must be > 0.")
    if scheduler.name not in {"none", "cosine", "plateau"}:
        raise ValueError("scheduler.name must be one of: none, cosine, plateau.")
    if scheduler.step_on not in {"step", "epoch"}:
        raise ValueError("scheduler.step_on must be one of: step, epoch.")
    if checkpoint.mode not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be one of: min, max.")
    if early_stopping.mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be one of: min, max.")
    if experiment_logging.backend not in {"none", "tensorboard"}:
        raise ValueError(
            "experiment_logging.backend must be one of: none, tensorboard."
        )

    return TrainingConfig(
        optimizer=optimizer,
        scheduler=scheduler,
        runtime=runtime,
        checkpoint=checkpoint,
        early_stopping=early_stopping,
        experiment_logging=experiment_logging,
    )


# ============================================================================
# Retrieval Evaluation Config (`configs/retrieval.yaml` -> `runtime/checkpoint/output`)
# ============================================================================


@dataclass(frozen=True)
class RetrievalEvalRuntimeConfig:
    """Retrieval evaluation runtime options."""

    split: RetrievalSplit
    directions: list[RetrievalDirection]
    ks: list[int]
    similarity_chunk_size: int
    use_amp: bool


@dataclass(frozen=True)
class RetrievalEvalCheckpointConfig:
    """Checkpoint loading options for evaluation."""

    path: Path | None
    strict: bool
    state_dict_key: str


@dataclass(frozen=True)
class RetrievalEvalOutputConfig:
    """Evaluation result serialization options."""

    save_json: bool
    output_dir: Path
    filename: str
    include_per_query: bool


@dataclass(frozen=True)
class RetrievalEvalConfig:
    """Typed retrieval evaluation config object."""

    runtime: RetrievalEvalRuntimeConfig
    checkpoint: RetrievalEvalCheckpointConfig
    output: RetrievalEvalOutputConfig


def load_retrieval_eval_config(path: str | Path | None = None) -> RetrievalEvalConfig:
    """Load and validate `configs/retrieval.yaml`."""
    config_path = path or (CONFIGS_DIR / "retrieval.yaml")
    raw = load_yaml(config_path)

    runtime_raw = raw.get("runtime", {})
    checkpoint_raw = raw.get("checkpoint", {})
    output_raw = raw.get("output", {})

    split = str(runtime_raw.get("split", "test")).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError("retrieval.runtime.split must be one of: val, test.")

    directions_raw = runtime_raw.get("directions", ["text_to_image", "image_to_text"])
    if not isinstance(directions_raw, list) or not directions_raw:
        raise ValueError("retrieval.runtime.directions must be a non-empty list.")
    directions: list[RetrievalDirection] = []
    for value in directions_raw:
        direction = str(value).strip().lower()
        if direction not in {"text_to_image", "image_to_text"}:
            raise ValueError(
                "retrieval.runtime.directions values must be one of: "
                "text_to_image, image_to_text."
            )
        directions.append(direction)  # type: ignore[arg-type]

    ks_raw = runtime_raw.get("ks", [1, 5, 10])
    if not isinstance(ks_raw, list) or not ks_raw:
        raise ValueError(
            "retrieval.runtime.ks must be a non-empty list of positive integers."
        )
    ks = sorted({int(k) for k in ks_raw})
    if any(k <= 0 for k in ks):
        raise ValueError("retrieval.runtime.ks values must be > 0.")

    chunk_size = int(runtime_raw.get("similarity_chunk_size", 1024))
    if chunk_size <= 0:
        raise ValueError("retrieval.runtime.similarity_chunk_size must be > 0.")

    ckpt_path_raw = checkpoint_raw.get("path")
    ckpt_path = _resolve_path(ckpt_path_raw) if ckpt_path_raw else None

    output_dir_raw = output_raw.get("output_dir", "artifacts/metadata")
    output_dir = _resolve_path(output_dir_raw)

    return RetrievalEvalConfig(
        runtime=RetrievalEvalRuntimeConfig(
            split=split,  # type: ignore[arg-type]
            directions=directions,
            ks=ks,
            similarity_chunk_size=chunk_size,
            use_amp=bool(runtime_raw.get("use_amp", True)),
        ),
        checkpoint=RetrievalEvalCheckpointConfig(
            path=ckpt_path,
            strict=bool(checkpoint_raw.get("strict", True)),
            state_dict_key=str(
                checkpoint_raw.get("state_dict_key", "model_state_dict")
            ),
        ),
        output=RetrievalEvalOutputConfig(
            save_json=bool(output_raw.get("save_json", True)),
            output_dir=output_dir,
            filename=str(output_raw.get("filename", "retrieval_eval.json")),
            include_per_query=bool(output_raw.get("include_per_query", True)),
        ),
    )


# ============================================================================
# Retrieval System Config (`configs/retrieval.yaml` -> embed/index/serving)
# ============================================================================


@dataclass(frozen=True)
class RetrievalEmbedConfig:
    """Embedding generation settings."""

    use_amp: bool
    normalize: bool
    image_batch_size: int
    text_batch_size: int


@dataclass(frozen=True)
class RetrievalIndexConfig:
    """Vector index build/search hyperparameters."""

    backend: RetrievalIndexBackend
    metric: RetrievalDistanceMetric
    nlist: int
    nprobe: int
    pq_m: int
    pq_nbits: int
    sq_qtype: str
    train_sample_size: int


@dataclass(frozen=True)
class RetrievalStorageConfig:
    """Artifacts storage paths and serialization settings."""

    embeddings_dir: Path
    index_dir: Path
    metadata_dir: Path
    use_mmap: bool
    embedding_dtype: str


@dataclass(frozen=True)
class RetrievalSearchConfig:
    """Search API defaults and limits."""

    default_top_k: int
    max_top_k: int
    return_metadata: bool


@dataclass(frozen=True)
class RetrievalRerankerConfig:
    """Second-stage reranking settings."""

    enabled: bool
    name: RetrievalRerankerName
    candidate_multiplier: int
    max_candidates: int
    blend_alpha: float
    mmr_lambda: float


@dataclass(frozen=True)
class RetrievalServingConfig:
    """Retrieval service host/port/route settings."""

    host: str
    port: int
    route: str


@dataclass(frozen=True)
class RetrievalConfig:
    """Typed root object for retrieval system settings."""

    embed: RetrievalEmbedConfig
    index: RetrievalIndexConfig
    storage: RetrievalStorageConfig
    search: RetrievalSearchConfig
    reranker: RetrievalRerankerConfig
    serving: RetrievalServingConfig


def load_retrieval_config(path: str | Path | None = None) -> RetrievalConfig:
    """Load and validate retrieval-system settings from `configs/retrieval.yaml`."""
    config_path = path or (CONFIGS_DIR / "retrieval.yaml")
    raw = load_yaml(config_path)

    embed_raw = raw.get("embed", {})
    index_raw = raw.get("index", {})
    storage_raw = raw.get("storage", {})
    search_raw = raw.get("search", {})
    reranker_raw = raw.get("reranker", {})
    serving_raw = raw.get("serving", {})

    backend = str(index_raw.get("backend", "flat")).strip().lower()
    metric = str(index_raw.get("metric", "cosine")).strip().lower()
    if backend not in {"flat", "ivfpq", "ivfsq"}:
        raise ValueError("retrieval.index.backend must be one of: flat, ivfpq, ivfsq.")
    if metric not in {"cosine", "ip", "l2"}:
        raise ValueError("retrieval.index.metric must be one of: cosine, ip, l2.")

    nlist = int(index_raw.get("nlist", 4096))
    nprobe = int(index_raw.get("nprobe", 32))
    pq_m = int(index_raw.get("pq_m", 64))
    pq_nbits = int(index_raw.get("pq_nbits", 8))
    train_sample_size = int(index_raw.get("train_sample_size", 200000))
    if nlist <= 0:
        raise ValueError("retrieval.index.nlist must be > 0.")
    if nprobe <= 0:
        raise ValueError("retrieval.index.nprobe must be > 0.")
    if pq_m <= 0:
        raise ValueError("retrieval.index.pq_m must be > 0.")
    if pq_nbits <= 0:
        raise ValueError("retrieval.index.pq_nbits must be > 0.")
    if train_sample_size <= 0:
        raise ValueError("retrieval.index.train_sample_size must be > 0.")

    image_batch_size = int(embed_raw.get("image_batch_size", 256))
    text_batch_size = int(embed_raw.get("text_batch_size", 512))
    if image_batch_size <= 0 or text_batch_size <= 0:
        raise ValueError("retrieval.embed image/text batch sizes must be > 0.")

    default_top_k = int(search_raw.get("default_top_k", 10))
    max_top_k = int(search_raw.get("max_top_k", 100))
    if default_top_k <= 0:
        raise ValueError("retrieval.search.default_top_k must be > 0.")
    if max_top_k <= 0:
        raise ValueError("retrieval.search.max_top_k must be > 0.")
    if default_top_k > max_top_k:
        raise ValueError(
            "retrieval.search.default_top_k must be <= retrieval.search.max_top_k."
        )

    reranker_name = str(reranker_raw.get("name", "none")).strip().lower()
    if reranker_name not in {"none", "blend", "mmr"}:
        raise ValueError("retrieval.reranker.name must be one of: none, blend, mmr.")
    reranker_candidate_multiplier = int(reranker_raw.get("candidate_multiplier", 5))
    reranker_max_candidates = int(reranker_raw.get("max_candidates", 200))
    reranker_blend_alpha = float(reranker_raw.get("blend_alpha", 0.5))
    reranker_mmr_lambda = float(reranker_raw.get("mmr_lambda", 0.7))
    if reranker_candidate_multiplier <= 0:
        raise ValueError("retrieval.reranker.candidate_multiplier must be > 0.")
    if reranker_max_candidates <= 0:
        raise ValueError("retrieval.reranker.max_candidates must be > 0.")
    if not 0.0 <= reranker_blend_alpha <= 1.0:
        raise ValueError("retrieval.reranker.blend_alpha must be in [0, 1].")
    if not 0.0 <= reranker_mmr_lambda <= 1.0:
        raise ValueError("retrieval.reranker.mmr_lambda must be in [0, 1].")

    port = int(serving_raw.get("port", 8001))
    if port <= 0 or port > 65535:
        raise ValueError("retrieval.serving.port must be in range [1, 65535].")

    embeddings_dir = _resolve_path(
        storage_raw.get("embeddings_dir", "artifacts/embeddings")
    )
    index_dir = _resolve_path(storage_raw.get("index_dir", "artifacts/indices"))
    metadata_dir = _resolve_path(storage_raw.get("metadata_dir", "artifacts/metadata"))

    return RetrievalConfig(
        embed=RetrievalEmbedConfig(
            use_amp=bool(embed_raw.get("use_amp", True)),
            normalize=bool(embed_raw.get("normalize", True)),
            image_batch_size=image_batch_size,
            text_batch_size=text_batch_size,
        ),
        index=RetrievalIndexConfig(
            backend=backend,  # type: ignore[arg-type]
            metric=metric,  # type: ignore[arg-type]
            nlist=nlist,
            nprobe=nprobe,
            pq_m=pq_m,
            pq_nbits=pq_nbits,
            sq_qtype=str(index_raw.get("sq_qtype", "8bit")),
            train_sample_size=train_sample_size,
        ),
        storage=RetrievalStorageConfig(
            embeddings_dir=embeddings_dir,
            index_dir=index_dir,
            metadata_dir=metadata_dir,
            use_mmap=bool(storage_raw.get("use_mmap", True)),
            embedding_dtype=str(storage_raw.get("embedding_dtype", "float32")),
        ),
        search=RetrievalSearchConfig(
            default_top_k=default_top_k,
            max_top_k=max_top_k,
            return_metadata=bool(search_raw.get("return_metadata", True)),
        ),
        reranker=RetrievalRerankerConfig(
            enabled=bool(reranker_raw.get("enabled", False)),
            name=reranker_name,  # type: ignore[arg-type]
            candidate_multiplier=reranker_candidate_multiplier,
            max_candidates=reranker_max_candidates,
            blend_alpha=reranker_blend_alpha,
            mmr_lambda=reranker_mmr_lambda,
        ),
        serving=RetrievalServingConfig(
            host=str(serving_raw.get("host", "0.0.0.0")),
            port=port,
            route=str(serving_raw.get("route", "/api/search")),
        ),
    )


# ============================================================================
# API + Inference Config (`configs/api.yaml` + runtime helper)
# ============================================================================


@dataclass(frozen=True)
class APIAppConfig:
    """API metadata shown in docs/openapi."""

    title: str
    version: str


@dataclass(frozen=True)
class APIServerConfig:
    """API server network and docs endpoints."""

    host: str
    port: int
    root_path: str
    docs_url: str
    redoc_url: str


@dataclass(frozen=True)
class APIPathsConfig:
    """API runtime paths to dependency config files."""

    retrieval_config: Path
    model_config: Path
    data_config: Path


@dataclass(frozen=True)
class APIIndexConfig:
    """Default index names loaded by the API."""

    image_index_name: str
    text_index_name: str


@dataclass(frozen=True)
class APIConfig:
    """Typed root object for `configs/api.yaml`."""

    app: APIAppConfig
    server: APIServerConfig
    paths: APIPathsConfig
    index: APIIndexConfig


@dataclass(frozen=True)
class InferenceModelConfig:
    """Minimal inference-time model loading configuration."""

    model_config_path: Path
    data_config_path: Path
    checkpoint_path: Path | None = None
    checkpoint_strict: bool = True
    checkpoint_state_dict_key: str = "model_state_dict"


def load_api_config(path: str | Path | None = None) -> APIConfig:
    """Load and validate API settings from `configs/api.yaml`."""
    config_path = path or (CONFIGS_DIR / "api.yaml")
    raw = load_yaml(config_path)

    app_raw = raw.get("app", {})
    server_raw = raw.get("server", {})
    paths_raw = raw.get("paths", {})
    index_raw = raw.get("index", {})
    if not isinstance(app_raw, dict):
        raise TypeError("api.app must be a mapping/dict.")
    if not isinstance(server_raw, dict):
        raise TypeError("api.server must be a mapping/dict.")
    if not isinstance(paths_raw, dict):
        raise TypeError("api.paths must be a mapping/dict.")
    if not isinstance(index_raw, dict):
        raise TypeError("api.index must be a mapping/dict.")

    port = int(server_raw.get("port", 8001))
    if port <= 0 or port > 65535:
        raise ValueError("api.server.port must be in range [1, 65535].")

    docs_url = str(server_raw.get("docs_url", "/docs"))
    redoc_url = str(server_raw.get("redoc_url", "/redoc"))
    if docs_url and not docs_url.startswith("/"):
        raise ValueError("api.server.docs_url must start with '/'.")
    if redoc_url and not redoc_url.startswith("/"):
        raise ValueError("api.server.redoc_url must start with '/'.")

    return APIConfig(
        app=APIAppConfig(
            title=str(app_raw.get("title", "CLIP Retrieval API")),
            version=str(app_raw.get("version", "0.1.0")),
        ),
        server=APIServerConfig(
            host=str(server_raw.get("host", "0.0.0.0")),
            port=port,
            root_path=str(server_raw.get("root_path", "")),
            docs_url=docs_url,
            redoc_url=redoc_url,
        ),
        paths=APIPathsConfig(
            retrieval_config=_resolve_path(
                paths_raw.get("retrieval_config", "configs/retrieval.yaml")
            ),
            model_config=_resolve_path(
                paths_raw.get("model_config", "configs/model.yaml")
            ),
            data_config=_resolve_path(
                paths_raw.get("data_config", "configs/data.yaml")
            ),
        ),
        index=APIIndexConfig(
            image_index_name=str(index_raw.get("image_index_name", "image_index")),
            text_index_name=str(index_raw.get("text_index_name", "text_index")),
        ),
    )


def create_inference_model_config(
    *,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_strict: bool = True,
    checkpoint_state_dict_key: str = "model_state_dict",
) -> InferenceModelConfig:
    """Build typed inference runtime config with project-default config paths."""
    return InferenceModelConfig(
        model_config_path=_resolve_path(
            model_config_path or (CONFIGS_DIR / "model.yaml")
        ),
        data_config_path=_resolve_path(data_config_path or (CONFIGS_DIR / "data.yaml")),
        checkpoint_path=(
            _resolve_path(checkpoint_path) if checkpoint_path is not None else None
        ),
        checkpoint_strict=bool(checkpoint_strict),
        checkpoint_state_dict_key=str(checkpoint_state_dict_key),
    )
