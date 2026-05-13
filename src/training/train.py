"""Training entrypoint and component builders."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, ReduceLROnPlateau

from src.data_logic.datamodule import CLIPDataModule, create_datamodule
from src.loss.contrastive_loss import create_contrastive_loss
from src.models.clip_model import CLIPModel, create_clip_model
from src.training.backbone_unfreeze_callback import build_backbone_unfreeze_callback
from src.training.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from src.training.trainer import (
    NoOpExperimentLogger,
    TensorBoardExperimentLogger,
    Trainer,
)
from src.utils.config import (
    ExperimentLoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingSchedulerStepOn,
    load_training_config,
)
from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINTS_DIR, CONFIGS_DIR, RUNS_DIR, ensure_base_dirs
from src.utils.seed import set_seed

LOGGER = setup_logger(
    name="train_entrypoint",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _exclude_from_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    if param.ndim <= 1:
        return True
    if name.endswith(".bias"):
        return True
    if "norm" in lowered or "bn" in lowered:
        return True
    if lowered.endswith("logit_scale_log") or lowered == "logit_scale_log":
        return True
    return False


def build_optimizer(
    model: CLIPModel,
    cfg: OptimizerConfig,
) -> Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if _exclude_from_weight_decay(name, param):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": cfg.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    LOGGER.info(
        "Optimizer param groups | decay=%d tensors | no_decay=%d tensors",
        len(decay_names),
        len(no_decay_names),
    )
    return AdamW(
        param_groups,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )


def _compute_total_optimizer_steps(
    *,
    dataloader_len: int,
    epochs: int,
    grad_accumulation_steps: int,
) -> int:
    per_epoch_steps = math.ceil(dataloader_len / grad_accumulation_steps)
    return max(1, per_epoch_steps * epochs)


def build_scheduler(
    *,
    optimizer: Optimizer,
    cfg: SchedulerConfig,
    total_optimizer_steps: int,
    total_training_epochs: int,
) -> tuple[LRScheduler | ReduceLROnPlateau | None, TrainingSchedulerStepOn]:
    if cfg.name == "none":
        return None, cfg.step_on

    if cfg.name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
        )
        return scheduler, "epoch"

    schedule_horizon = (
        max(1, total_optimizer_steps)
        if cfg.step_on == "step"
        else max(1, total_training_epochs)
    )
    warmup = max(0, cfg.warmup_steps)
    min_ratio = max(0.0, min(1.0, cfg.min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)

        progress_denom = max(1, schedule_horizon - warmup)
        progress = float(step - warmup) / float(progress_denom)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda), cfg.step_on


def _build_experiment_logger(cfg: ExperimentLoggingConfig):
    if cfg.backend == "tensorboard":
        log_dir = RUNS_DIR / cfg.run_name
        return TensorBoardExperimentLogger(log_dir=log_dir)
    return NoOpExperimentLogger()


def _build_callbacks(
    cfg: TrainingConfig,
    *,
    run_name: str,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> list:
    callbacks: list = []
    callbacks.insert(
        0,
        build_backbone_unfreeze_callback(
            unfreeze_at_epoch=4,
            backbone_lr=1e-5,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
        ),
    )

    if cfg.checkpoint.enabled:
        if cfg.checkpoint.dir:
            ckpt_dir = Path(cfg.checkpoint.dir)
            if not ckpt_dir.is_absolute():
                ckpt_dir = Path.cwd() / ckpt_dir
        else:
            ckpt_dir = CHECKPOINTS_DIR / run_name

        callbacks.append(
            ModelCheckpointCallback(
                save_dir=ckpt_dir,
                monitor=cfg.checkpoint.monitor,
                mode=cfg.checkpoint.mode,
                save_last=cfg.checkpoint.save_last,
                save_best=cfg.checkpoint.save_best,
                save_every_n_epochs=cfg.checkpoint.save_every_n_epochs,
                max_to_keep=cfg.checkpoint.max_to_keep,
                filename_prefix=cfg.checkpoint.filename_prefix,
            )
        )

    if cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                monitor=cfg.early_stopping.monitor,
                mode=cfg.early_stopping.mode,
                patience=cfg.early_stopping.patience,
                min_delta=cfg.early_stopping.min_delta,
                enabled=True,
            )
        )
    return callbacks


def build_training_components(
    *,
    training_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> tuple[TrainingConfig, CLIPDataModule, CLIPModel, torch.nn.Module, Optimizer]:
    cfg = load_training_config(training_config_path)
    set_seed(cfg.runtime.seed, deterministic=cfg.runtime.deterministic)

    datamodule = create_datamodule(config_path=data_config_path)
    datamodule.prepare_data()
    datamodule.setup("fit")

    model = create_clip_model(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    loss_fn = create_contrastive_loss(config_path=model_config_path)
    optimizer = build_optimizer(model, cfg.optimizer)
    return cfg, datamodule, model, loss_fn, optimizer


def run_training(
    *,
    training_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> list[dict[str, float]]:
    ensure_base_dirs()
    cfg, datamodule, model, loss_fn, optimizer = build_training_components(
        training_config_path=training_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    total_steps = _compute_total_optimizer_steps(
        dataloader_len=len(train_loader),
        epochs=cfg.runtime.epochs,
        grad_accumulation_steps=cfg.runtime.grad_accumulation_steps,
    )
    scheduler, scheduler_step_on = build_scheduler(
        optimizer=optimizer,
        cfg=cfg.scheduler,
        total_optimizer_steps=total_steps,
        total_training_epochs=cfg.runtime.epochs,
    )

    run_name = cfg.experiment_logging.run_name
    experiment_logger = _build_experiment_logger(cfg.experiment_logging)
    callbacks = _build_callbacks(
        cfg,
        run_name=run_name,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        use_amp=cfg.runtime.use_amp,
        grad_clip_norm=cfg.runtime.grad_clip_norm,
        grad_accumulation_steps=cfg.runtime.grad_accumulation_steps,
        log_every_n_steps=cfg.runtime.log_every_n_steps,
        scheduler_step_on=scheduler_step_on,
        callbacks=callbacks,
        experiment_logger=experiment_logger,
    )

    start_epoch = 1
    if cfg.runtime.resume_from:
        start_epoch = trainer.load_checkpoint(cfg.runtime.resume_from)
    remaining_epochs = max(0, cfg.runtime.epochs - (start_epoch - 1))
    if remaining_epochs == 0:
        LOGGER.info(
            "Resume checkpoint is already at/after target epochs (%d). Nothing to run.",
            cfg.runtime.epochs,
        )
        return trainer.history

    LOGGER.info(
        "Starting training | epochs=%d | amp=%s | scheduler=%s(%s) | total_steps=%d",
        remaining_epochs,
        cfg.runtime.use_amp,
        cfg.scheduler.name,
        scheduler_step_on,
        total_steps,
    )
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=remaining_epochs,
        start_epoch=start_epoch,
    )
    return history
