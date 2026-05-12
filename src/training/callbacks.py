"""Training callbacks for checkpointing and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.utils.logger import setup_logger
from src.utils.paths import CHECKPOINTS_DIR

LOGGER = setup_logger(
    name="training_callbacks",
    level="INFO",
    use_console=True,
    use_file=True,
)

ModeType = Literal["min", "max"]


class Callback:
    """Base callback with no-op hooks."""

    def on_train_start(self, trainer: Any) -> None:
        return None

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        return None

    def on_train_batch_end(
        self,
        trainer: Any,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        return None

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        return None

    def on_train_end(self, trainer: Any) -> None:
        return None


def _is_improved(current: float, best: float | None, *, mode: ModeType, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


@dataclass
class CheckpointRecord:
    path: Path
    epoch: int


class ModelCheckpointCallback(Callback):
    """Save last and/or best model checkpoints."""

    def __init__(
        self,
        *,
        save_dir: str | Path | None = None,
        monitor: str = "val/loss",
        mode: ModeType = "min",
        save_last: bool = True,
        save_best: bool = True,
        save_every_n_epochs: int = 1,
        max_to_keep: int = 3,
        filename_prefix: str = "clip",
        min_delta: float = 0.0,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if save_every_n_epochs <= 0:
            raise ValueError("save_every_n_epochs must be > 0.")
        if max_to_keep <= 0:
            raise ValueError("max_to_keep must be > 0.")

        self.save_dir = Path(save_dir) if save_dir is not None else CHECKPOINTS_DIR
        self.monitor = monitor
        self.mode = mode
        self.save_last = bool(save_last)
        self.save_best = bool(save_best)
        self.save_every_n_epochs = int(save_every_n_epochs)
        self.max_to_keep = int(max_to_keep)
        self.filename_prefix = filename_prefix
        self.min_delta = float(min_delta)

        self.best_score: float | None = None
        self._history: list[CheckpointRecord] = []

    def on_train_start(self, trainer: Any) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Checkpoint directory: %s", self.save_dir)

    def _save_checkpoint(
        self,
        *,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        tag: str,
        is_best: bool,
    ) -> Path:
        filename = f"{self.filename_prefix}_{tag}.pt"
        target = self.save_dir / filename
        state = trainer.state_dict_for_checkpoint(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_best=is_best,
        )
        torch.save(state, target)
        return target

    def _save_epoch_checkpoint(
        self,
        *,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        filename = f"{self.filename_prefix}_epoch_{epoch:04d}.pt"
        target = self.save_dir / filename
        state = trainer.state_dict_for_checkpoint(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            is_best=False,
        )
        torch.save(state, target)
        self._history.append(CheckpointRecord(path=target, epoch=epoch))

        while len(self._history) > self.max_to_keep:
            stale = self._history.pop(0)
            if stale.path.exists():
                stale.path.unlink()

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        if epoch % self.save_every_n_epochs == 0:
            self._save_epoch_checkpoint(
                trainer=trainer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        if self.save_last:
            self._save_checkpoint(
                trainer=trainer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                tag="last",
                is_best=False,
            )

        if not self.save_best:
            return
        if val_metrics is None or self.monitor not in val_metrics:
            return

        score = float(val_metrics[self.monitor])
        if _is_improved(score, self.best_score, mode=self.mode, min_delta=self.min_delta):
            self.best_score = score
            path = self._save_checkpoint(
                trainer=trainer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                tag="best",
                is_best=True,
            )
            LOGGER.info(
                "Saved new best checkpoint at epoch=%d (%s=%.6f): %s",
                epoch,
                self.monitor,
                score,
                path,
            )


class EarlyStoppingCallback(Callback):
    """Stop training when monitored metric stops improving."""

    def __init__(
        self,
        *,
        monitor: str = "val/loss",
        mode: ModeType = "min",
        patience: int = 5,
        min_delta: float = 0.0,
        enabled: bool = True,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if patience < 0:
            raise ValueError("patience must be >= 0.")

        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.enabled = bool(enabled)

        self.best_score: float | None = None
        self.bad_epochs = 0

    def on_epoch_end(
        self,
        trainer: Any,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> None:
        if not self.enabled:
            return
        if val_metrics is None or self.monitor not in val_metrics:
            return

        score = float(val_metrics[self.monitor])
        if _is_improved(score, self.best_score, mode=self.mode, min_delta=self.min_delta):
            self.best_score = score
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            trainer.should_stop = True
            LOGGER.info(
                "Early stopping triggered at epoch=%d (%s=%.6f, patience=%d).",
                epoch,
                self.monitor,
                score,
                self.patience,
            )


__all__ = [
    "Callback",
    "CheckpointRecord",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
]
