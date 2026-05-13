"""Core trainer for CLIP-style contrastive learning."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.training.callbacks import Callback
from src.utils.logger import setup_logger

LOGGER = setup_logger(
    name="trainer",
    level="INFO",
    use_console=True,
    use_file=True,
)


class ExperimentLogger(Protocol):
    """Minimal logger interface for training metrics."""

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None: ...

    def close(self) -> None: ...


class NoOpExperimentLogger:
    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        return None

    def close(self) -> None:
        return None


class TensorBoardExperimentLogger:
    """TensorBoard-backed experiment logger."""

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


@dataclass
class EpochAverages:
    loss: float
    acc_i2t: float
    acc_t2i: float
    temperature: float

    def to_dict(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}/loss": self.loss,
            f"{prefix}/acc_i2t": self.acc_i2t,
            f"{prefix}/acc_t2i": self.acc_t2i,
            f"{prefix}/temperature": self.temperature,
        }


class Trainer:
    """Simple, stable trainer for CLIP-style dual-encoder training."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | ReduceLROnPlateau | None = None,
        device: torch.device | str | None = None,
        use_amp: bool = True,
        grad_clip_norm: float | None = None,
        grad_accumulation_steps: int = 1,
        log_every_n_steps: int = 20,
        scheduler_step_on: str = "epoch",
        callbacks: list[Callback] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> None:
        if grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be > 0.")
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0.")
        if scheduler_step_on not in {"step", "epoch"}:
            raise ValueError("scheduler_step_on must be 'step' or 'epoch'.")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        if device is None:
            if hasattr(model, "device_obj"):
                self.device = model.device_obj
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = torch.device(device)

        self.use_amp = bool(use_amp)
        self.grad_clip_norm = grad_clip_norm
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.scheduler_step_on = scheduler_step_on

        self.callbacks = callbacks or []
        self.experiment_logger = experiment_logger or NoOpExperimentLogger()

        self._amp_enabled = self.use_amp and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)
        else:
            # Backward-compatible fallback for older torch versions.
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        self.should_stop = False
        self.global_step = 0
        self.current_epoch = 0
        self.history: list[dict[str, float]] = []

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    @property
    def _unwrapped_model(self) -> nn.Module:
        """Return the base model, bypassing DDP/DP wrappers if present."""
        return getattr(self.model, "module", self.model)

    def _prepare_text_tokens(
        self,
        batch: dict[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_bundle = batch.get("text_tokens")
        if isinstance(token_bundle, dict):
            input_ids = token_bundle["input_ids"]
            attention_mask = token_bundle.get("attention_mask")
        else:
            texts = batch.get("texts")
            if texts is None:
                raise ValueError("Batch must contain either `text_tokens` or `texts`.")
            if not hasattr(self._unwrapped_model, "text_encoder"):
                raise ValueError(
                    "Model missing `text_encoder`, cannot tokenize raw texts."
                )
            tokens = self._unwrapped_model.text_encoder.tokenize_texts(texts)  # type: ignore[attr-defined]
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask")

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a torch.Tensor when provided.")
        return input_ids, attention_mask

    def _compute_batch_accuracy(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> tuple[float, float]:
        target = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        pred_i2t = torch.argmax(logits_per_image, dim=1)
        pred_t2i = torch.argmax(logits_per_text, dim=1)
        acc_i2t = (pred_i2t == target).float().mean().item()
        acc_t2i = (pred_t2i == target).float().mean().item()
        return acc_i2t, acc_t2i

    def _current_temperature(self) -> float:
        if hasattr(self._unwrapped_model, "get_logit_scale"):
            scale = self._unwrapped_model.get_logit_scale()  # type: ignore[attr-defined]
            if isinstance(scale, torch.Tensor):
                scale_value = float(scale.detach().item())
                return 1.0 / max(scale_value, 1e-12)
        return float("nan")

    @staticmethod
    def _set_epoch_for_loader(loader: DataLoader, epoch: int) -> None:
        targets = (
            getattr(loader, "dataset", None),
            getattr(loader, "sampler", None),
            getattr(loader, "batch_sampler", None),
        )
        for target in targets:
            if target is not None and hasattr(target, "set_epoch"):
                target.set_epoch(epoch)

    def _clamp_logit_scale(self) -> None:
        if not hasattr(self._unwrapped_model, "logit_scale_log"):
            return
        if not hasattr(self._unwrapped_model, "config"):
            max_log = math.log(100.0)
            with torch.no_grad():
                self._unwrapped_model.logit_scale_log.clamp_(max=max_log)  # type: ignore[attr-defined]
            return

        max_scale = float(
            getattr(self._unwrapped_model.config, "logit_scale_clamp_max", 100.0)
        )
        min_scale = float(
            getattr(self._unwrapped_model.config, "logit_scale_clamp_min", 1e-3)
        )
        max_scale = max(max_scale, min_scale + 1e-12)
        with torch.no_grad():
            self._unwrapped_model.logit_scale_log.clamp_(  # type: ignore[attr-defined]
                min=math.log(min_scale),
                max=math.log(max_scale),
            )

    def _autocast_context(self):
        if not self._amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _finalize_scheduler_epoch(self, val_metrics: dict[str, float] | None) -> None:
        if self.scheduler is None or self.scheduler_step_on != "epoch":
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric = None
            if val_metrics is not None:
                metric = val_metrics.get("val/loss")
            if metric is None:
                return
            self.scheduler.step(metric)
            return
        self.scheduler.step()

    def _run_single_epoch(self, loader: DataLoader, *, training: bool) -> EpochAverages:
        if training:
            self.model.train()
        else:
            self.model.eval()

        sum_loss = 0.0
        sum_acc_i2t = 0.0
        sum_acc_t2i = 0.0
        total_samples = 0

        total_steps = len(loader)
        grad_ctx = torch.enable_grad() if training else torch.inference_mode()
        with grad_ctx:
            for step_idx, batch in enumerate(loader, start=1):
                images = batch["images"]
                if not isinstance(images, torch.Tensor):
                    raise TypeError("Batch key `images` must be a torch.Tensor.")
                input_ids, attention_mask = self._prepare_text_tokens(batch)

                image_batch = images.to(self.device, non_blocking=True)
                ids = input_ids.to(self.device, non_blocking=True)
                mask = (
                    attention_mask.to(self.device, non_blocking=True)
                    if attention_mask is not None
                    else None
                )

                with self._autocast_context():
                    outputs = self.model(
                        images=image_batch,
                        input_ids=ids,
                        attention_mask=mask,
                    )
                    loss = self.loss_fn(
                        logits_per_image=outputs["logits_per_image"],
                        logits_per_text=outputs["logits_per_text"],
                    )
                    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                        loss_scalar = loss.mean()
                    else:
                        loss_scalar = loss

                if training:
                    is_first_accum_step = (
                        step_idx % self.grad_accumulation_steps == 1
                    ) or self.grad_accumulation_steps == 1
                    if is_first_accum_step:
                        self.optimizer.zero_grad(set_to_none=True)

                    scaled = loss_scalar / self.grad_accumulation_steps
                    self.scaler.scale(scaled).backward()
                    should_step = (step_idx % self.grad_accumulation_steps == 0) or (
                        step_idx == total_steps
                    )
                    if should_step:
                        if self.grad_clip_norm is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip_norm
                            )

                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()

                        if scale_after >= scale_before:
                            self._clamp_logit_scale()
                            self.global_step += 1
                            if (
                                self.scheduler is not None
                                and self.scheduler_step_on == "step"
                            ):
                                self.scheduler.step()
                        else:
                            LOGGER.warning(
                                "Gradient overflow detected; skipping optimizer step (step_idx=%d).",
                                step_idx,
                            )

                batch_size = int(image_batch.shape[0])
                acc_i2t, acc_t2i = self._compute_batch_accuracy(
                    outputs["logits_per_image"].detach(),
                    outputs["logits_per_text"].detach(),
                )
                sum_loss += float(loss_scalar.detach().item()) * batch_size
                sum_acc_i2t += acc_i2t * batch_size
                sum_acc_t2i += acc_t2i * batch_size
                total_samples += batch_size

                if training and (step_idx % self.log_every_n_steps == 0):
                    live_metrics = {
                        "train/step_loss": float(loss_scalar.detach().item()),
                        "train/step_acc_i2t": acc_i2t,
                        "train/step_acc_t2i": acc_t2i,
                        "train/temperature": self._current_temperature(),
                    }
                    self.experiment_logger.log_metrics(
                        live_metrics,
                        step=self.global_step,
                    )
                    for callback in self.callbacks:
                        callback.on_train_batch_end(
                            self,
                            self.current_epoch,
                            step_idx,
                            live_metrics,
                        )

        denom = max(total_samples, 1)
        return EpochAverages(
            loss=sum_loss / denom,
            acc_i2t=sum_acc_i2t / denom,
            acc_t2i=sum_acc_t2i / denom,
            temperature=self._current_temperature(),
        )

    def train_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        metrics = self._run_single_epoch(loader, training=True).to_dict("train")
        return metrics

    def validate_one_epoch(self, loader: DataLoader) -> dict[str, float]:
        metrics = self._run_single_epoch(loader, training=False).to_dict("val")
        return metrics

    def fit(
        self,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int,
        start_epoch: int = 1,
    ) -> list[dict[str, float]]:
        for callback in self.callbacks:
            callback.on_train_start(self)

        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch
            self._set_epoch_for_loader(train_loader, epoch)
            if val_loader is not None:
                self._set_epoch_for_loader(val_loader, epoch)
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)

            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = (
                self.validate_one_epoch(val_loader) if val_loader is not None else None
            )

            merged = dict(train_metrics)
            if val_metrics is not None:
                merged.update(val_metrics)
            merged["epoch"] = float(epoch)
            self.history.append(merged)

            self.experiment_logger.log_metrics(merged, step=epoch)
            self._finalize_scheduler_epoch(val_metrics)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, train_metrics, val_metrics)

            LOGGER.info(
                "Epoch %d | train_loss=%.6f train_acc_i2t=%.4f train_acc_t2i=%.4f%s",
                epoch,
                train_metrics["train/loss"],
                train_metrics["train/acc_i2t"],
                train_metrics["train/acc_t2i"],
                (
                    f" | val_loss={val_metrics['val/loss']:.6f} "
                    f"val_acc_i2t={val_metrics['val/acc_i2t']:.4f} "
                    f"val_acc_t2i={val_metrics['val/acc_t2i']:.4f}"
                    if val_metrics is not None
                    else ""
                ),
            )

            if self.should_stop:
                break

        for callback in self.callbacks:
            callback.on_train_end(self)
        self.experiment_logger.close()
        return self.history

    def state_dict_for_checkpoint(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        is_best: bool,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
            "train_metrics": train_metrics,
            "is_best": is_best,
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        if val_metrics is not None:
            payload["val_metrics"] = val_metrics
        return payload

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> int:
        location = map_location or self.device
        payload = torch.load(checkpoint_path, map_location=location)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if "scaler_state_dict" in payload:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.global_step = int(payload.get("global_step", 0))

        history = payload.get("history")
        if isinstance(history, list):
            self.history = history

        epoch = int(payload.get("epoch", 0))
        LOGGER.info("Loaded checkpoint from %s (epoch=%d).", checkpoint_path, epoch)
        return epoch + 1


__all__ = [
    "ExperimentLogger",
    "NoOpExperimentLogger",
    "TensorBoardExperimentLogger",
    "Trainer",
]
