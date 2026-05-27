"""Training pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipelines.base import BasePipeline, PipelineError, PipelineResult, pipeline_step
from src.training.train import run_training
from src.utils.config import load_training_config
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR

LOGGER = setup_logger(
    name="train_pipeline",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _best_val_loss(history: list[dict[str, float]]) -> float | None:
    val_losses = [float(item["val/loss"]) for item in history if "val/loss" in item]
    if not val_losses:
        return None
    return min(val_losses)


class TrainPipeline(BasePipeline):
    """Pipeline wrapper around `src.training.run_training`."""

    def __init__(
        self,
        *,
        training_config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
    ) -> None:
        self.training_config_path = training_config_path or (CONFIGS_DIR / "training.yaml")
        self.model_config_path = model_config_path or (CONFIGS_DIR / "model.yaml")
        self.data_config_path = data_config_path or (CONFIGS_DIR / "data.yaml")
        super().__init__(
            name="train_pipeline",
            config_paths={
                "training": self.training_config_path,
                "model": self.model_config_path,
                "data": self.data_config_path,
            },
        )

    @pipeline_step("train")
    def execute(self, **kwargs) -> dict[str, Any]:
        history = run_training(
            training_config_path=self.training_config_path,
            model_config_path=self.model_config_path,
            data_config_path=self.data_config_path,
        )
        if not isinstance(history, list):
            raise PipelineError(
                pipeline_name=self.name,
                stage="train",
                message="Training returned unexpected history format.",
            )

        cfg = load_training_config(self.training_config_path)
        last_metrics = history[-1] if history else {}
        metrics: dict[str, float] = {}
        for key in ("train/loss", "train/acc_i2t", "train/acc_t2i", "val/loss", "val/acc_i2t", "val/acc_t2i"):
            if key in last_metrics:
                metrics[key] = float(last_metrics[key])
        best = _best_val_loss(history)
        if best is not None:
            metrics["best/val_loss"] = float(best)

        artifacts = {
            "checkpoint_dir": cfg.checkpoint.dir,
            "epochs_completed": len(history),
        }
        metadata = {
            "history_rows": len(history),
            "resume_from": cfg.runtime.resume_from,
        }
        return {
            "metrics": metrics,
            "artifacts": artifacts,
            "metadata": metadata,
        }


def run_train_pipeline(
    *,
    training_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> PipelineResult:
    pipeline = TrainPipeline(
        training_config_path=training_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
    )
    return pipeline.run()


__all__ = [
    "TrainPipeline",
    "run_train_pipeline",
]
