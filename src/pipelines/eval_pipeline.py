"""Evaluation pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.evaluate import evaluate_retrieval
from src.pipelines.base import BasePipeline, PipelineResult, pipeline_step
from src.utils.config import load_retrieval_eval_config
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR

LOGGER = setup_logger(
    name="eval_pipeline",
    level="INFO",
    use_console=True,
    use_file=True,
)


def _collect_eval_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics_block = payload.get("metrics", {})
    if not isinstance(metrics_block, dict):
        return metrics
    for direction in ("text_to_image", "image_to_text"):
        section = metrics_block.get(direction)
        if not isinstance(section, dict):
            continue
        if "mrr" in section:
            metrics[f"{direction}/mrr"] = float(section["mrr"])
        if "mean_rank" in section:
            metrics[f"{direction}/mean_rank"] = float(section["mean_rank"])
        if "median_rank" in section:
            metrics[f"{direction}/median_rank"] = float(section["median_rank"])
        recall = section.get("recall_at_k", {})
        if isinstance(recall, dict):
            for key, value in recall.items():
                metrics[f"{direction}/r@{key}"] = float(value)
    return metrics


class EvalPipeline(BasePipeline):
    """Pipeline wrapper around retrieval evaluation."""

    def __init__(
        self,
        *,
        retrieval_config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.retrieval_config_path = retrieval_config_path or (CONFIGS_DIR / "retrieval.yaml")
        self.model_config_path = model_config_path or (CONFIGS_DIR / "model.yaml")
        self.data_config_path = data_config_path or (CONFIGS_DIR / "data.yaml")
        self.checkpoint_path = checkpoint_path
        super().__init__(
            name="eval_pipeline",
            config_paths={
                "retrieval": self.retrieval_config_path,
                "model": self.model_config_path,
                "data": self.data_config_path,
            },
        )

    @pipeline_step("evaluate")
    def execute(self, **kwargs) -> dict[str, Any]:
        result = evaluate_retrieval(
            retrieval_config_path=self.retrieval_config_path,
            model_config_path=self.model_config_path,
            data_config_path=self.data_config_path,
            checkpoint_path=self.checkpoint_path,
        )
        report = result.to_report_dict()
        metrics = _collect_eval_metrics(report)
        eval_cfg = load_retrieval_eval_config(self.retrieval_config_path)
        artifacts = {
            "report_path": (
                str(eval_cfg.output.output_dir / eval_cfg.output.filename)
                if eval_cfg.output.save_json
                else None
            ),
        }
        return {
            "metrics": metrics,
            "artifacts": artifacts,
            "metadata": {"directions": eval_cfg.runtime.directions, "split": eval_cfg.runtime.split},
        }


def run_eval_pipeline(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> PipelineResult:
    pipeline = EvalPipeline(
        retrieval_config_path=retrieval_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        checkpoint_path=checkpoint_path,
    )
    return pipeline.run()


__all__ = [
    "EvalPipeline",
    "run_eval_pipeline",
]
