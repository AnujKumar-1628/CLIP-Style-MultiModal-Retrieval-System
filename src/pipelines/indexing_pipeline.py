"""Indexing pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipelines.base import BasePipeline, PipelineResult, pipeline_step
from src.retrieval.pipeline import run_retrieval_build_pipeline
from src.utils.config import load_retrieval_config
from src.utils.logger import setup_logger
from src.utils.paths import CONFIGS_DIR

LOGGER = setup_logger(
    name="indexing_pipeline",
    level="INFO",
    use_console=True,
    use_file=True,
)


class IndexingPipeline(BasePipeline):
    """Pipeline wrapper around embedding + index build orchestration."""

    def __init__(
        self,
        *,
        retrieval_config_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        data_config_path: str | Path | None = None,
        split: str = "test",
    ) -> None:
        self.retrieval_config_path = retrieval_config_path or (CONFIGS_DIR / "retrieval.yaml")
        self.model_config_path = model_config_path or (CONFIGS_DIR / "model.yaml")
        self.data_config_path = data_config_path or (CONFIGS_DIR / "data.yaml")
        self.split = split
        super().__init__(
            name="indexing_pipeline",
            config_paths={
                "retrieval": self.retrieval_config_path,
                "model": self.model_config_path,
                "data": self.data_config_path,
            },
        )

    @pipeline_step("indexing")
    def execute(self, **kwargs) -> dict[str, Any]:
        summary = run_retrieval_build_pipeline(
            retrieval_config_path=self.retrieval_config_path,
            model_config_path=self.model_config_path,
            data_config_path=self.data_config_path,
            split=self.split,
        )
        cfg = load_retrieval_config(self.retrieval_config_path)
        artifacts = {
            "embeddings_dir": str(cfg.storage.embeddings_dir),
            "index_dir": str(cfg.storage.index_dir),
            "metadata_dir": str(cfg.storage.metadata_dir),
        }
        metrics = {
            "image_index_size": float(summary.get("image_index_size", 0)),
            "text_index_size": float(summary.get("text_index_size", 0)),
        }
        metadata = {
            "image_backend": summary.get("image_index_backend"),
            "text_backend": summary.get("text_index_backend"),
            "split": self.split,
        }
        return {"metrics": metrics, "artifacts": artifacts, "metadata": metadata}


def run_indexing_pipeline(
    *,
    retrieval_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    data_config_path: str | Path | None = None,
    split: str = "test",
) -> PipelineResult:
    pipeline = IndexingPipeline(
        retrieval_config_path=retrieval_config_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        split=split,
    )
    return pipeline.run()


__all__ = [
    "IndexingPipeline",
    "run_indexing_pipeline",
]
