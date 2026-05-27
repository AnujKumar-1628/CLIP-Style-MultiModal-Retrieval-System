"""Shared abstractions for project pipelines."""

from __future__ import annotations

import subprocess
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from src.utils.config import load_yaml
from src.utils.logger import setup_logger
from src.utils.paths import PROJECT_ROOT

LOGGER = setup_logger(
    name="pipeline_base",
    level="INFO",
    use_console=True,
    use_file=True,
)

PipelineStatus = Literal["success", "failed"]


class PipelineError(RuntimeError):
    """Standardized pipeline exception with stage context."""

    def __init__(
        self,
        *,
        pipeline_name: str,
        stage: str,
        message: str,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.pipeline_name = pipeline_name
        self.stage = stage
        self.cause = cause


@dataclass(frozen=True)
class PipelineResult:
    pipeline_name: str
    status: PipelineStatus
    run_id: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    git_commit: str | None
    config_snapshot: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _duration_seconds(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _capture_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _snapshot_configs(config_paths: dict[str, str | Path | None]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, path_value in config_paths.items():
        if path_value is None:
            continue
        path = Path(path_value)
        try:
            snapshot[key] = {
                "path": str(path),
                "content": load_yaml(path),
            }
        except Exception as exc:
            snapshot[key] = {
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
    return snapshot


def pipeline_step(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that upgrades arbitrary exceptions to PipelineError."""

    def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def _wrapped(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(
                    pipeline_name=getattr(self, "name", "pipeline"),
                    stage=stage,
                    message=f"Stage '{stage}' failed: {exc}",
                    cause=exc,
                ) from exc

        return _wrapped

    return _decorate


class BasePipeline(ABC):
    """Abstract pipeline with standardized result contract."""

    def __init__(
        self,
        *,
        name: str,
        config_paths: dict[str, str | Path | None] | None = None,
    ) -> None:
        self.name = name
        self.config_paths = config_paths or {}

    @abstractmethod
    def execute(self, **kwargs) -> dict[str, Any] | None:
        """Execute pipeline-specific work and return structured payload."""

    def run(self, **kwargs) -> PipelineResult:
        run_id = str(uuid.uuid4())
        started = _utc_now()
        git_commit = _capture_git_commit()
        config_snapshot = _snapshot_configs(self.config_paths)

        try:
            payload = self.execute(**kwargs) or {}
            finished = _utc_now()
            metrics = {
                str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()
            }
            artifacts = _json_safe(dict(payload.get("artifacts", {})))
            warnings = [str(v) for v in payload.get("warnings", [])]
            metadata = _json_safe(dict(payload.get("metadata", {})))
            return PipelineResult(
                pipeline_name=self.name,
                status="success",
                run_id=run_id,
                started_at_utc=_to_iso(started),
                finished_at_utc=_to_iso(finished),
                duration_seconds=_duration_seconds(started, finished),
                git_commit=git_commit,
                config_snapshot=config_snapshot,
                metrics=metrics,
                artifacts=artifacts,
                warnings=warnings,
                metadata=metadata,
                error=None,
            )
        except PipelineError as exc:
            finished = _utc_now()
            LOGGER.error(
                "Pipeline '%s' failed at stage '%s': %s",
                exc.pipeline_name,
                exc.stage,
                exc,
            )
            return PipelineResult(
                pipeline_name=self.name,
                status="failed",
                run_id=run_id,
                started_at_utc=_to_iso(started),
                finished_at_utc=_to_iso(finished),
                duration_seconds=_duration_seconds(started, finished),
                git_commit=git_commit,
                config_snapshot=config_snapshot,
                error={
                    "type": type(exc).__name__,
                    "stage": exc.stage,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception as exc:
            finished = _utc_now()
            LOGGER.error("Pipeline '%s' crashed: %s", self.name, exc)
            return PipelineResult(
                pipeline_name=self.name,
                status="failed",
                run_id=run_id,
                started_at_utc=_to_iso(started),
                finished_at_utc=_to_iso(finished),
                duration_seconds=_duration_seconds(started, finished),
                git_commit=git_commit,
                config_snapshot=config_snapshot,
                error={
                    "type": type(exc).__name__,
                    "stage": "unknown",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


__all__ = [
    "BasePipeline",
    "PipelineError",
    "PipelineResult",
    "pipeline_step",
]
