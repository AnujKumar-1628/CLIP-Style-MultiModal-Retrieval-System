"""Project-wide logging utilities.

This module provides a structured, reusable way to create loggers with:
- consistent formatting
- optional console + file handlers
- safe reconfiguration without duplicate handlers
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

from src.utils.paths import LOGS_DIR, ensure_base_dirs


DEFAULT_LOGGER_NAME = "clip_retrieval"
DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _resolve_level(level: str | int) -> int:
    """Normalize a logging level provided as text or int."""
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    if normalized not in _LOG_LEVELS:
        valid = ", ".join(_LOG_LEVELS.keys())
        raise ValueError(f"Invalid log level '{level}'. Valid values: {valid}")
    return _LOG_LEVELS[normalized]


def _clear_handlers(logger: logging.Logger) -> None:
    """Detach and close all handlers from a logger."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _build_stream_handler(level: int) -> logging.Handler:
    """Create stdout/stderr stream handler with project format."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    return handler


def _build_file_handler(level: int, file_path: Path) -> logging.Handler:
    """Create rotating file handler."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT))
    return handler


def setup_logger(
    name: str = DEFAULT_LOGGER_NAME,
    *,
    level: str | int = "INFO",
    use_console: bool = True,
    use_file: bool = True,
    log_filename: str | None = None,
    log_dir: str | Path = LOGS_DIR,
    propagate: bool = False,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """Create and configure a logger for the project.

    Args:
        name: Logger name.
        level: Logging level (e.g. "INFO", "DEBUG", or numeric).
        use_console: Attach console stream handler.
        use_file: Attach rotating file handler.
        log_filename: Custom log filename. Defaults to "<name>.log".
        log_dir: Directory where file logs are written.
        propagate: Whether messages should bubble up to parent logger.
        force_reconfigure: If True, removes existing handlers and reconfigures.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    if not use_console and not use_file:
        raise ValueError("At least one handler must be enabled: use_console or use_file.")

    resolved_level = _resolve_level(level)
    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = propagate

    if logger.handlers and not force_reconfigure:
        return logger

    if force_reconfigure:
        _clear_handlers(logger)

    if use_console:
        logger.addHandler(_build_stream_handler(resolved_level))

    if use_file:
        ensure_base_dirs()
        target_dir = Path(log_dir)
        filename = log_filename or f"{name}.log"
        logger.addHandler(_build_file_handler(resolved_level, target_dir / filename))

    return logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return an existing logger by name without reconfiguration."""
    return logging.getLogger(name)


def set_external_log_levels(level_overrides: dict[str, str | int] | None = None) -> None:
    """Set log levels for noisy third-party libraries.

    Args:
        level_overrides: Optional mapping like
            ``{"urllib3": "WARNING", "transformers": "ERROR"}``.
            Defaults are used when omitted.
    """
    defaults = {
        "urllib3": "WARNING",
        "PIL": "WARNING",
        "matplotlib": "WARNING",
        "transformers": "WARNING",
        "huggingface_hub": "WARNING",
    }
    overrides = level_overrides or defaults
    for logger_name, level in overrides.items():
        logging.getLogger(logger_name).setLevel(_resolve_level(level))


def describe_logger(logger: logging.Logger) -> dict[str, object]:
    """Provide a quick snapshot of logger configuration for debugging."""
    handler_types: Iterable[str] = [type(handler).__name__ for handler in logger.handlers]
    return {
        "name": logger.name,
        "level": logging.getLevelName(logger.level),
        "propagate": logger.propagate,
        "handlers": list(handler_types),
    }


__all__ = [
    "DEFAULT_LOGGER_NAME",
    "setup_logger",
    "get_logger",
    "set_external_log_levels",
    "describe_logger",
]
