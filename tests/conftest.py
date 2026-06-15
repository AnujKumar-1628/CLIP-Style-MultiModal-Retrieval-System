"""Shared pytest fixtures for project test suite."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Write a YAML dictionary into a temporary file."""

    def _write(filename: str, payload: dict) -> Path:
        path = tmp_path / filename
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        return path

    return _write
