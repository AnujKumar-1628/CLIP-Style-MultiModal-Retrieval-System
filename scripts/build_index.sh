#!/usr/bin/env bash
set -euo pipefail

# Retrieval indexing pipeline launcher.
# Usage:
#   scripts/build_index.sh [retrieval_config] [model_config] [data_config] [split]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RETRIEVAL_CONFIG="${1:-configs/retrieval.yaml}"
MODEL_CONFIG="${2:-configs/model.yaml}"
DATA_CONFIG="${3:-configs/data.yaml}"
SPLIT="${4:-test}"

echo "[index] project_root=${PROJECT_ROOT}"
echo "[index] retrieval_config=${RETRIEVAL_CONFIG}"
echo "[index] model_config=${MODEL_CONFIG}"
echo "[index] data_config=${DATA_CONFIG}"
echo "[index] split=${SPLIT}"

python -c "
from src.pipelines.indexing_pipeline import run_indexing_pipeline
result = run_indexing_pipeline(
    retrieval_config_path='${RETRIEVAL_CONFIG}',
    model_config_path='${MODEL_CONFIG}',
    data_config_path='${DATA_CONFIG}',
    split='${SPLIT}',
)
print('[index] status=', result.status)
print('[index] run_id=', result.run_id)
print('[index] duration_seconds=', result.duration_seconds)
print('[index] metrics=', result.metrics)
print('[index] artifacts=', result.artifacts)
if result.error:
    raise SystemExit(f\"[index] failed: {result.error}\")
"
