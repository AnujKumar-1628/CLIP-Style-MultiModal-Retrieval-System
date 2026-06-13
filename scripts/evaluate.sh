#!/usr/bin/env bash
set -euo pipefail

# Retrieval evaluation pipeline launcher.
# Usage:
#   scripts/evaluate.sh [retrieval_config] [model_config] [data_config] [checkpoint_path]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RETRIEVAL_CONFIG="${1:-configs/retrieval.yaml}"
MODEL_CONFIG="${2:-configs/model.yaml}"
DATA_CONFIG="${3:-configs/data.yaml}"
CHECKPOINT_PATH="${4:-}"

echo "[eval] project_root=${PROJECT_ROOT}"
echo "[eval] retrieval_config=${RETRIEVAL_CONFIG}"
echo "[eval] model_config=${MODEL_CONFIG}"
echo "[eval] data_config=${DATA_CONFIG}"
echo "[eval] checkpoint_path=${CHECKPOINT_PATH:-<from retrieval config>}"

python -c "
from src.pipelines.eval_pipeline import run_eval_pipeline
result = run_eval_pipeline(
    retrieval_config_path='${RETRIEVAL_CONFIG}',
    model_config_path='${MODEL_CONFIG}',
    data_config_path='${DATA_CONFIG}',
    checkpoint_path='${CHECKPOINT_PATH}' or None,
)
print('[eval] status=', result.status)
print('[eval] run_id=', result.run_id)
print('[eval] duration_seconds=', result.duration_seconds)
print('[eval] metrics=', result.metrics)
print('[eval] artifacts=', result.artifacts)
if result.error:
    raise SystemExit(f\"[eval] failed: {result.error}\")
"
