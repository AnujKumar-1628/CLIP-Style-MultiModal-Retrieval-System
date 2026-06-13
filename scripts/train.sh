#!/usr/bin/env bash
set -euo pipefail

# Train pipeline launcher.
# Usage:
#   scripts/train.sh [training_config] [model_config] [data_config]

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TRAINING_CONFIG="${1:-configs/training.yaml}"
MODEL_CONFIG="${2:-configs/model.yaml}"
DATA_CONFIG="${3:-configs/data.yaml}"

echo "[train] project_root=${PROJECT_ROOT}"
echo "[train] training_config=${TRAINING_CONFIG}"
echo "[train] model_config=${MODEL_CONFIG}"
echo "[train] data_config=${DATA_CONFIG}"

python -c "
from src.pipelines.train_pipeline import run_train_pipeline
result = run_train_pipeline(
    training_config_path='${TRAINING_CONFIG}',
    model_config_path='${MODEL_CONFIG}',
    data_config_path='${DATA_CONFIG}',
)
print('[train] status=', result.status)
print('[train] run_id=', result.run_id)
print('[train] duration_seconds=', result.duration_seconds)
print('[train] metrics=', result.metrics)
print('[train] artifacts=', result.artifacts)
if result.error:
    raise SystemExit(f\"[train] failed: {result.error}\")
"
