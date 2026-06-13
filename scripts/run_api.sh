#!/usr/bin/env bash
set -euo pipefail

# API server launcher.
# Usage:
#   scripts/run_api.sh [host] [port]
#
# Defaults are read from configs/api.yaml via src.utils.config.load_api_config.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

API_HOST="${1:-}"
API_PORT="${2:-}"

readarray -t _API_DEFAULTS < <(python -c "
from src.utils.config import load_api_config
cfg = load_api_config()
print(cfg.server.host)
print(cfg.server.port)
")

DEFAULT_HOST="${_API_DEFAULTS[0]}"
DEFAULT_PORT="${_API_DEFAULTS[1]}"

HOST="${API_HOST:-${DEFAULT_HOST}}"
PORT="${API_PORT:-${DEFAULT_PORT}}"

if [[ -z "${API_HOST}" && "${HOST}" == "0.0.0.0" ]]; then
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            HOST="127.0.0.1"
            ;;
    esac
fi

echo "[api] project_root=${PROJECT_ROOT}"
echo "[api] host=${HOST}"
echo "[api] port=${PORT}"

python -m uvicorn api.main:app --host "${HOST}" --port "${PORT}"
