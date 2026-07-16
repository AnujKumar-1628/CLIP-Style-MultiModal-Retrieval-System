#!/usr/bin/env sh
set -eu

if [ "$#" -eq 0 ]; then
  set -- api
fi

case "$1" in
  api)
    shift
    exec python -m uvicorn api.main:app \
      --host "${API_HOST:-0.0.0.0}" \
      --port "${API_PORT:-8001}" \
      "$@"
    ;;
  train)
    shift
    exec bash scripts/train.sh "$@"
    ;;
  index)
    shift
    exec bash scripts/build_index.sh "$@"
    ;;
  evaluate)
    shift
    exec bash scripts/evaluate.sh "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
