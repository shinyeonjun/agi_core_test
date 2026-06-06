#!/usr/bin/env bash
set -euo pipefail

HOST_ADDRESS="${HOST_ADDRESS:-0.0.0.0}"
PORT="${PORT:-8765}"
DB="${HARNESS_DB:-data/harness.db}"
PROJECT_ROOT="${PROJECT_ROOT:-.}"

cd "$(dirname "$0")/.."
if [[ -f "venv/bin/activate" ]]; then
  . "venv/bin/activate"
fi

python -m neurokernel_seed.cli serve-core-api \
  --host "$HOST_ADDRESS" \
  --port "$PORT" \
  --db "$DB" \
  --project-root "$PROJECT_ROOT"
