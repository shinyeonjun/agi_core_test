#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [[ -f "venv/bin/activate" ]]; then
  . "venv/bin/activate"
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required. Put it in ${ENV_FILE} or export it before running." >&2
    exit 2
  fi
}

require_defined() {
  local name="$1"
  if [[ ! -v "$name" ]]; then
    echo "${name} is required. Put it in ${ENV_FILE} or export it before running." >&2
    exit 2
  fi
}

require_env HOST_ADDRESS
require_env PORT
require_env HARNESS_DB
require_env PROJECT_ROOT
require_env NEUROKERNEL_CORE_URL
require_env DISCORD_CHANNEL_ID
require_env DISCORD_BOT_TOKEN
require_env DISCORD_ALLOWED_USER_IDS
require_env DISCORD_ALLOW_DMS
require_defined NEUROKERNEL_BOT_PREFIX
require_env NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX
require_env NEUROKERNEL_BOT_AUTO_DO_LOW_RISK
require_env LOG_DIR
require_env NEUROKERNEL_ACTION_REGISTRY
require_env NEUROKERNEL_WORKER_ENABLED
require_env NEUROKERNEL_WORKER_QUEUES
require_env NEUROKERNEL_WORKER_BLOCK_MS
require_env NEUROKERNEL_REDIS_URL
require_env NEUROKERNEL_SELF_PATCH_RUN_ROOT
require_env NEUROKERNEL_SELF_PATCH_TEST_COMMAND
require_env NEUROKERNEL_SELF_PATCH_CODEX_TIMEOUT
require_env NEUROKERNEL_SELF_PATCH_TEST_TIMEOUT
require_env NEUROKERNEL_SELF_PATCH_ISOLATION
require_env NEUROKERNEL_ACTIVATION_TEST_COMMAND
require_env NEUROKERNEL_ACTIVATION_TEST_TIMEOUT
require_defined NEUROKERNEL_ACTIVATION_RELOAD_COMMAND
require_env NEUROKERNEL_ACTIVATION_RELOAD_TIMEOUT
require_env NEUROKERNEL_ACTIVATION_REQUIRE_CLEAN_GIT
require_env NEUROKERNEL_ACTIVATION_COMMIT
require_env NEUROKERNEL_ACTIVATION_ARCHIVE_ROOT
require_env NEUROKERNEL_ACTIVATION_VERIFY
require_env NEUROKERNEL_ACTIVATION_SMOKE_TIMEOUT
require_env NEUROKERNEL_CODEX_BIN
require_env NEUROKERNEL_LANGUAGE_WORKSPACE
require_env NEUROKERNEL_LANGUAGE_SCHEMA_DIR
require_env NEUROKERNEL_LANGUAGE_TIMEOUT

mkdir -p "$LOG_DIR" data

core_pid=""
bot_pid=""
worker_pid=""
improvement_pid=""

stop_children() {
  if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
    kill "$worker_pid" 2>/dev/null || true
  fi
  if [[ -n "$improvement_pid" ]] && kill -0 "$improvement_pid" 2>/dev/null; then
    kill "$improvement_pid" 2>/dev/null || true
  fi
  if [[ -n "$bot_pid" ]] && kill -0 "$bot_pid" 2>/dev/null; then
    kill "$bot_pid" 2>/dev/null || true
  fi
  if [[ -n "$core_pid" ]] && kill -0 "$core_pid" 2>/dev/null; then
    kill "$core_pid" 2>/dev/null || true
  fi
}
trap stop_children EXIT INT TERM

echo "[stack] starting Core API on ${HOST_ADDRESS}:${PORT}"
python -m neurokernel_seed.cli serve-core-api \
  --host "$HOST_ADDRESS" \
  --port "$PORT" \
  --db "$HARNESS_DB" \
  --project-root "$PROJECT_ROOT" \
  >> "$LOG_DIR/core_api.log" 2>&1 &
core_pid="$!"

echo "[stack] waiting for Core API health"
python - <<PY
import sys
import time
import urllib.request

url = "${NEUROKERNEL_CORE_URL%/}/health"
for _ in range(60):
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            if response.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(1)
print(f"Core API did not become healthy: {url}", file=sys.stderr)
sys.exit(1)
PY

echo "[stack] starting Discord bot for channel ${DISCORD_CHANNEL_ID}"
python -m neurokernel_seed.cli serve-discord-bot \
  --core-url "$NEUROKERNEL_CORE_URL" \
  --channel-id "$DISCORD_CHANNEL_ID" \
  --prefix "$NEUROKERNEL_BOT_PREFIX" \
  >> "$LOG_DIR/discord_bot.log" 2>&1 &
bot_pid="$!"

if [[ "$NEUROKERNEL_WORKER_ENABLED" =~ ^(1|true|yes|y)$ ]]; then
  echo "[stack] starting Work Worker for queues ${NEUROKERNEL_WORKER_QUEUES}"
  # shellcheck disable=SC2086
  python -m neurokernel_seed.cli serve-work-worker \
    --db "$HARNESS_DB" \
    --project-root "$PROJECT_ROOT" \
    --block-ms "$NEUROKERNEL_WORKER_BLOCK_MS" \
    --queues $NEUROKERNEL_WORKER_QUEUES \
    >> "$LOG_DIR/work_worker.log" 2>&1 &
  worker_pid="$!"
fi

if [[ "${NEUROKERNEL_IMPROVEMENT_ENABLED:-false}" =~ ^(1|true|yes|y)$ ]]; then
  echo "[stack] starting Improvement Watchdog"
  python -m neurokernel_seed.cli serve-improvement-watchdog \
    --db "$HARNESS_DB" \
    --project-root "$PROJECT_ROOT" \
    --interval-seconds "${NEUROKERNEL_IMPROVEMENT_INTERVAL_SECONDS:-120}" \
    --min-gap-count "${NEUROKERNEL_IMPROVEMENT_MIN_GAP_COUNT:-2}" \
    --lookback "${NEUROKERNEL_IMPROVEMENT_LOOKBACK:-200}" \
    >> "$LOG_DIR/improvement_watchdog.log" 2>&1 &
  improvement_pid="$!"
fi

echo "[stack] running. core_pid=${core_pid} bot_pid=${bot_pid} worker_pid=${worker_pid:-disabled} improvement_pid=${improvement_pid:-disabled}"
set +e
if [[ -n "$worker_pid" ]] && [[ -n "$improvement_pid" ]]; then
  wait -n "$core_pid" "$bot_pid" "$worker_pid" "$improvement_pid"
elif [[ -n "$worker_pid" ]]; then
  wait -n "$core_pid" "$bot_pid" "$worker_pid"
elif [[ -n "$improvement_pid" ]]; then
  wait -n "$core_pid" "$bot_pid" "$improvement_pid"
else
  wait -n "$core_pid" "$bot_pid"
fi
exit_code="$?"
set -e
echo "[stack] one child exited with code ${exit_code}; shutting down stack"
exit "$exit_code"
