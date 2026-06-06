#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -f "venv/bin/activate" ]]; then
  . "venv/bin/activate"
fi

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing ${name}. Export it before starting the bot." >&2
    exit 2
  fi
}

require_defined() {
  local name="$1"
  if [[ ! -v "$name" ]]; then
    echo "Missing ${name}. Export it before starting the bot." >&2
    exit 2
  fi
}

require_env NEUROKERNEL_CORE_URL
require_env DISCORD_BOT_TOKEN
require_defined NEUROKERNEL_BOT_PREFIX
require_env DISCORD_CHANNEL_ID
require_env DISCORD_ALLOWED_USER_IDS
require_env DISCORD_ALLOW_DMS
require_env NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX
require_env NEUROKERNEL_BOT_AUTO_DO_LOW_RISK

python -m neurokernel_seed.cli serve-discord-bot \
  --core-url "$NEUROKERNEL_CORE_URL" \
  --token-env DISCORD_BOT_TOKEN \
  --prefix "$NEUROKERNEL_BOT_PREFIX" \
  --channel-id "$DISCORD_CHANNEL_ID"
