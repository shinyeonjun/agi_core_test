#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-neurokernel-stack.service}"
UNIT_NAME="${UNIT_NAME:-neurokernel-stack-restart}"
DELAY="${DELAY:-2s}"

if ! command -v systemd-run >/dev/null 2>&1; then
  echo "systemd-run is required for deferred service restart" >&2
  exit 2
fi

systemd-run --user \
  --unit "$UNIT_NAME" \
  --on-active "$DELAY" \
  systemctl --user restart "$SERVICE_NAME"
