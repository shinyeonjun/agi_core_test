#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVICE_NAME="${SERVICE_NAME:-neurokernel-stack.service}"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "$SYSTEMD_USER_DIR"
cp "deploy/${SERVICE_NAME}" "${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
chmod 644 "${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
chmod +x tools/run_orangepi_stack.sh tools/run_core_api.sh tools/run_discord_bot.sh

systemctl --user daemon-reload
systemctl --user enable "$SERVICE_NAME"

echo "Installed ${SERVICE_NAME}."
echo "Start:   systemctl --user start ${SERVICE_NAME}"
echo "Status:  systemctl --user status ${SERVICE_NAME} --no-pager"
echo "Logs:    journalctl --user -u ${SERVICE_NAME} -f"
