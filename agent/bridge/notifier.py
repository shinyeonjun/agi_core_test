from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from agent.bridge.formatter import redact_discord_content, split_for_discord
from agent.config.defaults import db_path, env_bool, env_path, project_root
from agent.core.events import log_event

WebhookKind = Literal["summary", "update"]
BotChannelKind = Literal["chat", "approval"]
WEBHOOK_ENV = {
    "summary": "DISCORD_SUMMARY_WEBHOOK_URL",
    "update": "DISCORD_UPDATE_WEBHOOK_URL",
}
BOT_CHANNEL_ENV = {
    "chat": "DISCORD_CHAT_CHANNEL_ID",
    "approval": "DISCORD_APPROVAL_CHANNEL_ID",
}


def external_notifications_allowed() -> bool:
    if env_bool("AGENT_DISABLE_EXTERNAL_NOTIFICATIONS", False):
        return False
    if env_bool("AGENT_ALLOW_EXTERNAL_NOTIFICATIONS", False):
        return True
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    root = project_root()
    expected_db = (root / "data" / "agent.db").resolve()
    expected_env = (root / ".env").resolve()
    return db_path() == expected_db and env_path() == expected_env


def load_env_file(path: Path | None = None) -> None:
    path = path or env_path()
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def webhook_url(kind: WebhookKind) -> str | None:
    load_env_file()
    value = os.environ.get(WEBHOOK_ENV[kind], "").strip()
    return value or None


def bot_channel_id(kind: BotChannelKind) -> str | None:
    load_env_file()
    value = os.environ.get(BOT_CHANNEL_ENV[kind], "").strip()
    return value or None


def post_webhook(kind: WebhookKind, content: str, *, username: str = "Agent Core", dry_run: bool = False) -> dict[str, Any]:
    safe_content = redact_discord_content(content)
    chunks = split_for_discord(safe_content, 1800)
    url = webhook_url(kind)
    if not url:
        result = {"sent": False, "kind": kind, "reason": "missing_webhook_url", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
        log_event("discord", "webhook_missing", kind, {"kind": kind}, 0.45)
        return result
    if dry_run:
        return {"sent": False, "kind": kind, "reason": "dry_run", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
    if not external_notifications_allowed():
        return {"sent": False, "kind": kind, "reason": "external_notifications_disabled", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
    sent = 0
    for chunk in chunks:
        payload = json.dumps({"username": username, "content": chunk}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "agent-core-discord-control-plane"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status >= 300:
                    return {"sent": False, "kind": kind, "reason": f"http_{response.status}", "chunks": sent}
                sent += 1
        except urllib.error.URLError as exc:
            return {"sent": False, "kind": kind, "reason": "request_failed", "error": redact_discord_content(str(exc)), "chunks": sent}
    log_event("discord", "webhook_sent", kind, {"kind": kind, "chunks": sent}, 0.55)
    return {"sent": True, "kind": kind, "chunks": sent}


def post_bot_channel(kind: BotChannelKind, content: str, *, dry_run: bool = False) -> dict[str, Any]:
    safe_content = redact_discord_content(content)
    chunks = split_for_discord(safe_content, 1800)
    load_env_file()
    token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
    channel_id = bot_channel_id(kind)
    if not token or not channel_id:
        return {"sent": False, "kind": kind, "reason": "missing_bot_token_or_channel", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
    if dry_run:
        return {"sent": False, "kind": kind, "reason": "dry_run", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
    if not external_notifications_allowed():
        return {"sent": False, "kind": kind, "reason": "external_notifications_disabled", "chunks": len(chunks), "preview": chunks[0] if chunks else ""}
    sent = 0
    for chunk in chunks:
        payload = json.dumps({"content": chunk}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bot {token}", "User-Agent": "agent-core-discord-control-plane"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                if response.status >= 300:
                    return {"sent": False, "kind": kind, "reason": f"http_{response.status}", "chunks": sent}
                sent += 1
        except urllib.error.URLError as exc:
            return {"sent": False, "kind": kind, "reason": "request_failed", "error": redact_discord_content(str(exc)), "chunks": sent}
    log_event("discord", "bot_channel_sent", kind, {"kind": kind, "chunks": sent}, 0.6)
    return {"sent": True, "kind": kind, "chunks": sent}
