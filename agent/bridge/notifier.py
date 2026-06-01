from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from agent.bridge.formatter import redact_discord_content, split_for_discord
from agent.config.defaults import env_path
from agent.core.events import log_event

WebhookKind = Literal["summary", "update"]
WEBHOOK_ENV = {
    "summary": "DISCORD_SUMMARY_WEBHOOK_URL",
    "update": "DISCORD_UPDATE_WEBHOOK_URL",
}


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
