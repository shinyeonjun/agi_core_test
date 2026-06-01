from __future__ import annotations

import json
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db

MAX_EVENT_CONTENT_CHARS = 20000
SECRET_PATTERNS = (
    re.compile(r"(?i)(token|api[_-]?key|authorization:\s*bearer)\s*[:=]\s*[^\s]+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)


def redact_text(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: m.group(0).split("=", 1)[0] + "=<redacted>" if "=" in m.group(0) else "<redacted_secret>", redacted)
    if len(redacted) > MAX_EVENT_CONTENT_CHARS:
        redacted = redacted[:MAX_EVENT_CONTENT_CHARS] + "...[truncated]"
    return redacted


def log_event(source: str, event_type: str, content: str, metadata: dict[str, Any] | None = None, importance: float = 0.5) -> int:
    init_db()
    safe_content = redact_text(str(content))
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO events (ts, source, event_type, content, metadata_json, importance, processed)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (now_kst(), source, event_type, safe_content, json.dumps(metadata or {}, ensure_ascii=False), importance),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_events(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, ts, source, event_type, content, importance, processed FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_unprocessed_events() -> int:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM events WHERE processed = 0").fetchone()
    return int(row["count"])


def mark_events_processed(limit: int = 50) -> int:
    init_db()
    with connect() as conn:
        ids = [row["id"] for row in conn.execute(
            "SELECT id FROM events WHERE processed = 0 ORDER BY id ASC LIMIT ?", (limit,)
        ).fetchall()]
        if not ids:
            return 0
        conn.executemany("UPDATE events SET processed = 1 WHERE id = ?", [(event_id,) for event_id in ids])
        conn.commit()
        return len(ids)
