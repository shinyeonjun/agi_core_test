from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db

DEFAULT_COOLDOWNS = {
    "system_check": 30 * 60,
    "memory_cleanup": 60 * 60,
    "discord_answer": 3,
    "codex_renderer_call": 10,
    "suggestion_draft": 6 * 60 * 60,
}


def is_ready(key: str, seconds: int | None = None) -> tuple[bool, int]:
    init_db()
    seconds = int(seconds if seconds is not None else DEFAULT_COOLDOWNS.get(key, 60))
    with connect() as conn:
        row = conn.execute("SELECT updated_at, seconds FROM cooldowns WHERE key = ?", (key,)).fetchone()
    if not row:
        return True, 0
    last = datetime.fromisoformat(row["updated_at"])
    elapsed = (datetime.now(KST) - last).total_seconds()
    wait = max(0, int(row["seconds"] - elapsed))
    return wait == 0, wait


def mark(key: str, seconds: int | None = None, metadata: dict[str, Any] | None = None) -> None:
    init_db()
    seconds = int(seconds if seconds is not None else DEFAULT_COOLDOWNS.get(key, 60))
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO cooldowns (key, updated_at, seconds, metadata_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                updated_at = excluded.updated_at,
                seconds = excluded.seconds,
                metadata_json = excluded.metadata_json
            """,
            (key, now_kst(), seconds, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
