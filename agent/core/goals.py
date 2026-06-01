from __future__ import annotations

import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db

OPEN_STATUSES = ("proposed", "active", "waiting_approval", "blocked")


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def find_duplicate_goal(title: str, goal_type: str, threshold: float = 0.85) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM goals WHERE goal_type = ? AND status IN ('proposed', 'active', 'waiting_approval', 'blocked')",
            (goal_type,),
        ).fetchall()
    for row in rows:
        item = dict(row)
        if similar(str(item["title"]), title) >= threshold:
            return item
    return None


def create_goal(
    title: str,
    description: str | None = None,
    goal_type: str = "general",
    status: str = "active",
    priority: float = 0.5,
    risk_level: str = "low",
    requires_approval: bool = False,
    metadata: dict[str, Any] | None = None,
    dedupe: bool = True,
) -> int:
    init_db()
    if dedupe:
        duplicate = find_duplicate_goal(title, goal_type)
        if duplicate:
            return int(duplicate["id"])
    with connect() as conn:
        ts = now_kst()
        cur = conn.execute(
            """
            INSERT INTO goals (
                created_at, updated_at, title, description, goal_type, status,
                priority, risk_level, requires_approval, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, ts, title, description, goal_type, status, priority, risk_level,
                1 if requires_approval else 0, json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_goals(limit: int = 20, include_archived: bool = False) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM goals"
    if not include_archived:
        query += " WHERE status != 'archived'"
    query += " ORDER BY status = 'active' DESC, priority DESC, id DESC LIMIT ?"
    with connect() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [dict(row) for row in rows]


def count_open_goals() -> int:
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM goals WHERE status IN ('proposed', 'active', 'waiting_approval', 'blocked')"
        ).fetchone()
    return int(row["count"])


def has_recent_goal(goal_type: str, title: str, within_minutes: int = 60) -> bool:
    init_db()
    cutoff = (datetime.now(KST) - timedelta(minutes=within_minutes)).isoformat(timespec="seconds")
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title FROM goals
            WHERE goal_type = ? AND created_at >= ?
            ORDER BY id DESC
            """,
            (goal_type, cutoff),
        ).fetchall()
    return any(similar(row["title"], title) >= 0.85 for row in rows)


def mark_goal_done(goal_id: int) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE goals SET status = 'done', updated_at = ? WHERE id = ?",
            (now_kst(), goal_id),
        )
        conn.commit()
        return cur.rowcount > 0
