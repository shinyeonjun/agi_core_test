from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db

OPEN_STATUSES = ("proposed", "active", "waiting_approval", "blocked")


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
    with connect() as conn:
        if dedupe:
            existing = conn.execute(
                """
                SELECT id FROM goals
                WHERE title = ? AND goal_type = ? AND status IN ('proposed', 'active', 'waiting_approval', 'blocked')
                ORDER BY id DESC LIMIT 1
                """,
                (title, goal_type),
            ).fetchone()
            if existing:
                return int(existing["id"])
        ts = now_kst()
        cur = conn.execute(
            """
            INSERT INTO goals (
                created_at, updated_at, title, description, goal_type, status,
                priority, risk_level, requires_approval, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                ts,
                title,
                description,
                goal_type,
                status,
                priority,
                risk_level,
                1 if requires_approval else 0,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_goals(limit: int = 20, include_archived: bool = False) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM goals"
    params: tuple[Any, ...] = ()
    if not include_archived:
        query += " WHERE status != 'archived'"
    query += " ORDER BY status = 'active' DESC, priority DESC, id DESC LIMIT ?"
    params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
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
        row = conn.execute(
            """
            SELECT id FROM goals
            WHERE goal_type = ? AND title = ? AND created_at >= ?
            ORDER BY id DESC LIMIT 1
            """,
            (goal_type, title, cutoff),
        ).fetchone()
    return row is not None


def mark_goal_done(goal_id: int) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE goals SET status = 'done', updated_at = ? WHERE id = ?",
            (now_kst(), goal_id),
        )
        conn.commit()
        return cur.rowcount > 0
