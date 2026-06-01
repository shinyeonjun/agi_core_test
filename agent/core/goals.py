from __future__ import annotations

import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db

OPEN_STATUSES = ("proposed", "active", "waiting_approval", "blocked")
NOISE_GOAL_TYPES = {"answer_user", "test", "debug", "debug_smoke", "smoke"}
NOISE_TITLE_TOKENS = ("answer user input", "secret goal summary marker", "apply user negative feedback")


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


def goal_metadata(goal: dict[str, Any] | None) -> dict[str, Any]:
    if not goal:
        return {}
    raw = goal.get("metadata_json")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def update_goal_metadata(goal_id: int, metadata: dict[str, Any], *, status: str | None = None) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE goals SET metadata_json = ?, status = COALESCE(?, status), updated_at = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), status, now_kst(), goal_id),
        )
        conn.commit()
        return cur.rowcount > 0


def is_noise_goal_record(goal: dict[str, Any]) -> bool:
    title = str(goal.get("title") or "").lower()
    goal_type = str(goal.get("goal_type") or "").lower()
    return goal_type in NOISE_GOAL_TYPES or any(token in title for token in NOISE_TITLE_TOKENS)


def cleanup_noise_goals() -> dict[str, Any]:
    init_db()
    placeholders = ",".join("?" for _ in OPEN_STATUSES)
    with connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM goals WHERE status IN ({placeholders})",
            OPEN_STATUSES,
        ).fetchall()
        noise_ids = [int(row["id"]) for row in rows if is_noise_goal_record(dict(row))]
        redacted = conn.execute(
            """
            UPDATE goals
            SET description = CASE
                    WHEN description IS NOT NULL AND description != '' THEN 'redacted test/noise goal'
                    ELSE description
                END,
                metadata_json = '{}',
                updated_at = ?
            WHERE (
                description LIKE '%TOKEN%'
                OR description LIKE '%token%'
                OR description LIKE '%SECRET%'
                OR description LIKE '%secret%'
                OR metadata_json LIKE '%token%'
                OR metadata_json LIKE '%TOKEN%'
                OR metadata_json LIKE '%secret%'
                OR metadata_json LIKE '%SECRET%'
            )
            """,
            (now_kst(),),
        ).rowcount
        marked = 0
        if noise_ids:
            id_placeholders = ",".join("?" for _ in noise_ids)
            marked = conn.execute(
                f"UPDATE goals SET status = 'done', updated_at = ? WHERE id IN ({id_placeholders})",
                (now_kst(), *noise_ids),
            ).rowcount
        conn.commit()
    return {"marked_done": int(marked), "redacted": int(redacted), "noise_goal_ids": noise_ids}
