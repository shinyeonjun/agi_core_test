from __future__ import annotations

import json
import shlex
from datetime import datetime, timedelta
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json")
    try:
        row["metadata"] = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        row["metadata"] = {"decode_error": True}
    return row


def normalize_command(command: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(command, (list, tuple)):
        parts = [str(part) for part in command]
    else:
        try:
            decoded = json.loads(str(command))
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            parts = [str(part) for part in decoded]
        elif isinstance(decoded, str):
            parts = shlex.split(decoded)
        else:
            parts = shlex.split(str(command))
    return " ".join(parts).strip()


def create_action_proposal(
    *,
    goal_id: int | None,
    command: str,
    cwd: str | None,
    profile: str,
    risk_level: str,
    status: str,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO action_proposals (
                created_at, goal_id, command, cwd, profile, risk_level, status, reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst(),
                goal_id,
                command,
                cwd,
                profile,
                risk_level,
                status,
                reason,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_action_proposal_status(proposal_id: int, status: str, reason: str | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE action_proposals SET status = ?, reason = COALESCE(?, reason) WHERE id = ?",
            (status, reason, proposal_id),
        )
        conn.commit()


def list_action_proposals(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    if status:
        query = "SELECT * FROM action_proposals WHERE status = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (status, limit)
    else:
        query = "SELECT * FROM action_proposals ORDER BY id DESC LIMIT ?"
        params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_decode(dict(row)) for row in rows]


def get_action_proposal(proposal_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM action_proposals WHERE id = ?", (proposal_id,)).fetchone()
    return _decode(dict(row)) if row else None


def proposal_status_counts() -> dict[str, int]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM action_proposals GROUP BY status").fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def has_recent_goal_command(goal_id: int | None, command: str, within_hours: int = 24) -> bool:
    if goal_id is None:
        return False
    cutoff = (datetime.now(KST) - timedelta(hours=within_hours)).isoformat(timespec="seconds")
    target = normalize_command(command)
    init_db()
    with connect() as conn:
        proposals = conn.execute(
            """
            SELECT command FROM action_proposals
            WHERE goal_id = ?
              AND created_at >= ?
              AND status IN ('proposed', 'approved_by_policy', 'executed')
            ORDER BY id DESC
            """,
            (goal_id, cutoff),
        ).fetchall()
        runs = conn.execute(
            """
            SELECT command_json FROM action_runs
            WHERE goal_id = ?
              AND created_at >= ?
              AND status IN ('running', 'completed', 'timeout')
            ORDER BY id DESC
            """,
            (goal_id, cutoff),
        ).fetchall()
    return any(normalize_command(row["command"]) == target for row in proposals) or any(normalize_command(row["command_json"]) == target for row in runs)
