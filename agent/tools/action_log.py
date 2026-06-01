from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db


def create_action_run(
    *,
    goal_id: int | None,
    action_type: str,
    command: list[str] | str,
    cwd: str,
    profile: str,
    risk_level: str,
    status: str,
    before_snapshot: dict[str, Any] | None = None,
    result_summary: str | None = None,
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO action_runs (
                created_at, goal_id, action_type, command_json, cwd, profile, risk_level,
                status, before_snapshot_json, result_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst(),
                goal_id,
                action_type,
                json.dumps(command, ensure_ascii=False),
                cwd,
                profile,
                risk_level,
                status,
                json.dumps(before_snapshot or {}, ensure_ascii=False),
                result_summary,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def finish_action_run(
    action_id: int,
    *,
    status: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    after_snapshot: dict[str, Any] | None,
    result_summary: str,
) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE action_runs
            SET completed_at = ?, status = ?, returncode = ?, stdout = ?, stderr = ?,
                after_snapshot_json = ?, result_summary = ?
            WHERE id = ?
            """,
            (
                now_kst(),
                status,
                returncode,
                stdout,
                stderr,
                json.dumps(after_snapshot or {}, ensure_ascii=False),
                result_summary,
                action_id,
            ),
        )
        conn.commit()


def list_action_runs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM action_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_action_run(action_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM action_runs WHERE id = ?", (action_id,)).fetchone()
    return dict(row) if row else None
