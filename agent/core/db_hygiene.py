from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goals import goal_metadata
from agent.core.task_queue import doctor_tasks

OPEN_GOAL_STATUSES = ("proposed", "active", "waiting_approval", "blocked")
OPEN_TASK_STATUSES = ("queued", "running", "waiting_approval")
CANCEL_NOISE_TOKENS = ("목표에서", "없애", "삭제", "지워", "취소", "cancel", "remove", "delete")


def _placeholders(values: tuple[str, ...] | list[int]) -> str:
    return ",".join("?" for _ in values)


def _decode_json(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _is_cancel_noise(row: dict[str, Any]) -> bool:
    haystack = " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("description") or ""),
            str(row.get("source") or ""),
            json.dumps(_decode_json(row.get("metadata_json") or row.get("payload_json")), ensure_ascii=False),
        ]
    ).lower()
    has_cancel = any(token.lower() in haystack for token in CANCEL_NOISE_TOKENS)
    has_context = any(token in haystack for token in ("목표", "작업", "task", "goal", "#"))
    return has_cancel and has_context


def _stale_running_ids(max_age_seconds: int) -> list[int]:
    from datetime import datetime

    now = datetime.now(KST)
    threshold = timedelta(seconds=max(1, int(max_age_seconds)))
    ids: list[int] = []
    with connect() as conn:
        rows = conn.execute("SELECT id, claimed_at, locked_until FROM task_queue WHERE status = 'running'").fetchall()
    for row in rows:
        locked_until = row["locked_until"]
        claimed_at = row["claimed_at"]
        try:
            locked_dt = datetime.fromisoformat(str(locked_until)) if locked_until else None
            claimed_dt = datetime.fromisoformat(str(claimed_at)) if claimed_at else None
        except ValueError:
            locked_dt = None
            claimed_dt = None
        if locked_dt is not None and locked_dt > now:
            continue
        if claimed_dt is not None and now - claimed_dt < threshold:
            continue
        ids.append(int(row["id"]))
    return ids


def _duplicate_discord_event_count() -> int:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(SUM(count - 1), 0) AS count
            FROM (
                SELECT channel_id, user_id, message_id, COUNT(*) AS count
                FROM discord_events
                WHERE message_id IS NOT NULL AND message_id != ''
                GROUP BY channel_id, user_id, message_id
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
    return int(row["count"] if row else 0)


def _open_cancel_noise() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_status_sql = _placeholders(OPEN_TASK_STATUSES)
    goal_status_sql = _placeholders(OPEN_GOAL_STATUSES)
    with connect() as conn:
        task_rows = conn.execute(
            f"SELECT * FROM task_queue WHERE status IN ({task_status_sql}) ORDER BY id DESC LIMIT 200",
            OPEN_TASK_STATUSES,
        ).fetchall()
        goal_rows = conn.execute(
            f"SELECT * FROM goals WHERE status IN ({goal_status_sql}) ORDER BY id DESC LIMIT 200",
            OPEN_GOAL_STATUSES,
        ).fetchall()
    tasks = [dict(row) for row in task_rows]
    goals = [dict(row) for row in goal_rows]
    return [row for row in tasks if _is_cancel_noise(row)], [row for row in goals if _is_cancel_noise(row)]


def _open_tasks_for_closed_goals() -> list[dict[str, Any]]:
    task_status_sql = _placeholders(OPEN_TASK_STATUSES)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT task_queue.*
            FROM task_queue
            JOIN goals ON goals.id = task_queue.goal_id
            WHERE task_queue.status IN ({task_status_sql})
              AND goals.status IN ('done', 'archived')
            ORDER BY task_queue.id DESC
            LIMIT 200
            """,
            OPEN_TASK_STATUSES,
        ).fetchall()
    return [dict(row) for row in rows]


def cleanup_db_noise(*, apply: bool = False, stale_seconds: int = 1800) -> dict[str, Any]:
    init_db()
    cancel_tasks, cancel_goals = _open_cancel_noise()
    closed_goal_tasks = _open_tasks_for_closed_goals()
    stale_ids = _stale_running_ids(stale_seconds)
    duplicate_discord_events = _duplicate_discord_event_count()
    result: dict[str, Any] = {
        "applied": bool(apply),
        "cancel_noise_tasks": [int(row["id"]) for row in cancel_tasks],
        "cancel_noise_goals": [int(row["id"]) for row in cancel_goals],
        "closed_goal_tasks": [int(row["id"]) for row in closed_goal_tasks],
        "stale_running_tasks": stale_ids,
        "duplicate_discord_events": duplicate_discord_events,
        "changed": {"tasks_skipped": 0, "closed_goal_tasks_skipped": 0, "goals_archived": 0, "stale_recovered": 0, "stale_blocked": 0},
    }
    if not apply:
        return result

    ts = now_kst()
    with connect() as conn:
        task_ids = [int(row["id"]) for row in cancel_tasks]
        if task_ids:
            cur = conn.execute(
                f"""
                UPDATE task_queue
                SET status = 'skipped', updated_at = ?, completed_at = ?,
                    locked_until = NULL, locked_by = NULL,
                    result_json = ?
                WHERE id IN ({_placeholders(task_ids)}) AND status IN ({_placeholders(OPEN_TASK_STATUSES)})
                """,
                (
                    ts,
                    ts,
                    json.dumps({"reason": "db_hygiene_cancel_request_noise"}, ensure_ascii=False),
                    *task_ids,
                    *OPEN_TASK_STATUSES,
                ),
            )
            result["changed"]["tasks_skipped"] = int(cur.rowcount)
        closed_task_ids = [int(row["id"]) for row in closed_goal_tasks]
        if closed_task_ids:
            cur = conn.execute(
                f"""
                UPDATE task_queue
                SET status = 'skipped', updated_at = ?, completed_at = ?,
                    locked_until = NULL, locked_by = NULL,
                    result_json = ?
                WHERE id IN ({_placeholders(closed_task_ids)}) AND status IN ({_placeholders(OPEN_TASK_STATUSES)})
                """,
                (
                    ts,
                    ts,
                    json.dumps({"reason": "closed_goal"}, ensure_ascii=False),
                    *closed_task_ids,
                    *OPEN_TASK_STATUSES,
                ),
            )
            result["changed"]["closed_goal_tasks_skipped"] = int(cur.rowcount)
        goal_ids = [int(row["id"]) for row in cancel_goals]
        archived = 0
        for row in cancel_goals:
            metadata = goal_metadata(row)
            metadata["archived_by_db_hygiene"] = True
            metadata["archive_reason"] = "cancel_request_noise"
            cur = conn.execute(
                f"""
                UPDATE goals
                SET status = 'archived', updated_at = ?, metadata_json = ?
                WHERE id = ? AND status IN ({_placeholders(OPEN_GOAL_STATUSES)})
                """,
                (ts, json.dumps(metadata, ensure_ascii=False), int(row["id"]), *OPEN_GOAL_STATUSES),
            )
            archived += int(cur.rowcount)
        result["changed"]["goals_archived"] = archived
        if goal_ids:
            conn.execute(
                f"""
                UPDATE project_execution_plans
                SET status = 'blocked', updated_at = ?, result_json = ?
                WHERE goal_id IN ({_placeholders(goal_ids)}) AND status IN ('planned', 'running', 'blocked')
                """,
                (ts, json.dumps({"status": "cancelled", "reason": "db_hygiene_cancel_request_noise"}, ensure_ascii=False), *goal_ids),
            )
        conn.commit()

    doctor = doctor_tasks(max_age_seconds=stale_seconds)
    result["changed"]["stale_recovered"] = int((doctor.get("stale_running") or {}).get("recovered") or 0)
    result["changed"]["stale_blocked"] = int((doctor.get("stale_running") or {}).get("blocked") or 0)
    log_event("db", "db_hygiene_cleanup", "apply", result, 0.72)
    return result
