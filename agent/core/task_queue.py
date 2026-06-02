from __future__ import annotations

import json
from typing import Any, Literal

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event

QueueType = Literal["user", "autonomous"]
TaskStatus = Literal["queued", "running", "done", "blocked", "waiting_approval", "skipped"]

OPEN_TASK_STATUSES = ("queued", "running", "waiting_approval")


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    raw_payload = row.get("payload_json")
    raw_result = row.get("result_json")
    try:
        row["payload"] = json.loads(raw_payload) if raw_payload else {}
    except json.JSONDecodeError:
        row["payload"] = {"decode_error": True}
    try:
        row["result"] = json.loads(raw_result) if raw_result else {}
    except json.JSONDecodeError:
        row["result"] = {"decode_error": True}
    return row


def _existing_open_task(goal_id: int | None, queue_type: str | None = None) -> dict[str, Any] | None:
    if goal_id is None:
        return None
    init_db()
    placeholders = ",".join("?" for _ in OPEN_TASK_STATUSES)
    params: list[Any] = [goal_id, *OPEN_TASK_STATUSES]
    query = f"SELECT * FROM task_queue WHERE goal_id = ? AND status IN ({placeholders})"
    if queue_type:
        query += " AND queue_type = ?"
        params.append(queue_type)
    query += " ORDER BY id DESC LIMIT 1"
    with connect() as conn:
        row = conn.execute(query, tuple(params)).fetchone()
    return _decode(dict(row)) if row else None


def enqueue_task(
    queue_type: QueueType,
    *,
    goal_id: int | None,
    task_kind: str,
    title: str,
    source: str,
    priority: float = 0.5,
    status: TaskStatus = "queued",
    payload: dict[str, Any] | None = None,
    dedupe_goal: bool = True,
) -> int:
    if queue_type not in {"user", "autonomous"}:
        raise ValueError(f"invalid queue_type: {queue_type}")
    if status not in {"queued", "running", "done", "blocked", "waiting_approval", "skipped"}:
        raise ValueError(f"invalid task status: {status}")
    if dedupe_goal:
        existing = _existing_open_task(goal_id, queue_type)
        if existing:
            return int(existing["id"])
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO task_queue (
                created_at, updated_at, queue_type, status, priority, goal_id,
                task_kind, title, source, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, ts, queue_type, status, max(0.0, min(1.0, float(priority))),
                goal_id, task_kind, title[:160], source,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        conn.commit()
        task_id = int(cur.lastrowid)
    log_event("task_queue", "task_enqueued", f"{queue_type}:{task_kind}", {"task_id": task_id, "goal_id": goal_id, "status": status}, 0.7 if queue_type == "user" else 0.55)
    return task_id


def task_for_goal(goal_id: int, queue_type: str | None = None) -> dict[str, Any] | None:
    return _existing_open_task(goal_id, queue_type)


def _claim_where(where_sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute(
            f"""
            SELECT * FROM task_queue
            WHERE {where_sql}
            ORDER BY
                queue_type = 'user' DESC,
                priority DESC,
                id ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if not row:
            return None
        task = dict(row)
        ts = now_kst()
        conn.execute(
            """
            UPDATE task_queue
            SET status = 'running', updated_at = ?, claimed_at = ?, attempts = attempts + 1
            WHERE id = ? AND status = 'queued'
            """,
            (ts, ts, task["id"]),
        )
        conn.commit()
    task["status"] = "running"
    task["claimed_at"] = ts
    task["attempts"] = int(task.get("attempts") or 0) + 1
    return _decode(task)


def claim_next_task(queue_type: str | None = None) -> dict[str, Any] | None:
    if queue_type:
        return _claim_where("status = 'queued' AND queue_type = ?", (queue_type,))
    return _claim_where("status = 'queued'", ())


def claim_task(task_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM task_queue WHERE id = ? AND status = 'queued'", (task_id,)).fetchone()
    if not row:
        return None
    return _claim_where("status = 'queued' AND id = ?", (task_id,))


def finish_task(task_id: int, status: TaskStatus, result: dict[str, Any] | None = None) -> bool:
    if status not in {"done", "blocked", "waiting_approval", "skipped"}:
        raise ValueError(f"finish status must be terminal/waiting: {status}")
    init_db()
    ts = now_kst()
    completed_at = ts if status in {"done", "blocked", "skipped"} else None
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE task_queue
            SET status = ?, updated_at = ?, completed_at = ?, result_json = ?
            WHERE id = ?
            """,
            (status, ts, completed_at, json.dumps(result or {}, ensure_ascii=False), task_id),
        )
        conn.commit()
        ok = cur.rowcount > 0
    if ok:
        log_event("task_queue", f"task_{status}", str(task_id), {"task_id": task_id, "result": result or {}}, 0.7)
    return ok


def requeue_task(task_id: int, result: dict[str, Any] | None = None) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE task_queue
            SET status = 'queued', updated_at = ?, result_json = ?
            WHERE id = ?
            """,
            (now_kst(), json.dumps(result or {}, ensure_ascii=False), task_id),
        )
        conn.commit()
        return cur.rowcount > 0


def list_tasks(limit: int = 20, status: str | None = None, queue_type: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM task_queue"
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if queue_type:
        clauses.append("queue_type = ?")
        params.append(queue_type)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY status = 'running' DESC, status = 'queued' DESC, queue_type = 'user' DESC, priority DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_decode(dict(row)) for row in rows]


def task_status_counts() -> dict[str, int]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT queue_type, status, COUNT(*) AS count FROM task_queue GROUP BY queue_type, status").fetchall()
    return {f"{row['queue_type']}:{row['status']}": int(row["count"]) for row in rows}
