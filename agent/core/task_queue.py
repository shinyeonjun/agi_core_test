from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timedelta
from typing import Any, Literal

from agent.config.defaults import KST, env_int, now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.task_lifecycle import record_task_phase
from agent.core.wake_signals import emit_wake_signal

QueueType = Literal["user", "autonomous"]
TaskStatus = Literal["queued", "running", "done", "blocked", "waiting_approval", "skipped"]

OPEN_TASK_STATUSES = ("queued", "running", "waiting_approval")
MAX_TASK_ATTEMPTS = 3


def _worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _task_lock_seconds() -> int:
    return max(30, env_int("AGENT_TASK_LOCK_SECONDS", 1800))


def _now_dt() -> datetime:
    return datetime.now(KST)


def _future_kst(seconds: int) -> str:
    return (_now_dt() + timedelta(seconds=max(1, int(seconds)))).isoformat(timespec="seconds")


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


def _existing_idempotent_task(idempotency_key: str | None) -> dict[str, Any] | None:
    if not idempotency_key:
        return None
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM task_queue WHERE idempotency_key = ? ORDER BY id DESC LIMIT 1", (idempotency_key,)).fetchone()
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
    approval_id: int | None = None,
    payload: dict[str, Any] | None = None,
    dedupe_goal: bool = True,
    idempotency_key: str | None = None,
    not_before: str | None = None,
    due_at: str | None = None,
    max_attempts: int = MAX_TASK_ATTEMPTS,
) -> int:
    if queue_type not in {"user", "autonomous"}:
        raise ValueError(f"invalid queue_type: {queue_type}")
    if status not in {"queued", "running", "done", "blocked", "waiting_approval", "skipped"}:
        raise ValueError(f"invalid task status: {status}")
    if dedupe_goal:
        existing = _existing_open_task(goal_id, queue_type)
        if existing:
            return int(existing["id"])
    existing_idempotent = _existing_idempotent_task(idempotency_key)
    if existing_idempotent:
        return int(existing_idempotent["id"])
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO task_queue (
                created_at, updated_at, queue_type, status, priority, goal_id, approval_id,
                task_kind, title, source, payload_json, max_attempts, idempotency_key, not_before, due_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts, ts, queue_type, status, max(0.0, min(1.0, float(priority))),
                goal_id, approval_id, task_kind, title[:160], source,
                json.dumps(payload or {}, ensure_ascii=False),
                max(1, int(max_attempts)), idempotency_key, not_before, due_at,
            ),
        )
        conn.commit()
        task_id = int(cur.lastrowid)
    log_event("task_queue", "task_enqueued", f"{queue_type}:{task_kind}", {"task_id": task_id, "goal_id": goal_id, "status": status}, 0.7 if queue_type == "user" else 0.55)
    if status == "queued":
        emit_wake_signal(
            f"{queue_type}_task_queued",
            "task_queue",
            priority=0.95 if queue_type == "user" else 0.62,
            payload={"task_id": task_id, "goal_id": goal_id, "task_kind": task_kind, "queue_type": queue_type},
            dedupe_key=f"task_queued:{task_id}",
        )
    record_task_phase(
        task_id,
        "queued",
        status,
        "사용자 작업 큐에 등록됨" if queue_type == "user" else "자율 작업 큐에 등록됨",
        queue_type=queue_type,
        metadata={"goal_id": goal_id, "task_kind": task_kind, "source": source},
    )
    return task_id


def task_for_goal(goal_id: int, queue_type: str | None = None) -> dict[str, Any] | None:
    return _existing_open_task(goal_id, queue_type)


def _claim_where(where_sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    init_db()
    ts = now_kst()
    lock_until = _future_kst(_task_lock_seconds())
    locked_by = _worker_id()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"""
            SELECT * FROM task_queue
            WHERE {where_sql}
              AND attempts < COALESCE(max_attempts, ?)
              AND (not_before IS NULL OR not_before <= ?)
              AND (locked_until IS NULL OR locked_until <= ?)
            ORDER BY
                queue_type = 'user' DESC,
                priority DESC,
                id ASC
            LIMIT 1
            """,
            (*params, MAX_TASK_ATTEMPTS, ts, ts),
        ).fetchone()
        if not row:
            conn.commit()
            return None
        task = dict(row)
        conn.execute(
            """
            UPDATE task_queue
            SET status = 'running', updated_at = ?, claimed_at = ?, locked_until = ?, locked_by = ?, attempts = attempts + 1
            WHERE id = ? AND status = 'queued'
            """,
            (ts, ts, lock_until, locked_by, task["id"]),
        )
        conn.commit()
    task["status"] = "running"
    task["claimed_at"] = ts
    task["locked_until"] = lock_until
    task["locked_by"] = locked_by
    task["attempts"] = int(task.get("attempts") or 0) + 1
    record_task_phase(
        int(task["id"]),
        "planning",
        "claimed",
        "작업자가 큐에서 작업을 가져옴",
        queue_type=task.get("queue_type"),
        metadata={"attempts": task["attempts"], "goal_id": task.get("goal_id")},
    )
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
            SET status = ?, updated_at = ?, completed_at = ?, locked_until = NULL, locked_by = NULL, result_json = ?
            WHERE id = ?
            """,
            (status, ts, completed_at, json.dumps(result or {}, ensure_ascii=False), task_id),
        )
        conn.commit()
        ok = cur.rowcount > 0
    if ok:
        log_event("task_queue", f"task_{status}", str(task_id), {"task_id": task_id, "result": result or {}}, 0.7)
        emit_wake_signal(
            f"task_{status}",
            "task_queue",
            priority=0.82 if status == "blocked" else 0.58,
            payload={"task_id": task_id, "status": status, "result_status": (result or {}).get("status"), "reason": (result or {}).get("reason")},
            dedupe_key=f"task_finished:{task_id}:{status}",
        )
        record_task_phase(
            task_id,
            "reporting",
            status,
            "작업 결과를 큐에 기록함",
            queue_type=(result or {}).get("queue_type"),
            metadata={"result_status": (result or {}).get("status"), "reason": (result or {}).get("reason")},
        )
        try:
            from agent.bridge.task_notifications import notify_task_finished

            notify_task_finished(task_id, status, result or {})
        except Exception as exc:
            log_event("discord", "task_finished_notify_failed", str(task_id), {"error": type(exc).__name__, "status": status}, 0.35)
    return ok


def requeue_task(task_id: int, result: dict[str, Any] | None = None) -> bool:
    init_db()
    payload = result or {}
    with connect() as conn:
        row = conn.execute("SELECT attempts FROM task_queue WHERE id = ?", (task_id,)).fetchone()
        if not row:
            return False
        attempts = int(row["attempts"] or 0)
        if attempts >= MAX_TASK_ATTEMPTS:
            cur = conn.execute(
                """
                UPDATE task_queue
                SET status = 'blocked', updated_at = ?, completed_at = ?, locked_until = NULL, locked_by = NULL, result_json = ?
                WHERE id = ?
                """,
                (now_kst(), now_kst(), json.dumps({**payload, "reason": "max_attempts_exceeded"}, ensure_ascii=False), task_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
            if ok:
                emit_wake_signal(
                    "task_blocked",
                    "task_queue",
                    priority=0.82,
                    payload={"task_id": task_id, "reason": "max_attempts_exceeded"},
                    dedupe_key=f"task_blocked:{task_id}:max_attempts",
                )
                record_task_phase(task_id, "reporting", "blocked", "최대 재시도 초과로 작업 차단", metadata=payload)
            return ok
        cur = conn.execute(
            """
            UPDATE task_queue
            SET status = 'queued', updated_at = ?, claimed_at = NULL, locked_until = NULL, locked_by = NULL, result_json = ?
            WHERE id = ?
            """,
            (now_kst(), json.dumps(payload, ensure_ascii=False), task_id),
        )
        conn.commit()
        ok = cur.rowcount > 0
    if ok:
        emit_wake_signal(
            "task_requeued",
            "task_queue",
            priority=0.72,
            payload={"task_id": task_id, "reason": payload.get("reason")},
            dedupe_key=f"task_requeued:{task_id}",
        )
        record_task_phase(task_id, "queued", "requeued", "작업을 다시 큐에 넣음", metadata=payload)
    return ok


def resume_tasks_for_approval(approval_id: int) -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE task_queue
            SET status = 'queued', updated_at = ?, locked_until = NULL, locked_by = NULL, result_json = NULL
            WHERE approval_id = ? AND status = 'waiting_approval'
            """,
            (ts, approval_id),
        )
        conn.commit()
        count = cur.rowcount
    if count:
        log_event("task_queue", "approval_tasks_resumed", str(approval_id), {"approval_id": approval_id, "count": count}, 0.75)
        emit_wake_signal("approval_changed", "task_queue", priority=0.9, payload={"approval_id": approval_id, "resumed": count}, dedupe_key=f"approval_changed:{approval_id}:resumed")
    return count


def block_tasks_for_approval(approval_id: int, reason: str = "approval_rejected") -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE task_queue
            SET status = 'blocked', updated_at = ?, completed_at = ?, locked_until = NULL, locked_by = NULL, result_json = ?
            WHERE approval_id = ? AND status = 'waiting_approval'
            """,
            (ts, ts, json.dumps({"reason": reason}, ensure_ascii=False), approval_id),
        )
        conn.commit()
        count = cur.rowcount
    if count:
        log_event("task_queue", "approval_tasks_blocked", str(approval_id), {"approval_id": approval_id, "count": count, "reason": reason}, 0.75)
        emit_wake_signal("approval_changed", "task_queue", priority=0.78, payload={"approval_id": approval_id, "blocked": count, "reason": reason}, dedupe_key=f"approval_changed:{approval_id}:blocked")
    return count


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def recover_stale_running(max_age_seconds: int = 1800) -> dict[str, int]:
    init_db()
    now = _now_dt()
    threshold = timedelta(seconds=max(1, int(max_age_seconds)))
    recovered = 0
    blocked = 0
    checked = 0
    with connect() as conn:
        rows = conn.execute("SELECT * FROM task_queue WHERE status = 'running'").fetchall()
    for raw in rows:
        checked += 1
        task = dict(raw)
        locked_until = _parse_time(task.get("locked_until"))
        if locked_until is not None and locked_until > now:
            continue
        claimed = _parse_time(task.get("claimed_at"))
        if claimed is not None and now - claimed < threshold:
            continue
        task_id = int(task["id"])
        if int(task.get("attempts") or 0) >= MAX_TASK_ATTEMPTS:
            if finish_task(task_id, "blocked", {"reason": "stale_running_max_attempts"}):
                blocked += 1
        elif requeue_task(task_id, {"reason": "stale_running_recovered"}):
            recovered += 1
    if recovered or blocked:
        log_event("task_queue", "task_doctor_recovered", "stale_running", {"checked": checked, "recovered": recovered, "blocked": blocked}, 0.75)
    return {"checked": checked, "recovered": recovered, "blocked": blocked}


def doctor_tasks(max_age_seconds: int = 1800) -> dict[str, Any]:
    stale = recover_stale_running(max_age_seconds=max_age_seconds)
    counts = task_status_counts()
    return {
        "ok": True,
        "stale_running": stale,
        "counts": counts,
        "max_attempts": MAX_TASK_ATTEMPTS,
    }


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
