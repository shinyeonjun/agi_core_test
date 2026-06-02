from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event

TASK_PHASES = ("queued", "planning", "executing", "verifying", "reporting", "learned")

PHASE_LABELS = {
    "queued": "대기 등록",
    "planning": "계획",
    "executing": "실행",
    "verifying": "검증",
    "reporting": "보고",
    "learned": "학습",
}


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json")
    try:
        row["metadata"] = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        row["metadata"] = {"decode_error": True}
    return row


def record_task_phase(
    task_id: int,
    phase: str,
    status: str,
    summary: str,
    *,
    queue_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    if phase not in TASK_PHASES:
        raise ValueError(f"invalid task phase: {phase}")
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO task_lifecycle_events (
                created_at, task_id, queue_type, phase, status, summary, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), int(task_id), queue_type, phase, status[:40], summary[:300], _json(metadata)),
        )
        conn.commit()
        event_id = int(cur.lastrowid)
    log_event(
        "task_kernel",
        f"task_{phase}",
        summary[:300],
        {"task_id": int(task_id), "phase": phase, "status": status, "queue_type": queue_type, **(metadata or {})},
        0.76 if queue_type == "user" else 0.62,
    )
    return event_id


def list_task_lifecycle(task_id: int, limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM task_lifecycle_events
            WHERE task_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (int(task_id), int(limit)),
        ).fetchall()
    return [_decode(dict(row)) for row in rows]


def latest_task_lifecycle_events(limit: int = 50, queue_type: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM task_lifecycle_events"
    params: list[Any] = []
    if queue_type:
        query += " WHERE queue_type = ?"
        params.append(queue_type)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_decode(dict(row)) for row in rows]


def task_lifecycle_summary(task: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    events = list_task_lifecycle(int(task["id"]), limit=limit)
    completed_phases = list(dict.fromkeys(str(row["phase"]) for row in events))
    last = events[-1] if events else None
    missing = [phase for phase in TASK_PHASES if phase not in completed_phases]
    return {
        "task_id": int(task["id"]),
        "queue_type": task.get("queue_type"),
        "status": task.get("status"),
        "title": task.get("title"),
        "completed_phases": completed_phases,
        "missing_phases": missing,
        "last_phase": last.get("phase") if last else None,
        "last_label": PHASE_LABELS.get(str(last.get("phase"))) if last else None,
        "last_summary": last.get("summary") if last else None,
        "events": events,
    }
