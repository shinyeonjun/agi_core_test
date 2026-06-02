from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db
from agent.core.project_execution import get_project_plan
from agent.core.task_lifecycle import task_lifecycle_summary
from agent.core.task_queue import list_tasks

RUNNING_STATES = {"running"}
WAITING_STATES = {"queued", "waiting_approval", "planned"}
TERMINAL_STATES = {"done", "blocked", "skipped", "completed", "timeout", "failed"}


def _decode_json(value: object, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _age_seconds(value: object) -> int | None:
    if not value:
        return None
    try:
        created = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return int(max(0.0, (datetime.now(created.tzinfo or KST) - created).total_seconds()))


def _plan_for_task(task: dict[str, Any]) -> dict[str, Any] | None:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else _decode_json(task.get("payload_json"), {})
    plan_id = payload.get("project_plan_id") if isinstance(payload, dict) else None
    if plan_id is not None:
        return get_project_plan(int(plan_id))
    goal_id = task.get("goal_id")
    if goal_id is None:
        return None
    init_db()
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM project_execution_plans WHERE task_id = ? OR goal_id = ? ORDER BY id DESC LIMIT 1",
            (task.get("id"), goal_id),
        ).fetchone()
    return get_project_plan(int(row["id"])) if row else None


def _plan_progress(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {"plan_id": None, "plan_status": None, "step": None, "progress": None, "failure_category": None}
    steps = plan.get("steps") or []
    done = len([step for step in steps if step.get("status") == "done"])
    blocked = len([step for step in steps if step.get("status") == "blocked"])
    current = next((step for step in steps if step.get("status") == "running"), None)
    if current is None:
        current = next((step for step in steps if step.get("status") == "pending"), None)
    if current is None and steps:
        current = steps[-1]
    return {
        "plan_id": plan.get("id"),
        "plan_status": plan.get("status"),
        "step": {
            "index": current.get("step_index") if current else None,
            "title": current.get("title") if current else None,
            "status": current.get("status") if current else None,
            "kind": current.get("task_kind") if current else None,
        } if current else None,
        "progress": {
            "done": done,
            "blocked": blocked,
            "total": len(steps),
            "ratio": round(done / len(steps), 4) if steps else 0.0,
        },
        "failure_category": (plan.get("result") or {}).get("failure_category"),
    }


def _state_group(status: str) -> str:
    if status in RUNNING_STATES:
        return "running"
    if status in WAITING_STATES:
        return "waiting"
    if status in TERMINAL_STATES:
        return "terminal"
    return "unknown"


def _next_for_task(task: dict[str, Any], plan: dict[str, Any] | None) -> str:
    status = str(task.get("status") or "")
    if status == "queued" and task.get("queue_type") == "user":
        return "사용자 작업 큐에서 즉시 처리 대기 중"
    if status == "queued":
        return "자율 lab tick의 다음 주기에서 처리 예정"
    if status == "waiting_approval":
        return f"승인 #{task.get('approval_id')} 처리 대기 중"
    if status == "running":
        progress = _plan_progress(plan)
        step = progress.get("step") or {}
        return f"현재 단계 진행 중: {step.get('title') or '작업 실행'}"
    result = task.get("result") if isinstance(task.get("result"), dict) else _decode_json(task.get("result_json"), {})
    if status == "blocked":
        return f"실패 원인 확인 필요: {(result or {}).get('reason') or (result or {}).get('status') or 'unknown'}"
    return "후속 조치 없음"


def _task_process(task: dict[str, Any]) -> dict[str, Any]:
    plan = _plan_for_task(task)
    lifecycle = task_lifecycle_summary(task, limit=20)
    result = task.get("result") if isinstance(task.get("result"), dict) else _decode_json(task.get("result_json"), {})
    status = str(task.get("status") or "unknown")
    progress = _plan_progress(plan)
    return {
        "pid": f"task:{task.get('id')}",
        "kind": "task",
        "state": status,
        "state_group": _state_group(status),
        "queue": task.get("queue_type"),
        "title": task.get("title"),
        "goal_id": task.get("goal_id"),
        "task_id": task.get("id"),
        "priority": task.get("priority"),
        "attempts": task.get("attempts"),
        "approval_id": task.get("approval_id"),
        "age_seconds": _age_seconds(task.get("created_at")),
        "updated_at": task.get("updated_at"),
        "locked_by": task.get("locked_by"),
        "plan": progress,
        "lifecycle": {
            "last_phase": lifecycle.get("last_phase"),
            "last_label": lifecycle.get("last_label"),
            "missing_phases": lifecycle.get("missing_phases"),
        },
        "result_status": (result or {}).get("status"),
        "failure_category": progress.get("failure_category") or (result or {}).get("failure_category"),
        "next": _next_for_task(task, plan),
    }


def _orphan_project_process(plan: dict[str, Any]) -> dict[str, Any]:
    status = str(plan.get("status") or "unknown")
    progress = _plan_progress(plan)
    return {
        "pid": f"project:{plan.get('id')}",
        "kind": "project",
        "state": status,
        "state_group": _state_group(status),
        "queue": None,
        "title": plan.get("title"),
        "goal_id": plan.get("goal_id"),
        "task_id": plan.get("task_id"),
        "priority": plan.get("priority"),
        "attempts": None,
        "approval_id": None,
        "age_seconds": _age_seconds(plan.get("created_at")),
        "updated_at": plan.get("updated_at"),
        "locked_by": None,
        "plan": progress,
        "lifecycle": {},
        "result_status": (plan.get("result") or {}).get("status"),
        "failure_category": progress.get("failure_category"),
        "next": "연결된 task 없음. goal sync 또는 run-user 상태 확인 필요",
    }


def list_processes(limit: int = 20, state: str | None = None) -> list[dict[str, Any]]:
    init_db()
    tasks = [_task_process(task) for task in list_tasks(limit=max(limit * 2, 20))]
    seen_plan_ids = {item.get("plan", {}).get("plan_id") for item in tasks if item.get("plan")}
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id FROM project_execution_plans
            WHERE status IN ('planned', 'running', 'blocked')
            ORDER BY status = 'running' DESC, priority DESC, id DESC
            LIMIT ?
            """,
            (max(limit, 20),),
        ).fetchall()
    projects = []
    for row in rows:
        plan_id = int(row["id"])
        if plan_id in seen_plan_ids:
            continue
        plan = get_project_plan(plan_id)
        if plan:
            projects.append(_orphan_project_process(plan))
    items = [*tasks, *projects]
    if state:
        items = [item for item in items if item.get("state") == state or item.get("state_group") == state]
    items.sort(key=lambda item: (item["state_group"] != "running", item["state_group"] != "waiting", -(float(item.get("priority") or 0.0)), item.get("pid") or ""))
    return items[:limit]


def get_process(pid: str) -> dict[str, Any] | None:
    if pid.startswith("task:"):
        task_id = int(pid.split(":", 1)[1])
        task = next((row for row in list_tasks(limit=500) if int(row.get("id", -1)) == task_id), None)
        return _task_process(task) if task else None
    if pid.startswith("project:"):
        plan = get_project_plan(int(pid.split(":", 1)[1]))
        return _orphan_project_process(plan) if plan else None
    return None


def process_snapshot(limit: int = 20) -> dict[str, Any]:
    items = list_processes(limit=limit)
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("state_group") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"created_at": now_kst(), "counts": counts, "items": items}
