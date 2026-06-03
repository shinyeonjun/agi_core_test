from __future__ import annotations

import json
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.approvals import ApprovalStore
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goals import create_goal, goal_metadata, update_goal_metadata
from agent.core.policy import PolicyEngine
from agent.core.project_execution import create_project_execution_plan, link_plan_task, needs_project_plan
from agent.core.self_improvement_planner import enqueue_user_self_improvement_request
from agent.core.task_queue import enqueue_task
from agent.language.engine import interpret_user_message
from agent.language.fallback_rule import classify_user_goal_kind_rule


def is_user_goal_request(text: str, *, interpretation: dict[str, Any] | None = None) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("!"):
        return False
    interpretation = interpretation or interpret_user_message(cleaned, {"purpose": "user_goal_detection"}, log=False)
    execution = interpretation.get("execution") or {}
    return interpretation.get("intent") in {"task_request", "project_request", "report_request", "self_improvement_request"} and bool(execution.get("requires_action"))


def classify_user_goal_kind(text: str, *, interpretation: dict[str, Any] | None = None) -> str:
    if interpretation:
        target = str(interpretation.get("target") or "")
        if target in {"project_spec", "report", "improvement_plan", "workspace_experiment", "task_note", "code_change"}:
            return target
        suggested = (interpretation.get("execution") or {}).get("suggested_queue_type")
        if suggested in {"project_spec", "report", "improvement_plan", "workspace_experiment", "task_note", "code_change"}:
            return str(suggested)
    return classify_user_goal_kind_rule(text)


def _title_from_text(text: str) -> str:
    title = re.sub(r"\s+", " ", text.strip())
    return title[:90] if len(title) > 90 else title


TARGET_ID_RE = re.compile(r"#\s*(\d+)|(?:작업|task|goal|목표|!cancel|!remove)\s*(\d+)", re.IGNORECASE)
CANCEL_TOKENS = ("취소", "삭제", "지워", "없애", "빼", "빼줘", "빼달", "정리", "중단", "cancel", "remove", "delete", "drop")
TARGET_CONTEXT_TOKENS = ("목표", "작업", "큐", "goal", "task", "#")
OPEN_TASK_STATUSES = ("queued", "running", "waiting_approval")
OPEN_GOAL_STATUSES = ("proposed", "active", "waiting_approval", "blocked")


def is_cancel_request(text: str) -> bool:
    cleaned = text.strip().lower()
    if not cleaned:
        return False
    if cleaned.startswith(("!cancel", "!remove")):
        return True
    return any(token in cleaned for token in CANCEL_TOKENS) and any(token in cleaned for token in TARGET_CONTEXT_TOKENS)


def _target_id_from_text(text: str) -> int | None:
    match = TARGET_ID_RE.search(text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    return int(raw) if raw and raw.isdigit() else None


def _archive_goal(conn, goal_id: int, *, reason: str, source_event_id: int | None) -> int:
    row = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,)).fetchone()
    if not row:
        return 0
    metadata = goal_metadata(dict(row))
    metadata["cancelled"] = True
    metadata["cancel_reason"] = reason
    metadata["cancel_source_event_id"] = source_event_id
    cur = conn.execute(
        """
        UPDATE goals
        SET status = 'archived', updated_at = ?, metadata_json = ?
        WHERE id = ? AND status IN ({})
        """.format(",".join("?" for _ in OPEN_GOAL_STATUSES)),
        (now_kst(), json.dumps(metadata, ensure_ascii=False), goal_id, *OPEN_GOAL_STATUSES),
    )
    return int(cur.rowcount)


def cancel_goal_or_task_target(text: str, *, source_event_id: int | None = None) -> dict[str, Any] | None:
    if not is_cancel_request(text):
        return None
    target_id = _target_id_from_text(text)
    answer_goal_id = create_goal("Cancel requested target", text, goal_type="answer_user", status="done", priority=0.95, risk_level="low", metadata={"source_event_id": source_event_id, "control_action": "cancel"}, dedupe=False)
    if target_id is None:
        return {
            "id": answer_goal_id,
            "status": "done",
            "goal_type": "answer_user",
            "title": "취소 대상 확인 필요",
            "task_kind": "control",
            "control_action": "cancel",
            "cancel_result": {"status": "missing_target", "reason": "대상 번호가 없음"},
            "denied": False,
        }

    init_db()
    ts = now_kst()
    result: dict[str, Any] = {
        "status": "cancelled",
        "target_id": target_id,
        "target_type": None,
        "cancelled_goal_ids": [],
        "cancelled_task_ids": [],
        "cancelled_tasks": 0,
        "archived_goals": 0,
        "reason": "user_cancelled",
    }
    with connect() as conn:
        task_row = conn.execute("SELECT * FROM task_queue WHERE id = ?", (target_id,)).fetchone()
        goal_id: int | None = None
        target_task_ids: list[int] = []
        if task_row:
            task = dict(task_row)
            goal_id = int(task["goal_id"]) if task.get("goal_id") is not None else None
            result["target_type"] = "task"
            if goal_id is not None:
                rows = conn.execute(
                    """
                    SELECT id FROM task_queue
                    WHERE goal_id = ? AND status IN ({})
                    """.format(",".join("?" for _ in OPEN_TASK_STATUSES)),
                    (goal_id, *OPEN_TASK_STATUSES),
                ).fetchall()
                target_task_ids = [int(row["id"]) for row in rows]
            elif str(task.get("status")) in OPEN_TASK_STATUSES:
                target_task_ids = [target_id]
        else:
            goal_row = conn.execute("SELECT * FROM goals WHERE id = ?", (target_id,)).fetchone()
            if goal_row:
                goal_id = target_id
                result["target_type"] = "goal"
                rows = conn.execute(
                    """
                    SELECT id FROM task_queue
                    WHERE goal_id = ? AND status IN ({})
                    """.format(",".join("?" for _ in OPEN_TASK_STATUSES)),
                    (goal_id, *OPEN_TASK_STATUSES),
                ).fetchall()
                target_task_ids = [int(row["id"]) for row in rows]
        if result["target_type"] is None:
            result["status"] = "not_found"
            result["reason"] = "target_not_found"
        else:
            if target_task_ids:
                cur = conn.execute(
                    """
                    UPDATE task_queue
                    SET status = 'skipped', updated_at = ?, completed_at = ?,
                        locked_until = NULL, locked_by = NULL,
                        result_json = ?
                    WHERE id IN ({})
                    """.format(",".join("?" for _ in target_task_ids)),
                    (ts, ts, json.dumps({"reason": "user_cancelled", "source_event_id": source_event_id, "target_id": target_id}, ensure_ascii=False), *target_task_ids),
                )
                result["cancelled_tasks"] = int(cur.rowcount)
                result["cancelled_task_ids"] = target_task_ids
            if goal_id is not None:
                archived = _archive_goal(conn, goal_id, reason="user_cancelled", source_event_id=source_event_id)
                result["archived_goals"] = archived
                result["cancelled_goal_ids"] = [goal_id] if archived else []
                conn.execute(
                    """
                    UPDATE project_execution_plans
                    SET status = 'blocked', updated_at = ?, result_json = ?
                    WHERE goal_id = ? AND status IN ('planned', 'running', 'blocked')
                    """,
                    (ts, json.dumps({"status": "cancelled", "reason": "user_cancelled", "source_event_id": source_event_id}, ensure_ascii=False), goal_id),
                )
        conn.commit()
    log_event("task_queue", "user_cancelled_target", str(target_id), result, 0.78)
    return {
        "id": answer_goal_id,
        "status": "done",
        "goal_type": "answer_user",
        "title": "취소 처리",
        "task_kind": "control",
        "control_action": "cancel",
        "cancel_result": result,
        "denied": False,
    }


def maybe_create_user_goal(text: str, *, source_event_id: int | None = None, metadata: dict[str, Any] | None = None, interpretation: dict[str, Any] | None = None) -> dict[str, Any] | None:
    cancellation = cancel_goal_or_task_target(text, source_event_id=source_event_id)
    if cancellation:
        return cancellation
    if not is_user_goal_request(text, interpretation=interpretation):
        return None

    if (interpretation or {}).get("intent") == "self_improvement_request":
        result = enqueue_user_self_improvement_request(text, source_event_id=source_event_id, limit=1)
        item = (result.get("created") or [{}])[0]
        ticket = item.get("ticket") or {}
        return {
            "id": item.get("goal_id"),
            "task_id": item.get("task_id"),
            "status": "active",
            "goal_type": "self_improvement_proposal",
            "title": str(ticket.get("title") or "Core self-improvement"),
            "task_kind": "code_change",
            "risk_level": str(ticket.get("risk_level") or "medium"),
            "requires_approval": False,
            "approval_id": None,
            "project_plan_id": None,
            "denied": False,
            "reason": "self_improvement_queued",
            "self_improvement": True,
            "ticket": ticket,
        }

    engine = PolicyEngine()
    policy = engine.classify_decision(text, action_type="user_directive")
    proposal = engine.classify_text(text, action_type="user_directive")
    task_kind = classify_user_goal_kind(text, interpretation=interpretation)
    approval_id: int | None = None
    if policy.denied:
        status = "blocked"
    elif policy.requires_approval:
        status = "waiting_approval"
        approval_id = ApprovalStore().create_approval(proposal)
    else:
        status = "active"

    goal_metadata = {
        "source": "user_directive",
        "source_event_id": source_event_id,
        "task_kind": task_kind,
        "raw_user_text": text,
        "priority_owner": "user",
        "language_interpretation": interpretation or {},
        "policy": policy.to_dict(),
        "approval_id": approval_id,
    }
    goal_metadata.update(metadata or {})
    goal_id = create_goal(
        _title_from_text(text),
        text,
        goal_type="user_directed",
        status=status,
        priority=0.98,
        risk_level=policy.risk_level,
        requires_approval=policy.requires_approval,
        metadata=goal_metadata,
        dedupe=True,
    )
    plan_id: int | None = None
    if status == "active" and needs_project_plan(text, task_kind, interpretation):
        plan = create_project_execution_plan(
            goal_id=goal_id,
            source="user_directive",
            owner="user",
            title=_title_from_text(text),
            objective=text,
            task_kind=task_kind,
            priority=0.98,
        )
        plan_id = int(plan["id"])
        goal_metadata["project_plan_id"] = plan_id
        update_goal_metadata(goal_id, goal_metadata)
    task_status = "blocked" if status == "blocked" else "waiting_approval" if status == "waiting_approval" else "queued"
    task_id = enqueue_task(
        "user",
        goal_id=goal_id,
        task_kind=task_kind,
        title=_title_from_text(text),
        source="discord_user_directive",
        priority=0.98,
        status=task_status,
        approval_id=approval_id,
        payload={"source_event_id": source_event_id, "risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "approval_id": approval_id, "project_plan_id": plan_id},
    )
    if plan_id is not None:
        link_plan_task(plan_id, task_id)
    return {
        "id": goal_id,
        "task_id": task_id,
        "status": status,
        "goal_type": "user_directed",
        "title": _title_from_text(text),
        "task_kind": task_kind,
        "risk_level": policy.risk_level,
        "requires_approval": policy.requires_approval,
        "approval_id": approval_id,
        "project_plan_id": plan_id,
        "denied": policy.denied,
        "reason": policy.reason,
    }
