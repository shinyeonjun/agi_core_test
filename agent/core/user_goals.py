from __future__ import annotations

import re
from typing import Any

from agent.core.approvals import ApprovalStore
from agent.core.goals import create_goal, update_goal_metadata
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


def maybe_create_user_goal(text: str, *, source_event_id: int | None = None, metadata: dict[str, Any] | None = None, interpretation: dict[str, Any] | None = None) -> dict[str, Any] | None:
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
