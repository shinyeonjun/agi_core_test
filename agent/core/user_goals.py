from __future__ import annotations

import re
from typing import Any

from agent.core.goals import create_goal
from agent.core.policy import PolicyEngine
from agent.language.engine import interpret_user_message
from agent.language.fallback_rule import classify_user_goal_kind_rule


def is_user_goal_request(text: str, *, interpretation: dict[str, Any] | None = None) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("!"):
        return False
    interpretation = interpretation or interpret_user_message(cleaned, {"purpose": "user_goal_detection"}, log=False)
    execution = interpretation.get("execution") or {}
    return interpretation.get("intent") in {"task_request", "project_request", "report_request"} and bool(execution.get("requires_action"))


def classify_user_goal_kind(text: str, *, interpretation: dict[str, Any] | None = None) -> str:
    if interpretation:
        target = str(interpretation.get("target") or "")
        if target in {"project_spec", "report", "improvement_plan", "workspace_experiment", "task_note"}:
            return target
        suggested = (interpretation.get("execution") or {}).get("suggested_queue_type")
        if suggested in {"project_spec", "report", "improvement_plan", "workspace_experiment", "task_note"}:
            return str(suggested)
    return classify_user_goal_kind_rule(text)


def _title_from_text(text: str) -> str:
    title = re.sub(r"\s+", " ", text.strip())
    return title[:90] if len(title) > 90 else title


def maybe_create_user_goal(text: str, *, source_event_id: int | None = None, metadata: dict[str, Any] | None = None, interpretation: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not is_user_goal_request(text, interpretation=interpretation):
        return None

    policy = PolicyEngine().classify_decision(text, action_type="user_directive")
    task_kind = classify_user_goal_kind(text, interpretation=interpretation)
    if policy.denied:
        status = "blocked"
    elif policy.requires_approval:
        status = "waiting_approval"
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
    return {
        "id": goal_id,
        "status": status,
        "goal_type": "user_directed",
        "title": _title_from_text(text),
        "task_kind": task_kind,
        "risk_level": policy.risk_level,
        "requires_approval": policy.requires_approval,
        "denied": policy.denied,
        "reason": policy.reason,
    }
