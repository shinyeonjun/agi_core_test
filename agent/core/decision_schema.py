from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.core.failure import failure_report


@dataclass(frozen=True)
class DecisionSchema:
    version: str
    kind: str
    intent: str
    target: str | None
    selected_goal_id: int | None
    risk_level: str
    requires_approval: bool
    memory_ids: list[int] = field(default_factory=list)
    skill_names: list[str] = field(default_factory=list)
    tool_plan: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    failure_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ids(rows: list[dict[str, Any]]) -> list[int]:
    result = []
    for row in rows:
        try:
            result.append(int(row["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _skill_names(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("name")) for row in rows if row.get("name")]


def build_decision_schema(decision: dict[str, Any]) -> dict[str, Any]:
    interpretation = decision.get("language_interpretation") or {}
    policy = decision.get("policy_summary") or {}
    user_goal = decision.get("user_directed_goal") or {}
    project_plan = user_goal.get("project_plan") or {}
    tool_routes = (decision.get("routing") or {}).get("tools") or []
    schema = DecisionSchema(
        version=str(decision.get("version") or "unknown"),
        kind=str(decision.get("kind") or "unknown"),
        intent=str(interpretation.get("intent") or "unknown"),
        target=interpretation.get("target"),
        selected_goal_id=decision.get("selected_goal_id"),
        risk_level=str(policy.get("risk_level") or decision.get("risk_level") or "low"),
        requires_approval=bool(policy.get("requires_approval")),
        memory_ids=_ids(decision.get("selected_memories") or decision.get("relevant_memories") or []),
        skill_names=_skill_names(decision.get("relevant_skills") or []),
        tool_plan=tool_routes,
        plan={
            "user_goal_created": bool(decision.get("user_goal_created")),
            "task_id": user_goal.get("task_id"),
            "project_plan_id": user_goal.get("project_plan_id"),
            "project_plan_status": project_plan.get("status"),
        },
        verification={
            "must_include": decision.get("must_include") or [],
            "must_not_include": decision.get("must_not_include") or [],
            "renderer": decision.get("renderer"),
            "answer_contract_kind": (decision.get("answer_contract") or {}).get("kind") if isinstance(decision.get("answer_contract"), dict) else None,
        },
        failure_policy=failure_report(policy),
    )
    return schema.to_dict()
