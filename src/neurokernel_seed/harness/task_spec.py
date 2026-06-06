from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .action_catalog import ActionDefinition, RiskLevel, default_action_catalog

TaskMode = Literal["dry_run", "dry_run_or_readonly", "readonly"]


class TaskSpecError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    goal: str
    target: str
    context: dict[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    blocked_actions: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    risk_level: RiskLevel = "low"
    requires_approval: bool = False
    timeout_seconds: int = 30
    mode: TaskMode = "dry_run_or_readonly"
    rollback_plan: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_actions"] = list(self.allowed_actions)
        payload["blocked_actions"] = list(self.blocked_actions)
        payload["success_criteria"] = list(self.success_criteria)
        return payload


ALLOWED_TASK_FIELDS = set(TaskSpec.__dataclass_fields__)
ALLOWED_TARGETS = {"local", "orangepi5"}
RISK_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "forbidden": 4}


def task_spec_from_dict(data: dict[str, Any], *, catalog: dict[str, ActionDefinition] | None = None) -> TaskSpec:
    catalog = catalog or default_action_catalog()
    unknown_fields = sorted(set(data) - ALLOWED_TASK_FIELDS)
    if unknown_fields:
        raise TaskSpecError(f"unknown task fields: {unknown_fields}")
    task_id = str(data.get("task_id") or f"task_{uuid.uuid4().hex[:12]}")
    goal = str(data.get("goal") or "").strip()
    if not goal:
        raise TaskSpecError("goal is required")
    target = str(data.get("target") or "local")
    if target not in ALLOWED_TARGETS:
        raise TaskSpecError(f"target is not allowed: {target}")
    allowed_actions = _string_tuple(data.get("allowed_actions") or ())
    blocked_actions = _string_tuple(data.get("blocked_actions") or ())
    success_criteria = _string_tuple(data.get("success_criteria") or ())
    for action_id in set(allowed_actions + blocked_actions):
        if action_id not in catalog:
            raise TaskSpecError(f"unknown action: {action_id}")
    overlap = sorted(set(allowed_actions) & set(blocked_actions))
    if overlap:
        raise TaskSpecError(f"blocked actions cannot also be allowed: {overlap}")
    risk_level = str(data.get("risk_level") or "low")
    if risk_level not in RISK_ORDER:
        raise TaskSpecError(f"unknown risk_level: {risk_level}")
    requires_approval = bool(data.get("requires_approval", False))
    if risk_level in {"medium", "high", "forbidden"} and not requires_approval:
        raise TaskSpecError(f"{risk_level} task requires approval")
    timeout_seconds = int(data.get("timeout_seconds", 30))
    if timeout_seconds <= 0 or timeout_seconds > 300:
        raise TaskSpecError("timeout_seconds must be between 1 and 300")
    mode = str(data.get("mode") or "dry_run_or_readonly")
    if mode not in {"dry_run", "dry_run_or_readonly", "readonly"}:
        raise TaskSpecError(f"unknown mode: {mode}")
    context = data.get("context") or {}
    if not isinstance(context, dict):
        raise TaskSpecError("context must be an object")
    rollback_plan = data.get("rollback_plan")
    if rollback_plan is not None:
        rollback_plan = str(rollback_plan)
    return TaskSpec(
        task_id=task_id,
        goal=goal,
        target=target,
        context=context,
        allowed_actions=allowed_actions,
        blocked_actions=blocked_actions,
        success_criteria=success_criteria,
        risk_level=risk_level,  # type: ignore[arg-type]
        requires_approval=requires_approval,
        timeout_seconds=timeout_seconds,
        mode=mode,  # type: ignore[arg-type]
        rollback_plan=rollback_plan,
    )


def validate_task_spec(data: dict[str, Any], *, catalog: dict[str, ActionDefinition] | None = None) -> dict[str, Any]:
    spec = task_spec_from_dict(data, catalog=catalog)
    return {"accepted": True, "task_spec": spec.as_dict()}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TaskSpecError("action and criteria fields must be arrays")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise TaskSpecError("duplicate values are not allowed")
    return result

