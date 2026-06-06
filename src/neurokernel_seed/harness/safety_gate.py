from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from .action_catalog import ActionDefinition, RiskLevel, default_action_catalog
from .task_spec import TaskSpec

SafetyDecisionType = Literal["allow", "deny", "requires_approval", "dry_run_only"]


@dataclass(frozen=True)
class SafetyDecision:
    decision: SafetyDecisionType
    risk_level: RiskLevel
    requires_approval: bool
    reason: str
    policy_hits: tuple[str, ...] = field(default_factory=tuple)
    approval_scope: str | None = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["policy_hits"] = list(self.policy_hits)
        return payload


def check_action_safety(
    task: TaskSpec,
    action_id: str,
    *,
    approved: bool = False,
    catalog: dict[str, ActionDefinition] | None = None,
) -> SafetyDecision:
    catalog = catalog or default_action_catalog()
    if action_id not in catalog:
        return SafetyDecision("deny", "forbidden", False, f"unknown action: {action_id}", ("deny_unknown_action",))
    action = catalog[action_id]
    if action.status != "active":
        return SafetyDecision("deny", action.risk_level, action.requires_approval, f"action is not active: {action.status}", ("deny_inactive_action",))
    if action_id in task.blocked_actions:
        return SafetyDecision("deny", action.risk_level, action.requires_approval, "action is explicitly blocked by task", ("deny_blocked_action",))
    if task.allowed_actions and action_id not in task.allowed_actions:
        return SafetyDecision("deny", action.risk_level, action.requires_approval, "action is not in allowed_actions", ("deny_default",))
    if task.target not in action.allowed_targets:
        return SafetyDecision("deny", action.risk_level, action.requires_approval, f"target is not allowed for action: {task.target}", ("deny_target",))
    if action.risk_level == "forbidden":
        return SafetyDecision("deny", "forbidden", True, "forbidden action in Harness v0", ("deny_forbidden",))
    if task.mode == "dry_run":
        return SafetyDecision("dry_run_only", action.risk_level, action.requires_approval, "task is dry-run only", ("dry_run_mode",))
    if action.requires_approval or action.risk_level in {"medium", "high"} or task.requires_approval:
        if approved:
            return SafetyDecision("allow", action.risk_level, True, "approved action", ("allow_after_approval",), "single_action")
        return SafetyDecision("requires_approval", action.risk_level, True, "action requires approval", ("approval_required",), "single_action")
    if action.side_effect:
        return SafetyDecision("requires_approval", action.risk_level, True, "side-effect action requires approval", ("side_effect_requires_approval",), "single_action")
    return SafetyDecision("allow", action.risk_level, False, "read-only action in allowed_actions", ("allow_readonly",))
