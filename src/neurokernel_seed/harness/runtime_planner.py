from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_catalog import ActionDefinition
from .runtime_policy import RuntimeActionPolicy
from .safety_gate import check_action_safety
from .task_spec import TaskSpec


@dataclass(frozen=True)
class RuntimePlan:
    decisions: list[dict[str, Any]]
    chosen: dict[str, Any] | None
    policy: dict[str, Any]

    @property
    def chosen_action(self) -> str | None:
        return str(self.chosen.get("action_id")) if self.chosen else None


class RuntimeActionPlanner:
    def __init__(self, *, catalog: dict[str, ActionDefinition], runtime_policy: RuntimeActionPolicy):
        self.catalog = catalog
        self.runtime_policy = runtime_policy

    def plan(self, *, task: TaskSpec, candidates: list[str]) -> RuntimePlan:
        decisions = _safety_decisions(task=task, candidates=candidates, catalog=self.catalog)
        deterministic = _first_safe_or_approval_candidate(decisions)
        policy = self.runtime_policy.rank(task=task, decisions=decisions)
        _attach_model_scores(decisions, policy)
        chosen = _model_ranked_choice(decisions, policy) if policy.get("model_used") else None
        return RuntimePlan(decisions=decisions, chosen=chosen or deterministic, policy=policy)


def decision_policy_name(policy: dict[str, Any]) -> str:
    if policy.get("model_used"):
        return "runtime_model_ranked_safety_gated"
    return "deterministic_safety_first"


def policy_decision_summary(policy: dict[str, Any]) -> dict[str, Any]:
    keys = ("model_used", "reason", "detail", "model_path", "schema_version", "top_action", "ranked_actions")
    return {key: policy.get(key) for key in keys if key in policy}


def _safety_decisions(*, task: TaskSpec, candidates: list[str], catalog: dict[str, ActionDefinition]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for action_id in candidates:
        safety = check_action_safety(task, action_id, catalog=catalog)
        decisions.append({"action_id": action_id, "safety": safety.as_dict()})
    return decisions


def _first_safe_or_approval_candidate(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    requires_approval = None
    for item in decisions:
        safety = item.get("safety") if isinstance(item.get("safety"), dict) else {}
        if safety.get("decision") in {"allow", "dry_run_only"}:
            return item
        if safety.get("decision") == "requires_approval" and requires_approval is None:
            requires_approval = item
    return requires_approval


def _attach_model_scores(decisions: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    scores = policy.get("scores") if isinstance(policy.get("scores"), dict) else {}
    for item in decisions:
        action_id = str(item["action_id"])
        item["model_score"] = scores.get(action_id, {"model_used": False, "reason": policy.get("reason")})


def _model_ranked_choice(decisions: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any] | None:
    by_action = {str(item["action_id"]): item for item in decisions}
    for action_id in policy.get("ranked_actions") or []:
        item = by_action.get(str(action_id))
        safety = item.get("safety") if item else {}
        if isinstance(safety, dict) and safety.get("decision") in {"allow", "dry_run_only"}:
            return item
    return None
