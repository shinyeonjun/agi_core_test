from __future__ import annotations

from typing import Any

from agent.config.defaults import project_root
from agent.core.autonomy import current_profile
from agent.core.drives import compute_drives
from agent.core.events import log_event
from agent.core.goals import list_goals
from agent.core.learner import create_reflection
from agent.core.metrics import collect_metrics
from agent.core.policy import PolicyEngine
from agent.core.state import load_state
from agent.lab.proposals import create_action_proposal, list_action_proposals, proposal_status_counts, update_action_proposal_status
from agent.tools.action_log import list_action_runs
from agent.tools.full_device import run_action


def _goal_kind(goal: dict[str, Any] | None, drives: dict[str, float]) -> str:
    if goal:
        kind = str(goal.get("goal_type") or "").strip()
        if kind:
            return kind
    if drives.get("memory_hygiene", 0.0) >= 0.65:
        return "memory_hygiene"
    if drives.get("system_maintenance", 0.0) >= 0.15:
        return "system_maintenance"
    return "lab_experiment"


def _active_goal() -> dict[str, Any] | None:
    for goal in list_goals(limit=10):
        if goal.get("status") in {"active", "proposed", "blocked"}:
            return goal
    return None


def build_action_proposals(goal: dict[str, Any] | None, drives: dict[str, float], state: dict[str, Any]) -> list[dict[str, Any]]:
    kind = _goal_kind(goal, drives)
    goal_id = int(goal["id"]) if goal and goal.get("id") is not None else None
    cwd = str(project_root())
    if kind == "memory_hygiene":
        return []
    if kind == "lab_experiment":
        return [{
            "goal_id": goal_id,
            "command": "./venv/bin/agentctl workspace report --title 'Lab experiment report'",
            "cwd": cwd,
            "reason": "lab_experiment_workspace_report",
            "metadata": {"planner": "rule_based", "goal_type": kind, "state_focus": state.get("current_focus")},
        }]
    return [
        {"goal_id": goal_id, "command": "df -h /", "cwd": cwd, "reason": "system_disk_check", "metadata": {"planner": "rule_based", "goal_type": kind}},
        {"goal_id": goal_id, "command": "free -h", "cwd": cwd, "reason": "system_memory_check", "metadata": {"planner": "rule_based", "goal_type": kind}},
        {"goal_id": goal_id, "command": "systemctl --failed --no-pager", "cwd": cwd, "reason": "system_failed_services_check", "metadata": {"planner": "rule_based", "goal_type": kind}},
    ]


def plan_action_proposals(goal: dict[str, Any] | None = None, *, limit: int = 3) -> list[dict[str, Any]]:
    state = load_state()
    drives = compute_drives()
    profile = current_profile()
    selected_goal = goal or _active_goal()
    rows: list[dict[str, Any]] = []
    for candidate in build_action_proposals(selected_goal, drives, state)[:limit]:
        policy = PolicyEngine(profile=profile).classify_text(candidate["command"])
        if profile != "full_device_lab":
            status = "proposed"
            reason = "profile_not_full_device_lab"
        elif policy.denied or policy.requires_approval:
            status = "blocked"
            reason = policy.denied_reason or "approval_required"
        else:
            status = "approved_by_policy"
            reason = candidate.get("reason")
        metadata = dict(candidate.get("metadata") or {})
        metadata["policy"] = policy.to_dict()
        proposal_id = create_action_proposal(
            goal_id=candidate.get("goal_id"),
            command=candidate["command"],
            cwd=candidate.get("cwd"),
            profile=profile,
            risk_level=policy.risk_level,
            status=status,
            reason=reason,
            metadata=metadata,
        )
        row = dict(candidate)
        row.update({"id": proposal_id, "profile": profile, "risk_level": policy.risk_level, "status": status, "reason": reason, "policy": policy.to_dict()})
        rows.append(row)
    return rows


def run_lab_tick() -> dict[str, Any]:
    profile = current_profile()
    proposals = plan_action_proposals(limit=1)
    if profile != "full_device_lab":
        reflection_id = create_reflection(
            "Lab tick generated proposals but did not execute because profile is not full_device_lab.",
            learned={"profile": profile, "proposals": len(proposals)},
            confidence=0.78,
        )
        result = {
            "status": "skipped",
            "executed": False,
            "reason": "profile_not_full_device_lab",
            "profile": profile,
            "proposal_id": proposals[0]["id"] if proposals else None,
            "reflection_id": reflection_id,
        }
        log_event("lab", "lab_tick_skipped", result["reason"], result, 0.6)
        return result
    approved = next((item for item in proposals if item["status"] == "approved_by_policy"), None)
    if not approved:
        reflection_id = create_reflection(
            "Lab tick produced no executable proposal.",
            learned={"profile": profile, "proposals": len(proposals)},
            confidence=0.72,
        )
        result = {"status": "blocked", "executed": False, "reason": "no_approved_proposal", "profile": profile, "reflection_id": reflection_id}
        log_event("lab", "lab_tick_blocked", result["reason"], result, 0.65)
        return result
    action_result = run_action(approved["command"], cwd=approved.get("cwd"), goal_id=approved.get("goal_id"))
    proposal_status = "executed" if action_result.get("executed") else "blocked"
    update_action_proposal_status(int(approved["id"]), proposal_status, str(action_result.get("reason") or action_result.get("status")))
    reflection_id = create_reflection(
        "Lab tick executed one approved local action and recorded the result.",
        goal_id=approved.get("goal_id"),
        learned={"proposal_id": approved["id"], "action_id": action_result.get("id"), "status": action_result.get("status")},
        confidence=0.8,
    )
    result = {
        "status": action_result.get("status"),
        "executed": bool(action_result.get("executed")),
        "profile": profile,
        "proposal_id": approved["id"],
        "action_id": action_result.get("id"),
        "returncode": action_result.get("returncode"),
        "reflection_id": reflection_id,
    }
    log_event("lab", "lab_tick_executed", approved["command"], result, 0.75)
    return result


def run_lab_tick_if_enabled(*, notify: bool = False) -> dict[str, Any]:
    profile = current_profile()
    if profile != "full_device_lab":
        result = {
            "status": "skipped",
            "executed": False,
            "reason": "profile_not_full_device_lab",
            "profile": profile,
            "proposal_id": None,
            "action_id": None,
        }
        log_event("lab", "lab_tick_timer_skipped", result["reason"], result, 0.45)
        return result

    result = run_lab_tick()
    if notify and result.get("action_id"):
        try:
            from agent.bridge.reports import notify_action

            result["notification"] = notify_action(int(result["action_id"]))
        except Exception as exc:  # pragma: no cover - defensive timer boundary
            result["notification"] = {"ok": False, "error": type(exc).__name__}
            log_event("discord", "action_update_notify_failed", type(exc).__name__, result["notification"], 0.55)
    return result


def lab_report(limit: int = 10) -> dict[str, Any]:
    actions = list_action_runs(limit)
    proposals = list_action_proposals(limit)
    completed = sum(1 for row in actions if row.get("status") == "completed")
    return {
        "profile": current_profile(),
        "actions_total": len(actions),
        "actions_completed": completed,
        "actions_blocked": sum(1 for row in actions if row.get("status") == "blocked"),
        "proposal_status_counts": proposal_status_counts(),
        "recent_actions": [
            {"id": row.get("id"), "status": row.get("status"), "profile": row.get("profile"), "risk_level": row.get("risk_level"), "summary": row.get("result_summary")}
            for row in actions
        ],
        "recent_proposals": [
            {"id": row.get("id"), "status": row.get("status"), "profile": row.get("profile"), "risk_level": row.get("risk_level"), "reason": row.get("reason")}
            for row in proposals
        ],
        "metrics": collect_metrics(),
    }
