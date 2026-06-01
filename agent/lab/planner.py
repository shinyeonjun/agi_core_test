from __future__ import annotations

from typing import Any

from agent.config.defaults import project_root
from agent.core.autonomy import current_profile
from agent.core.drives import compute_drives
from agent.core.events import log_event
from agent.core.goals import goal_metadata, mark_goal_done, update_goal_metadata
from agent.core.goal_generator import generate_goal_candidates, meaningful_open_goals
from agent.core.learner import create_reflection
from agent.core.metrics import collect_metrics
from agent.core.policy import PolicyEngine
from agent.core.state import load_state
from agent.lab.proposals import create_action_proposal, has_recent_goal_command, list_action_proposals, normalize_command, proposal_status_counts, update_action_proposal_status
from agent.tools.action_log import list_action_runs
from agent.tools.full_device import run_action
from agent.workspace.executor import create_project_spec, create_status_report, write_text_artifact


SYSTEM_OBSERVATION_SEQUENCE = [
    {"step": "disk", "command": "df -h /", "reason": "system_disk_check"},
    {"step": "memory", "command": "free -h", "reason": "system_memory_check"},
    {"step": "swap", "command": "swapon --show", "reason": "system_swap_check"},
    {"step": "zram", "command": "zramctl", "reason": "system_zram_check"},
    {"step": "failed_services", "command": "systemctl --failed --no-pager", "reason": "system_failed_services_check"},
    {"step": "report", "command": "./venv/bin/agentctl workspace report --title 'Orange Pi observation report'", "reason": "system_observation_report"},
]

ARTIFACT_ONLY_GOAL_TYPES = {
    "workspace_experiment",
    "self_improvement_proposal",
    "project_incubation",
    "research_note",
    "skill_review",
    "memory_cleanup",
}


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
    goals = meaningful_open_goals(limit=50)
    return goals[0] if goals else None


def _completed_steps_from_history(goal_id: int | None) -> set[str]:
    if goal_id is None:
        return set()
    completed: set[str] = set()
    command_to_step = {normalize_command(item["command"]): item["step"] for item in SYSTEM_OBSERVATION_SEQUENCE}
    for row in list_action_proposals(200):
        if row.get("goal_id") == goal_id and row.get("status") == "executed":
            step = command_to_step.get(normalize_command(str(row.get("command") or "")))
            if step:
                completed.add(step)
    for row in list_action_runs(200):
        if row.get("goal_id") == goal_id and row.get("status") in {"completed", "timeout"}:
            step = command_to_step.get(normalize_command(str(row.get("command_json") or "")))
            if step:
                completed.add(step)
    return completed


def _goal_completed_steps(goal: dict[str, Any] | None) -> set[str]:
    metadata = goal_metadata(goal)
    completed = set(str(step) for step in metadata.get("completed_steps", []) if step)
    goal_id = int(goal["id"]) if goal and goal.get("id") is not None else None
    return completed | _completed_steps_from_history(goal_id)


def _persist_goal_step(goal: dict[str, Any] | None, step: str) -> bool:
    if not goal or goal.get("id") is None:
        return False
    metadata = goal_metadata(goal)
    sequence = metadata.get("sequence") or [item["step"] for item in SYSTEM_OBSERVATION_SEQUENCE]
    completed = list(dict.fromkeys([*metadata.get("completed_steps", []), step]))
    metadata.update({"sequence": sequence, "completed_steps": completed})
    done = all(item in completed for item in sequence)
    return update_goal_metadata(int(goal["id"]), metadata, status="done" if done else None)


def _proposal_for_step(goal_id: int | None, kind: str, step: dict[str, str], state: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_id": goal_id,
        "command": step["command"],
        "cwd": str(project_root()),
        "reason": step["reason"],
        "metadata": {"planner": "rule_based", "goal_type": kind, "step": step["step"], "state_focus": state.get("current_focus")},
    }


def build_action_proposals(goal: dict[str, Any] | None, drives: dict[str, float], state: dict[str, Any]) -> list[dict[str, Any]]:
    kind = _goal_kind(goal, drives)
    goal_id = int(goal["id"]) if goal and goal.get("id") is not None else None
    cwd = str(project_root())
    if kind == "memory_hygiene":
        return []
    if kind == "system_observation":
        completed_steps = _goal_completed_steps(goal)
        for step in SYSTEM_OBSERVATION_SEQUENCE:
            if step["step"] not in completed_steps:
                return [_proposal_for_step(goal_id, kind, step, state)]
        if goal_id is not None:
            mark_goal_done(goal_id)
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
    for candidate in build_action_proposals(selected_goal, drives, state):
        policy = PolicyEngine(profile=profile).classify_text(candidate["command"])
        duplicate = has_recent_goal_command(candidate.get("goal_id"), candidate["command"])
        if duplicate:
            status = "rejected"
            reason = "duplicate_recent_action"
        elif profile != "full_device_lab":
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
        if status == "approved_by_policy" and len([item for item in rows if item["status"] == "approved_by_policy"]) >= limit:
            break
        if len(rows) >= limit and status != "rejected":
            break
    return rows


def _artifact_content(title: str, goal: dict[str, Any], metrics: dict[str, Any], drives: dict[str, Any]) -> str:
    return "\n".join([
        f"# {title}",
        "",
        f"goal_id: {goal.get('id')}",
        f"goal_type: {goal.get('goal_type')}",
        f"title: {goal.get('title')}",
        "",
        "## Summary",
        "",
        str(goal.get("description") or "Safe proposal-only lab artifact."),
        "",
        "## Current Metrics",
        "",
        *[f"- {key}: {value}" for key, value in sorted(metrics.items())],
        "",
        "## Drives",
        "",
        *[f"- {key}: {value}" for key, value in sorted(drives.items())],
        "",
    ])


def _handle_artifact_goal(goal: dict[str, Any], drives: dict[str, float]) -> dict[str, Any] | None:
    kind = str(goal.get("goal_type") or "")
    if kind not in ARTIFACT_ONLY_GOAL_TYPES:
        return None
    metrics = collect_metrics()
    if kind == "workspace_experiment":
        artifact = create_status_report("Workspace experiment report", metrics=metrics, drives=drives)
    elif kind == "project_incubation":
        artifact = create_project_spec(
            str(goal.get("title") or "Core project candidate"),
            str(goal.get("description") or "Draft a safe workspace-only project candidate."),
            ["Generated by lab planner", "No OS mutation", "No external network action"],
        )
    else:
        titles = {
            "self_improvement_proposal": "Core self-improvement proposal",
            "research_note": "Autonomous goal quality research note",
            "skill_review": "Core skill review note",
            "memory_cleanup": "Memory cleanup review",
        }
        artifact = write_text_artifact(
            "reports",
            f"lab-{kind}-{goal.get('id')}.md",
            _artifact_content(titles.get(kind, "Lab artifact"), goal, metrics, drives),
            kind,
            titles.get(kind, "Lab artifact"),
            {"source": "lab_planner", "goal_id": goal.get("id"), "goal_type": kind},
        )
    mark_goal_done(int(goal["id"]))
    reflection_id = create_reflection(
        "Lab tick created a workspace artifact for a non-shell goal and marked it done.",
        goal_id=goal.get("id"),
        learned={"artifact_id": artifact.get("id"), "goal_type": kind},
        confidence=0.82,
    )
    result = {
        "status": "artifact_created",
        "executed": False,
        "profile": current_profile(),
        "goal_id": goal.get("id"),
        "artifact_id": artifact.get("id"),
        "artifact_type": artifact.get("artifact_type"),
        "reflection_id": reflection_id,
    }
    log_event("lab", "lab_artifact_goal_completed", kind, result, 0.72)
    return result


def run_lab_tick() -> dict[str, Any]:
    profile = current_profile()
    selected_goal = _active_goal()
    artifact_result = _handle_artifact_goal(selected_goal, compute_drives()) if profile == "full_device_lab" and selected_goal else None
    if artifact_result:
        return artifact_result
    proposals = plan_action_proposals(selected_goal, limit=1)
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
    if action_result.get("executed") and approved.get("metadata", {}).get("step"):
        _persist_goal_step(selected_goal, str(approved["metadata"]["step"]))
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
    if not meaningful_open_goals():
        generation = generate_goal_candidates(dry_run=False)
        if generation.get("created_goal_id"):
            result = {
                "status": "goal_generated",
                "executed": False,
                "reason": "generated_goal_created",
                "profile": profile,
                "generated_goal_id": generation.get("created_goal_id"),
                "candidate_id": (generation.get("candidates") or [{}])[0].get("id"),
                "proposal_id": None,
                "action_id": None,
            }
            log_event("lab", "lab_tick_goal_generated", result["reason"], result, 0.7)
            return result
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
