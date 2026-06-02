from __future__ import annotations

from typing import Any

from agent.config.defaults import project_root
from agent.core.autonomy import current_profile
from agent.core.drives import compute_drives
from agent.core.events import log_event
from agent.core.goals import get_goal, goal_metadata, mark_goal_done, update_goal_metadata
from agent.core.goal_generator import generate_goal_candidates, meaningful_open_goals
from agent.core.learner import create_reflection
from agent.core.metrics import collect_metrics
from agent.core.operating_intelligence import refresh_goal_priorities
from agent.core.policy import PolicyEngine
from agent.core.state import load_state
from agent.core.task_lifecycle import record_task_phase
from agent.core.task_queue import claim_next_task, claim_task, enqueue_task, finish_task, list_tasks, requeue_task, task_status_counts
from agent.lab.proposals import create_action_proposal, has_recent_goal_command, list_action_proposals, normalize_command, proposal_status_counts, update_action_proposal_status
from agent.tools.action_log import list_action_runs
from agent.tools.full_device import run_action
from agent.lab.codex_worker import run_codex_work
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
    "user_directed",
    "workspace_experiment",
    "self_improvement_proposal",
    "project_incubation",
    "research_note",
    "skill_review",
    "memory_cleanup",
}
SHELL_PLANNED_GOAL_TYPES = {"system_observation", "lab_experiment"}


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


def _queue_type_for_goal(goal: dict[str, Any]) -> str:
    metadata = goal_metadata(goal)
    if goal.get("goal_type") == "user_directed" or metadata.get("priority_owner") == "user":
        return "user"
    return "autonomous"


def _task_status_for_goal(goal: dict[str, Any]) -> str:
    status = str(goal.get("status") or "queued")
    if status == "waiting_approval":
        return "waiting_approval"
    if status == "blocked":
        return "blocked"
    return "queued"


def _task_kind_for_goal(goal: dict[str, Any]) -> str:
    metadata = goal_metadata(goal)
    return str(metadata.get("task_kind") or goal.get("goal_type") or "general")


def sync_open_goals_to_tasks(limit: int = 100) -> dict[str, Any]:
    priority_refresh = refresh_goal_priorities(limit=limit)
    synced = 0
    skipped = 0
    for goal in meaningful_open_goals(limit=limit):
        status = _task_status_for_goal(goal)
        if status in {"blocked", "waiting_approval"}:
            skipped += 1
        task_id = enqueue_task(
            _queue_type_for_goal(goal),  # type: ignore[arg-type]
            goal_id=int(goal["id"]),
            task_kind=_task_kind_for_goal(goal),
            title=str(goal.get("title") or "Untitled task"),
            source="goal_sync",
            priority=float(goal.get("priority") or 0.5),
            status=status,  # type: ignore[arg-type]
            payload={"goal_type": goal.get("goal_type"), "goal_status": goal.get("status")},
        )
        if task_id:
            synced += 1
    return {"synced": synced, "skipped": skipped, "priority_refresh": priority_refresh}


def _goal_from_task(task: dict[str, Any] | None) -> dict[str, Any] | None:
    if not task or task.get("goal_id") is None:
        return None
    return get_goal(int(task["goal_id"]))


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
    update_goal_metadata(int(goal["id"]), metadata, status="done" if done else None)
    return done


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
    if kind in ARTIFACT_ONLY_GOAL_TYPES:
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
    log_event("lab", "unknown_goal_shell_fallback_suppressed", kind, {"goal_id": goal_id, "goal_type": kind}, 0.55)
    return []


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


def _handle_user_directed_goal(goal: dict[str, Any], drives: dict[str, float], *, task_id: int | None = None) -> dict[str, Any]:
    metadata = goal_metadata(goal)
    task_kind = str(metadata.get("task_kind") or "task_note")
    user_text = str(metadata.get("raw_user_text") or goal.get("description") or goal.get("title") or "")
    metrics = collect_metrics()
    if task_kind == "code_change":
        result = run_codex_work(user_text, goal_id=int(goal["id"]) if goal.get("id") is not None else None, task_id=task_id)
        if result.get("status") == "codex_work_completed":
            mark_goal_done(int(goal["id"]))
            reflection_id = create_reflection(
                "Codex work worker completed a user-directed code-change task.",
                goal_id=goal.get("id"),
                learned={"artifact_id": result.get("artifact_id"), "task_kind": task_kind, "returncode": result.get("returncode")},
                confidence=0.82,
            )
            result["reflection_id"] = reflection_id
        return {
            **result,
            "profile": current_profile(),
            "goal_id": goal.get("id"),
            "task_kind": task_kind,
            "artifact_type": "codex_work_report",
        }
    if task_kind == "project_spec":
        artifact = create_project_spec(
            str(goal.get("title") or "User requested project"),
            user_text,
            ["User-directed priority", "Workspace-only first pass", "No OS mutation without policy approval"],
        )
    else:
        title = {
            "report": "User requested report",
            "improvement_plan": "User requested improvement plan",
            "workspace_experiment": "User requested workspace experiment plan",
            "task_note": "User requested task note",
        }.get(task_kind, "User requested task note")
        artifact = write_text_artifact(
            "reports",
            f"user-goal-{goal.get('id')}-{task_kind}.md",
            _artifact_content(title, goal, metrics, drives),
            f"user_directed_{task_kind}",
            title,
            {"source": "lab_planner", "goal_id": goal.get("id"), "goal_type": "user_directed", "task_kind": task_kind},
        )
    mark_goal_done(int(goal["id"]))
    reflection_id = create_reflection(
        "Lab tick completed a user-directed goal before autonomous goals.",
        goal_id=goal.get("id"),
        learned={"artifact_id": artifact.get("id"), "task_kind": task_kind, "priority_owner": "user"},
        confidence=0.86,
    )
    result = {
        "status": "user_goal_completed",
        "executed": False,
        "profile": current_profile(),
        "goal_id": goal.get("id"),
        "artifact_id": artifact.get("id"),
        "artifact_type": artifact.get("artifact_type"),
        "task_kind": task_kind,
        "reflection_id": reflection_id,
    }
    log_event("lab", "lab_user_goal_completed", task_kind, result, 0.82)
    return result


def _handle_artifact_goal(goal: dict[str, Any], drives: dict[str, float], *, task_id: int | None = None) -> dict[str, Any] | None:
    kind = str(goal.get("goal_type") or "")
    if kind in SHELL_PLANNED_GOAL_TYPES:
        return None
    if goal.get("status") not in {"active", "proposed"}:
        return None
    if kind == "user_directed":
        return _handle_user_directed_goal(goal, drives, task_id=task_id)
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
        artifact_type = kind if kind in ARTIFACT_ONLY_GOAL_TYPES else "goal_review"
        title = titles.get(kind, "Unsupported goal review")
        artifact = write_text_artifact(
            "reports",
            f"lab-{artifact_type}-{goal.get('id')}.md",
            _artifact_content(title, goal, metrics, drives),
            artifact_type,
            title,
            {"source": "lab_planner", "goal_id": goal.get("id"), "goal_type": kind, "unknown_goal_review": kind not in ARTIFACT_ONLY_GOAL_TYPES},
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


def _finish_claimed_task(task: dict[str, Any], status: str, result: dict[str, Any]) -> dict[str, Any]:
    finish_status = "done" if status in {"user_goal_completed", "artifact_created", "completed", "done", "codex_work_completed"} else "blocked" if status in {"blocked", "failed", "timeout", "codex_work_failed", "codex_work_blocked"} else "skipped"
    result["task_id"] = int(task["id"])
    result["queue_type"] = task.get("queue_type")
    finish_task(int(task["id"]), finish_status, result)
    return result


def _process_claimed_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = int(task["id"])
    queue_type = str(task.get("queue_type") or "")
    record_task_phase(
        task_id,
        "planning",
        "started",
        "작업 목표와 실행 방식을 확인하는 중",
        queue_type=queue_type,
        metadata={"goal_id": task.get("goal_id"), "task_kind": task.get("task_kind")},
    )
    goal = _goal_from_task(task)
    if not goal:
        result = {"status": "blocked", "executed": False, "reason": "goal_not_found", "task_id": task.get("id")}
        record_task_phase(task_id, "verifying", "failed", "연결된 목표를 찾지 못함", queue_type=queue_type, metadata=result)
        finish_task(int(task["id"]), "blocked", result)
        return result

    drives = compute_drives()
    artifact_result = _handle_artifact_goal(goal, drives, task_id=task_id)
    if artifact_result:
        artifact_status = str(artifact_result.get("status") or "artifact_created")
        executing_status = "codex_work_completed" if artifact_status == "codex_work_completed" else "blocked" if artifact_status == "codex_work_blocked" else "artifact_created"
        verifying_status = "passed" if artifact_status in {"codex_work_completed", "artifact_created", "user_goal_completed"} else "blocked"
        record_task_phase(
            task_id,
            "executing",
            executing_status,
            "워크스페이스 산출물을 생성함",
            queue_type=queue_type,
            metadata={"artifact_id": artifact_result.get("artifact_id"), "goal_id": goal.get("id")},
        )
        record_task_phase(
            task_id,
            "verifying",
            verifying_status,
            "산출물 생성과 목표 완료 상태를 확인함",
            queue_type=queue_type,
            metadata={"status": artifact_status, "goal_id": goal.get("id")},
        )
        record_task_phase(
            task_id,
            "learned",
            "recorded",
            "작업 결과를 reflection으로 남김",
            queue_type=queue_type,
            metadata={"reflection_id": artifact_result.get("reflection_id")},
        )
        return _finish_claimed_task(task, str(artifact_result.get("status")), artifact_result)

    proposals = plan_action_proposals(goal, limit=1)
    profile = current_profile()
    if profile != "full_device_lab":
        result = {
            "status": "skipped",
            "executed": False,
            "reason": "profile_not_full_device_lab",
            "profile": profile,
            "proposal_id": proposals[0]["id"] if proposals else None,
            "goal_id": goal.get("id"),
        }
        record_task_phase(task_id, "verifying", "requeued", "현재 모드에서 실행할 수 없어 재대기", queue_type=queue_type, metadata=result)
        requeue_task(int(task["id"]), result)
        log_event("lab", "task_requeued", result["reason"], result, 0.55)
        return {**result, "task_id": int(task["id"]), "queue_type": task.get("queue_type")}

    approved = next((item for item in proposals if item["status"] == "approved_by_policy"), None)
    if not approved:
        reflection_id = create_reflection(
            "Task worker produced no executable proposal.",
            goal_id=goal.get("id"),
            learned={"queue_type": task.get("queue_type"), "task_id": task.get("id"), "proposals": len(proposals)},
            confidence=0.72,
        )
        result = {"status": "blocked", "executed": False, "reason": "no_approved_proposal", "profile": profile, "goal_id": goal.get("id"), "reflection_id": reflection_id}
        record_task_phase(task_id, "verifying", "blocked", "정책을 통과한 실행안이 없어 차단함", queue_type=queue_type, metadata={"proposals": len(proposals), "goal_id": goal.get("id")})
        record_task_phase(task_id, "learned", "recorded", "차단 사유를 reflection으로 남김", queue_type=queue_type, metadata={"reflection_id": reflection_id})
        return _finish_claimed_task(task, "blocked", result)

    record_task_phase(
        task_id,
        "executing",
        "started",
        "승인된 로컬 action을 실행함",
        queue_type=queue_type,
        metadata={"proposal_id": approved["id"], "goal_id": goal.get("id"), "risk_level": approved.get("risk_level")},
    )
    action_result = run_action(approved["command"], cwd=approved.get("cwd"), goal_id=approved.get("goal_id"))
    proposal_status = "executed" if action_result.get("executed") else "blocked"
    update_action_proposal_status(int(approved["id"]), proposal_status, str(action_result.get("reason") or action_result.get("status")))
    goal_done = False
    if action_result.get("executed") and approved.get("metadata", {}).get("step"):
        goal_done = _persist_goal_step(goal, str(approved["metadata"]["step"]))
    record_task_phase(
        task_id,
        "verifying",
        "passed" if action_result.get("executed") else "blocked",
        "action 결과와 목표 진행 상태를 확인함",
        queue_type=queue_type,
        metadata={"action_id": action_result.get("id"), "status": action_result.get("status"), "goal_done": goal_done},
    )
    reflection_id = create_reflection(
        "Task worker executed one approved local action and recorded the result.",
        goal_id=approved.get("goal_id"),
        learned={"queue_type": task.get("queue_type"), "task_id": task.get("id"), "proposal_id": approved["id"], "action_id": action_result.get("id"), "status": action_result.get("status")},
        confidence=0.8,
    )
    record_task_phase(
        task_id,
        "learned",
        "recorded",
        "action 실행 결과를 reflection으로 남김",
        queue_type=queue_type,
        metadata={"reflection_id": reflection_id, "action_id": action_result.get("id")},
    )
    result = {
        "status": action_result.get("status"),
        "executed": bool(action_result.get("executed")),
        "profile": profile,
        "goal_id": goal.get("id"),
        "goal_done": goal_done,
        "proposal_id": approved["id"],
        "action_id": action_result.get("id"),
        "returncode": action_result.get("returncode"),
        "reflection_id": reflection_id,
    }
    if goal_done:
        return _finish_claimed_task(task, "done", result)
    requeue_task(int(task["id"]), result)
    return {**result, "task_id": int(task["id"]), "queue_type": task.get("queue_type")}


def run_user_task(task_id: int) -> dict[str, Any]:
    task = claim_task(task_id)
    if not task:
        existing = next((row for row in list_tasks(limit=50) if int(row.get("id", -1)) == int(task_id)), None)
        return {"status": "skipped", "executed": False, "reason": "task_not_queued", "task_id": task_id, "task": existing}
    if task.get("queue_type") != "user":
        requeue_task(int(task["id"]), {"reason": "not_user_task"})
        return {"status": "skipped", "executed": False, "reason": "not_user_task", "task_id": task_id}
    result = _process_claimed_task(task)
    log_event("lab", "user_task_processed", str(task_id), result, 0.82)
    return result


def run_lab_tick() -> dict[str, Any]:
    profile = current_profile()
    sync_open_goals_to_tasks()
    if profile != "full_device_lab":
        reflection_id = create_reflection(
            "Autonomous worker skipped because profile is not full_device_lab.",
            learned={"profile": profile},
            confidence=0.78,
        )
        result = {
            "status": "skipped",
            "executed": False,
            "reason": "profile_not_full_device_lab",
            "profile": profile,
            "proposal_id": None,
            "reflection_id": reflection_id,
        }
        log_event("lab", "lab_tick_skipped", result["reason"], result, 0.6)
        return result

    task = claim_next_task("autonomous")
    if not task:
        reflection_id = create_reflection(
            "Autonomous worker found no queued autonomous task.",
            learned={"profile": profile},
            confidence=0.72,
        )
        result = {"status": "idle", "executed": False, "reason": "no_autonomous_task", "profile": profile, "reflection_id": reflection_id}
        log_event("lab", "lab_tick_blocked", result["reason"], result, 0.65)
        return result
    result = _process_claimed_task(task)
    log_event("lab", "autonomous_task_processed", str(task.get("id")), result, 0.75)
    return result


def run_lab_tick_if_enabled(*, notify: bool = False) -> dict[str, Any]:
    profile = current_profile()
    autonomous_open_goals = [goal for goal in meaningful_open_goals() if goal.get("goal_type") != "user_directed"]
    autonomous_queued = list_tasks(limit=1, status="queued", queue_type="autonomous")
    if not autonomous_open_goals and not autonomous_queued:
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
        "task_status_counts": task_status_counts(),
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
