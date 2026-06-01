from __future__ import annotations

from agent.core.cooldown import is_ready, mark
from agent.core.drives import compute_drives
from agent.core.events import log_event, mark_events_processed
from agent.core.goals import count_open_goals, create_goal, mark_goal_done
from agent.core.metrics import collect_metrics
from agent.workspace.executor import create_status_report

IDLE_CHECK_TITLE = "Idle tick status check"


def run_idle_policy() -> dict[str, object]:
    drives = compute_drives()
    open_goals = count_open_goals()
    processed = mark_events_processed(limit=50)
    goal_id = None
    workspace_artifact_id = None
    skipped_reason = None
    ready, wait_seconds = is_ready("memory_cleanup", 3600)
    if open_goals == 0:
        if not ready:
            skipped_reason = f"cooldown:memory_cleanup:{wait_seconds}s"
            log_event("scheduler", "idle_action_skipped", skipped_reason, {"cooldown_key": "memory_cleanup"}, 0.4)
        else:
            goal_id = create_goal(IDLE_CHECK_TITLE, "Check event processing state during idle tick.", goal_type="memory_cleanup", status="active", priority=max(0.3, drives.get("memory_hygiene", 0.0)), risk_level="low", metadata={"drives": drives})
            mark_goal_done(goal_id)
            mark("memory_cleanup", 3600, {"goal_id": goal_id})
    workspace_ready, workspace_wait = is_ready("workspace_status_report", 3600)
    if workspace_ready:
        artifact = create_status_report("Idle workspace status", metrics=collect_metrics(), drives=drives)
        workspace_artifact_id = artifact["id"]
        mark("workspace_status_report", 3600, {"artifact_id": workspace_artifact_id})
    elif skipped_reason is None:
        skipped_reason = f"cooldown:workspace_status_report:{workspace_wait}s"
    return {"drives": drives, "open_goals": open_goals, "created_goal_id": goal_id, "workspace_artifact_id": workspace_artifact_id, "processed_events": processed, "skipped_reason": skipped_reason}
