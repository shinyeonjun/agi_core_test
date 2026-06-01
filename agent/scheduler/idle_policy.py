from __future__ import annotations

from agent.core.cooldown import is_ready, mark
from agent.core.drives import compute_drives
from agent.core.events import log_event, mark_events_processed
from agent.core.goals import count_open_goals, create_goal, mark_goal_done
from agent.core.metrics import collect_metrics
from agent.core.self_map import refresh_self_map
from agent.workspace.executor import create_status_report

IDLE_CHECK_TITLE = "Idle tick status check"


def run_idle_policy() -> dict[str, object]:
    drives = compute_drives()
    open_goals = count_open_goals()
    processed = mark_events_processed(limit=50)
    goal_id = None
    workspace_artifact_id = None
    self_map_id = None
    skipped_reason = None
    self_map_ready, self_map_wait = is_ready("self_map_refresh", 1800)
    if self_map_ready:
        try:
            self_map = refresh_self_map()
            self_map_id = int(self_map["id"])
            mark("self_map_refresh", 1800, {"self_map_id": self_map_id, "changed": self_map["changed"]})
        except Exception as exc:  # pragma: no cover - defensive timer boundary
            log_event("self_map", "self_map_refresh_failed", type(exc).__name__, {"error": str(exc)[:300]}, 0.65)
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
    if skipped_reason is None and not self_map_ready:
        skipped_reason = f"cooldown:self_map_refresh:{self_map_wait}s"
    return {"drives": drives, "open_goals": open_goals, "created_goal_id": goal_id, "workspace_artifact_id": workspace_artifact_id, "self_map_id": self_map_id, "processed_events": processed, "skipped_reason": skipped_reason}
