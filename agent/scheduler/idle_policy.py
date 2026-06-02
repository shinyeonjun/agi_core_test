from __future__ import annotations

from agent.core.cooldown import is_ready, mark
from agent.core.cognitive_engine import cognitive_growth_snapshot
from agent.core.cognitive_pipeline import maybe_enqueue_growth_task
from agent.core.drives import compute_drives
from agent.core.events import log_event, mark_events_processed
from agent.core.goals import count_open_goals, create_goal, mark_goal_done
from agent.core.memory_intelligence import run_memory_intelligence
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
    memory_intelligence_result = None
    cognitive_growth_result = None
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
    metrics = collect_metrics()
    memory_ready, memory_wait = is_ready("memory_intelligence", 21600)
    if memory_ready and (int(metrics.get("memories_count") or 0) >= 200 or int(metrics.get("reflections_count") or 0) >= 300):
        try:
            memory_intelligence_result = run_memory_intelligence(dry_run=False)
            mark("memory_intelligence", 21600, {
                "created_memories": len((memory_intelligence_result.get("memory") or {}).get("created") or []),
                "promoted_skills": len((memory_intelligence_result.get("skills") or {}).get("promoted") or []),
            })
        except Exception as exc:  # pragma: no cover - defensive timer boundary
            log_event("memory", "memory_intelligence_failed", type(exc).__name__, {"error": str(exc)[:300]}, 0.65)
    elif skipped_reason is None:
        skipped_reason = f"cooldown:memory_intelligence:{memory_wait}s"
    growth_ready, growth_wait = is_ready("cognitive_growth_snapshot", 600)
    if growth_ready:
        try:
            growth_snapshot = cognitive_growth_snapshot(persist=True, limit=5)
            growth_task = maybe_enqueue_growth_task(growth_snapshot)
            cognitive_growth_result = {
                "snapshot_id": growth_snapshot.get("snapshot_id"),
                "mode": (growth_snapshot.get("active_inference") or {}).get("mode"),
                "free_energy": (growth_snapshot.get("active_inference") or {}).get("free_energy"),
                "top_curiosity": (growth_snapshot.get("curiosity") or [{}])[0].get("topic"),
                "task": growth_task,
            }
            mark("cognitive_growth_snapshot", 600, cognitive_growth_result)
        except Exception as exc:  # pragma: no cover - defensive timer boundary
            log_event("cognition", "cognitive_growth_failed", type(exc).__name__, {"error": str(exc)[:300]}, 0.65)
    elif skipped_reason is None:
        skipped_reason = f"cooldown:cognitive_growth_snapshot:{growth_wait}s"
    return {
        "drives": drives,
        "open_goals": open_goals,
        "created_goal_id": goal_id,
        "workspace_artifact_id": workspace_artifact_id,
        "self_map_id": self_map_id,
        "memory_intelligence": memory_intelligence_result,
        "cognitive_growth": cognitive_growth_result,
        "processed_events": processed,
        "skipped_reason": skipped_reason,
    }
