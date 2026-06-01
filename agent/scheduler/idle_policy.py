from __future__ import annotations

from agent.core.drives import compute_drives
from agent.core.events import mark_events_processed
from agent.core.goals import count_open_goals, create_goal, has_recent_goal, mark_goal_done

IDLE_CHECK_TITLE = "\uc785\ub825 \uc5c6\ub294 tick \uc0c1\ud0dc \uc810\uac80"


def run_idle_policy() -> dict[str, object]:
    drives = compute_drives()
    open_goals = count_open_goals()
    processed = mark_events_processed(limit=50)
    goal_id = None
    skipped_reason = None

    if open_goals == 0:
        if has_recent_goal("memory_cleanup", IDLE_CHECK_TITLE, within_minutes=60):
            skipped_reason = "cooldown:memory_cleanup"
        else:
            goal_id = create_goal(
                title=IDLE_CHECK_TITLE,
                description="\uc785\ub825 \uc5c6\ub294 tick\uc5d0\uc11c event \ucc98\ub9ac \uc0c1\ud0dc\ub97c \uc810\uac80\ud55c\ub2e4.",
                goal_type="memory_cleanup",
                status="active",
                priority=max(0.3, drives.get("memory_hygiene", 0.0)),
                risk_level="low",
                metadata={"drives": drives},
            )
            mark_goal_done(goal_id)

    return {
        "drives": drives,
        "open_goals": open_goals,
        "created_goal_id": goal_id,
        "processed_events": processed,
        "skipped_reason": skipped_reason,
    }
