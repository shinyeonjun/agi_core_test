from __future__ import annotations

from agent.core.database import init_db
from agent.core.events import log_event
from agent.core.learner import create_reflection
from agent.core.state import mark_idle_tick
from agent.scheduler.idle_policy import run_idle_policy


def run_tick() -> dict[str, object]:
    init_db()
    state = mark_idle_tick()
    result = run_idle_policy()
    event_id = log_event("scheduler", "idle_tick", "agentctl tick executed", result, importance=0.4)
    reflection_id = create_reflection("Recorded idle action and cooldown state after tick.", event_id, result.get("created_goal_id"), {"tick_result": result}, confidence=0.68)
    growth = result.get("cognitive_growth") if isinstance(result, dict) else None
    if isinstance(growth, dict):
        growth_task = growth.get("task") if isinstance(growth.get("task"), dict) else {}
        growth_summary = f"{growth.get('mode')}/{growth.get('top_curiosity')}/task={growth_task.get('task_id')}"
    else:
        growth_summary = "skipped"
    message = (
        f"tick complete: event=#{event_id}, reflection=#{reflection_id}, "
        f"created_goal={result.get('created_goal_id')}, workspace_artifact={result.get('workspace_artifact_id')}, "
        f"growth={growth_summary}, processed_events={result.get('processed_events')}"
    )
    return {"message": message, "state": state, "result": result, "reflection_id": reflection_id}
