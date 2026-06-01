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
    message = f"tick complete: event=#{event_id}, reflection=#{reflection_id}, created_goal={result.get('created_goal_id')}, workspace_artifact={result.get('workspace_artifact_id')}, processed_events={result.get('processed_events')}"
    return {"message": message, "state": state, "result": result, "reflection_id": reflection_id}
