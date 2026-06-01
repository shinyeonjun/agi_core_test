from __future__ import annotations

from agent.core.database import init_db
from agent.core.events import log_event
from agent.core.state import mark_idle_tick
from agent.scheduler.idle_policy import run_idle_policy


def run_tick() -> dict[str, object]:
    init_db()
    state = mark_idle_tick()
    result = run_idle_policy()
    event_id = log_event("scheduler", "idle_tick", "agentctl tick executed", result, importance=0.4)
    message = (
        f"tick \uc644\ub8cc: event=#{event_id}, "
        f"created_goal={result.get('created_goal_id')}, "
        f"processed_events={result.get('processed_events')}"
    )
    return {"message": message, "state": state, "result": result}
