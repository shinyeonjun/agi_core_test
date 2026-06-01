from __future__ import annotations

from agent.core.events import count_unprocessed_events
from agent.core.goals import count_open_goals
from agent.core.state import load_state


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_drives() -> dict[str, float]:
    state = load_state()
    unprocessed = count_unprocessed_events()
    open_goals = count_open_goals()
    memory_clutter = float(state.get("memory_clutter", 0.0))

    return {
        "accuracy": clamp(float(state.get("uncertainty_level", 0.0)) * 0.5),
        "completion": clamp(open_goals * 0.15),
        "curiosity": 0.10,
        "memory_hygiene": clamp(unprocessed * 0.01 + memory_clutter * 0.5),
        "skill_improvement": 0.05,
        "risk_avoidance": clamp(float(state.get("risk_level", 0.0))),
        "user_alignment": clamp(float(state.get("user_alignment_pressure", 0.0))),
        "system_maintenance": clamp(float(state.get("system_health_pressure", 0.0))),
    }
