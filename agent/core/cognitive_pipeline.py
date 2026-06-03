from __future__ import annotations

from typing import Any

from agent.core.cooldown import is_ready, mark
from agent.core.events import log_event
from agent.core.goals import create_goal
from agent.core.task_queue import enqueue_task, task_status_counts


TOPIC_TASKS: dict[str, dict[str, str]] = {
    "memory_hygiene": {
        "goal_type": "memory_cleanup",
        "task_kind": "memory_cleanup",
        "title": "Consolidate memory and reflection pressure",
        "description": "Review accumulated memories and reflections, then produce a safe cleanup note.",
    },
    "memory_retrieval": {
        "goal_type": "memory_cleanup",
        "task_kind": "memory_cleanup",
        "title": "Review memory retrieval quality",
        "description": "Inspect memory retrieval coverage and write a safe improvement note.",
    },
    "action_reliability": {
        "goal_type": "self_improvement_proposal",
        "task_kind": "improvement_plan",
        "title": "Improve action reliability feedback",
        "description": "Review recent action outcomes and propose safer execution feedback improvements.",
    },
    "renderer_quality": {
        "goal_type": "self_improvement_proposal",
        "task_kind": "improvement_plan",
        "title": "Improve conversational rendering quality",
        "description": "Review renderer quality signals and propose clearer Discord response behavior.",
    },
    "goal_queue": {
        "goal_type": "skill_review",
        "task_kind": "skill_review",
        "title": "Review goal queue separation",
        "description": "Review user and autonomous queue separation and produce a skill/process note.",
    },
    "runtime_body": {
        "goal_type": "system_observation",
        "task_kind": "system_observation",
        "title": "Refresh runtime body understanding",
        "description": "Use safe read-only system observation steps to keep Core body state understandable.",
    },
}

MODE_FALLBACKS: dict[str, dict[str, str]] = {
    "serve_user": {
        "goal_type": "skill_review",
        "task_kind": "skill_review",
        "title": "Keep autonomous loop behind user work",
        "description": "Record why autonomous work should wait while user-directed tasks are pending.",
    },
    "stabilize": TOPIC_TASKS["action_reliability"],
    "explore": {
        "goal_type": "research_note",
        "task_kind": "research_note",
        "title": "Explore the strongest growth signal",
        "description": "Write a safe research note about the strongest current curiosity signal.",
    },
    "consolidate": TOPIC_TASKS["memory_hygiene"],
    "repair_environment": {
        "goal_type": "self_improvement_proposal",
        "task_kind": "improvement_plan",
        "title": "Repair Python dependency readiness",
        "description": "Check venv Python dependencies and propose safe repair steps before rerunning work.",
    },
}


def _top_signal(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    signals = snapshot.get("curiosity") or []
    if not signals:
        return None
    return max(signals, key=lambda row: float(row.get("pressure") or 0.0))


def growth_task_decision(snapshot: dict[str, Any]) -> dict[str, Any]:
    inference = snapshot.get("active_inference") or {}
    mode = str(inference.get("mode") or "consolidate")
    signal = _top_signal(snapshot)
    topic = str((signal or {}).get("topic") or "")
    pressure = float((signal or {}).get("pressure") or 0.0)
    counts = task_status_counts()
    user_waiting = int(counts.get("user:queued", 0) or 0) + int(counts.get("user:running", 0) or 0)
    if user_waiting > 0:
        return {
            "should_enqueue": False,
            "reason": "user_queue_has_priority",
            "mode": mode,
            "topic": topic,
            "user_waiting": user_waiting,
        }
    if pressure < 0.18 and float(inference.get("free_energy") or 0.0) < 0.18:
        return {
            "should_enqueue": False,
            "reason": "growth_pressure_low",
            "mode": mode,
            "topic": topic,
            "pressure": round(pressure, 4),
        }
    template = TOPIC_TASKS.get(topic) or MODE_FALLBACKS.get(mode) or MODE_FALLBACKS["consolidate"]
    priority = max(0.35, min(0.92, pressure * 0.72 + float(inference.get("free_energy") or 0.0) * 0.28))
    return {
        "should_enqueue": True,
        "reason": "growth_pressure_ready",
        "mode": mode,
        "topic": topic or "mode",
        "pressure": round(pressure, 4),
        "priority": round(priority, 4),
        **template,
        "signal": signal or {},
        "inference": inference,
    }


def maybe_enqueue_growth_task(snapshot: dict[str, Any]) -> dict[str, Any]:
    decision = growth_task_decision(snapshot)
    if not decision.get("should_enqueue"):
        log_event("cognition", "growth_task_skipped", str(decision.get("reason")), decision, 0.45)
        return {**decision, "created_goal_id": None, "task_id": None}
    cooldown_key = f"growth_task:{decision['mode']}:{decision['topic']}"
    ready, wait_seconds = is_ready(cooldown_key, 3600)
    if not ready:
        result = {**decision, "should_enqueue": False, "reason": "growth_task_cooldown", "wait_seconds": wait_seconds, "created_goal_id": None, "task_id": None}
        log_event("cognition", "growth_task_skipped", result["reason"], result, 0.45)
        return result
    metadata = {
        "source": "cognitive_pipeline",
        "growth_mode": decision.get("mode"),
        "growth_topic": decision.get("topic"),
        "growth_pressure": decision.get("pressure"),
        "active_inference": decision.get("inference"),
        "curiosity_signal": decision.get("signal"),
    }
    goal_id = create_goal(
        str(decision["title"]),
        str(decision["description"]),
        goal_type=str(decision["goal_type"]),
        status="active",
        priority=float(decision["priority"]),
        risk_level="low" if decision["goal_type"] != "system_observation" else "medium",
        metadata=metadata,
        dedupe=True,
    )
    task_id = enqueue_task(
        "autonomous",
        goal_id=goal_id,
        task_kind=str(decision["task_kind"]),
        title=str(decision["title"]),
        source="cognitive_pipeline",
        priority=float(decision["priority"]),
        payload=metadata,
    )
    mark(cooldown_key, 3600, {"goal_id": goal_id, "task_id": task_id, "topic": decision.get("topic")})
    result = {**decision, "created_goal_id": goal_id, "task_id": task_id}
    log_event("cognition", "growth_task_enqueued", str(decision.get("topic")), result, 0.72)
    return result
