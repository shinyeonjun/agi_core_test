from __future__ import annotations

from typing import Any

from agent.config.defaults import now_kst
from agent.core.drives import compute_drives
from agent.core.goals import create_goal
from agent.core.learner import retrieve_skills
from agent.core.policy import PolicyEngine
from agent.memory.store import search_memories


def build_talk_decision(user_message: str, source_event_id: int | None = None) -> dict[str, Any]:
    memories = search_memories(user_message, limit=5) if user_message.strip() else []
    goal_id = create_goal("Answer user input", user_message, goal_type="answer_user", status="done", priority=0.95, risk_level="low", metadata={"renderer": "fallback", "source_event_id": source_event_id}, dedupe=False)
    drives = compute_drives()
    policy = PolicyEngine().classify_decision(user_message, action_type="user_message")
    skills = retrieve_skills(user_message, tags=["talk", "core"], limit=3)
    selected_goal = {"id": goal_id, "title": "Answer user input", "goal_type": "answer_user"}
    return {
        "version": "0.8",
        "kind": "talk_response",
        "created_at": now_kst(),
        "user_input": user_message,
        "user_message": user_message,
        "source_event_id": source_event_id,
        "selected_goal": selected_goal,
        "selected_goal_id": goal_id,
        "relevant_memories": memories,
        "selected_memories": memories,
        "relevant_skills": skills,
        "drive_scores": drives,
        "policy_summary": {"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "reason": policy.reason, "matched_rules": policy.matched_rules},
        "core_judgment": "Core stored the input as an event and used memory, skill, goal, and state to build a verifiable response.",
        "confidence": 0.82,
        "decision_confidence": 0.82,
        "risk_level": policy.risk_level,
        "renderer": "fallback",
        "must_include": ["v0.8", "Core", "event", "goal"],
        "must_not_include": ["auto sudo execution", "consciousness emerged", "OS change without approval", "AGI achieved"],
        "renderer_hint": {"language": "ko", "style": "calm, precise"},
    }
