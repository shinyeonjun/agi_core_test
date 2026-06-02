from __future__ import annotations

from typing import Any

import os

from agent.core.capabilities import collect_capability_map
from agent.config.defaults import now_kst
from agent.core.drives import compute_drives
from agent.core.goals import create_goal
from agent.core.learner import retrieve_skills
from agent.core.metrics import collect_metrics
from agent.core.observability import decision_trace
from agent.core.policy import PolicyEngine
from agent.core.self_map import self_map_brief
from agent.core.style import apply_style_feedback, get_active_style_profile, style_directives
from agent.core.user_goals import maybe_create_user_goal
from agent.language.engine import interpret_user_message
from agent.memory.store import search_memories


def build_talk_decision(user_message: str, source_event_id: int | None = None) -> dict[str, Any]:
    memories = search_memories(user_message, limit=5) if user_message.strip() else []
    language_interpretation = interpret_user_message(user_message, {"surface": "talk"}, source_event_id=source_event_id)
    style_feedback = apply_style_feedback(user_message, interpretation=language_interpretation)
    style_profile = get_active_style_profile()
    user_goal = maybe_create_user_goal(user_message, source_event_id=source_event_id, interpretation=language_interpretation)
    renderer_name = os.getenv("AGENT_CHAT_RENDERER", "codex").strip().lower() or "codex"
    answer_goal_id = create_goal("Answer user input", user_message, goal_type="answer_user", status="done", priority=0.95, risk_level="low", metadata={"renderer": renderer_name, "source_event_id": source_event_id}, dedupe=False)
    drives = compute_drives()
    metrics = collect_metrics()
    runtime_self_map = self_map_brief()
    capability_map = collect_capability_map()
    policy = PolicyEngine().classify_decision(user_message, action_type="user_message")
    skills = retrieve_skills(user_message, tags=["talk", "core"], limit=3)
    selected_goal = user_goal or {"id": answer_goal_id, "title": "Answer user input", "goal_type": "answer_user"}
    selected_goal_id = int(selected_goal["id"])
    decision = {
        "version": "0.16",
        "kind": "talk_response",
        "created_at": now_kst(),
        "user_input": user_message,
        "user_message": user_message,
        "source_event_id": source_event_id,
        "selected_goal": selected_goal,
        "selected_goal_id": selected_goal_id,
        "answer_goal_id": answer_goal_id,
        "user_directed_goal": user_goal,
        "user_goal_created": user_goal is not None and user_goal.get("status") in {"active", "waiting_approval"},
        "language_interpretation": language_interpretation,
        "style_profile": style_profile,
        "style_directives": style_directives(style_profile),
        "style_feedback": style_feedback,
        "relevant_memories": memories,
        "selected_memories": memories,
        "relevant_skills": skills,
        "drive_scores": drives,
        "metrics": metrics,
        "runtime_self_map": runtime_self_map,
        "capability_map": capability_map,
        "policy_summary": {"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "reason": policy.reason, "matched_rules": policy.matched_rules},
        "core_judgment": "Core stored the input as an event and used memory, skill, goal, and state to build a verifiable response.",
        "confidence": 0.82,
        "decision_confidence": 0.82,
        "risk_level": policy.risk_level,
        "renderer": renderer_name,
        "must_include": [],
        "must_not_include": ["auto sudo execution", "consciousness emerged", "OS change without approval", "AGI achieved"],
        "renderer_hint": {"language": "ko", "style": "calm, precise"},
    }
    decision["decision_trace"] = decision_trace(decision)
    return decision
