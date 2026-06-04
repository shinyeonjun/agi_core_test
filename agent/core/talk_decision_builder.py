from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agent.config.defaults import now_kst
from agent.core.capabilities import collect_capability_map
from agent.core.decision_schema import build_decision_schema
from agent.core.drives import compute_drives
from agent.core.goals import create_goal
from agent.core.learner import retrieve_skills
from agent.core.metrics import collect_metrics
from agent.core.observability import decision_trace
from agent.core.policy import PolicyEngine
from agent.core.routing import memory_route, skill_route, tool_routes
from agent.core.self_map import self_map_brief
from agent.core.self_report import build_self_report_context
from agent.core.style import apply_style_feedback, get_active_style_profile, style_directives
from agent.core.user_goals import maybe_create_user_goal
from agent.language.engine import interpret_user_message
from agent.memory.store import search_memories
from agent.renderer.answer_contract import build_answer_contract


@dataclass(frozen=True)
class TalkDecisionContext:
    user_message: str
    source_event_id: int | None
    renderer_name: str
    memories: list[dict[str, Any]]
    language_interpretation: dict[str, Any]
    style_feedback: Any
    style_profile: dict[str, Any]
    user_goal: dict[str, Any] | None
    answer_goal_id: int
    drives: dict[str, Any]
    metrics: dict[str, Any]
    runtime_self_map: dict[str, Any]
    capability_map: dict[str, Any]
    self_report_context: dict[str, Any]
    answer_contract: dict[str, Any]
    policy: Any
    skills: list[dict[str, Any]]
    routing: dict[str, Any]


def selected_chat_renderer_name() -> str:
    return os.getenv("AGENT_CHAT_RENDERER", "codex").strip().lower() or "codex"


def collect_talk_decision_context(user_message: str, source_event_id: int | None = None) -> TalkDecisionContext:
    memories = search_memories(user_message, limit=5) if user_message.strip() else []
    language_interpretation = interpret_user_message(user_message, {"surface": "talk"}, source_event_id=source_event_id)
    style_feedback = apply_style_feedback(user_message, interpretation=language_interpretation)
    style_profile = get_active_style_profile()
    user_goal = maybe_create_user_goal(user_message, source_event_id=source_event_id, interpretation=language_interpretation)
    renderer_name = selected_chat_renderer_name()
    answer_goal_id = create_goal(
        "Answer user input",
        user_message,
        goal_type="answer_user",
        status="done",
        priority=0.95,
        risk_level="low",
        metadata={"renderer": renderer_name, "source_event_id": source_event_id},
        dedupe=False,
    )
    drives = compute_drives()
    metrics = collect_metrics()
    runtime_self_map = self_map_brief()
    capability_map = collect_capability_map()
    self_report_context = build_self_report_context(user_message, capability_map=capability_map, metrics=metrics)
    answer_contract = build_answer_contract(user_message, language_interpretation, self_report_context, user_goal)
    policy = PolicyEngine().classify_decision(user_message, action_type="user_message")
    skills = retrieve_skills(user_message, tags=["talk", "core"], limit=3)
    routing = _build_talk_routing(user_message, memories, skills, language_interpretation, policy, capability_map)
    return TalkDecisionContext(
        user_message=user_message,
        source_event_id=source_event_id,
        renderer_name=renderer_name,
        memories=memories,
        language_interpretation=language_interpretation,
        style_feedback=style_feedback,
        style_profile=style_profile,
        user_goal=user_goal,
        answer_goal_id=answer_goal_id,
        drives=drives,
        metrics=metrics,
        runtime_self_map=runtime_self_map,
        capability_map=capability_map,
        self_report_context=self_report_context,
        answer_contract=answer_contract,
        policy=policy,
        skills=skills,
        routing=routing,
    )


def _build_talk_routing(
    user_message: str,
    memories: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    language_interpretation: dict[str, Any],
    policy: Any,
    capability_map: dict[str, Any],
) -> dict[str, Any]:
    return {
        "memory": memory_route(user_message, memories),
        "skills": skill_route(user_message, skills),
        "tools": tool_routes(language_interpretation, policy, capability_map),
    }


def assemble_talk_decision(context: TalkDecisionContext) -> dict[str, Any]:
    selected_goal = context.user_goal or {"id": context.answer_goal_id, "title": "Answer user input", "goal_type": "answer_user"}
    selected_goal_id = int(selected_goal["id"])
    policy = context.policy
    decision = {
        "version": "0.17",
        "kind": "talk_response",
        "created_at": now_kst(),
        "user_input": context.user_message,
        "user_message": context.user_message,
        "source_event_id": context.source_event_id,
        "selected_goal": selected_goal,
        "selected_goal_id": selected_goal_id,
        "answer_goal_id": context.answer_goal_id,
        "user_directed_goal": context.user_goal,
        "user_goal_created": context.user_goal is not None and context.user_goal.get("status") in {"active", "waiting_approval"},
        "language_interpretation": context.language_interpretation,
        "style_profile": context.style_profile,
        "style_directives": style_directives(context.style_profile),
        "style_feedback": context.style_feedback,
        "relevant_memories": context.memories,
        "selected_memories": context.memories,
        "relevant_skills": context.skills,
        "drive_scores": context.drives,
        "metrics": context.metrics,
        "runtime_self_map": context.runtime_self_map,
        "capability_map": context.capability_map,
        "self_report_context": context.self_report_context,
        "answer_contract": context.answer_contract,
        "routing": context.routing,
        "policy_summary": {
            "risk_level": policy.risk_level,
            "requires_approval": policy.requires_approval,
            "denied": policy.denied,
            "reason": policy.reason,
            "matched_rules": policy.matched_rules,
        },
        "core_judgment": "Core stored the input as an event and used memory, skill, goal, and state to build a verifiable response.",
        "confidence": 0.82,
        "decision_confidence": 0.82,
        "risk_level": policy.risk_level,
        "renderer": context.renderer_name,
        "must_include": [],
        "must_not_include": ["auto sudo execution", "consciousness emerged", "OS change without approval", "AGI achieved"],
        "renderer_hint": {"language": "ko", "style": "calm, precise"},
    }
    decision["decision_schema"] = build_decision_schema(decision)
    decision["decision_trace"] = decision_trace(decision)
    return decision


def build_talk_decision(user_message: str, source_event_id: int | None = None) -> dict[str, Any]:
    context = collect_talk_decision_context(user_message, source_event_id=source_event_id)
    return assemble_talk_decision(context)
