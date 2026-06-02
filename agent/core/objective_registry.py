from __future__ import annotations

from typing import Any

ALLOWED_GENERATED_GOAL_TYPES = {
    "reporting",
    "system_observation",
    "workspace_experiment",
    "research_note",
    "self_improvement_proposal",
    "skill_review",
    "memory_cleanup",
    "project_incubation",
}

DEFAULT_OBJECTIVES: tuple[dict[str, Any], ...] = (
    {"objective_type": "self_maintenance", "title": "Self maintenance", "description": "Keep Core state, DB records, events, memories, goals, and reflections understandable and tidy.", "priority": 0.82, "cooldown_seconds": 21600},
    {"objective_type": "system_observation", "title": "System observation", "description": "Observe Orange Pi system health with read-only checks and summarize changes.", "priority": 0.86, "cooldown_seconds": 21600},
    {"objective_type": "workspace_experiment", "title": "Workspace experiment", "description": "Create small safe workspace experiments and reports without mutating the OS.", "priority": 0.76, "cooldown_seconds": 28800},
    {"objective_type": "project_incubation", "title": "Project incubation", "description": "Develop small project candidates that can become useful tools or reports.", "priority": 0.70, "cooldown_seconds": 43200},
    {"objective_type": "self_improvement_proposal", "title": "Self-improvement proposal", "description": "Write proposal-only improvement plans for Core behavior, policy, tests, or UX.", "priority": 0.78, "cooldown_seconds": 43200},
    {"objective_type": "skill_growth", "title": "Skill growth", "description": "Review repeated success/failure patterns and suggest skill improvements.", "priority": 0.72, "cooldown_seconds": 43200},
    {"objective_type": "research_loop", "title": "Research loop", "description": "Create safe research notes about Core goals, policies, digital-world changes, and evaluation criteria.", "priority": 0.68, "cooldown_seconds": 43200},
)

TEMPLATES: dict[str, dict[str, Any]] = {
    "self_maintenance": {"title": "Review recent Core state and cleanup opportunities", "description": "Summarize recent events, memories, goals, and reflections, then identify safe cleanup opportunities.", "goal_type": "memory_cleanup", "novelty": 0.45, "utility": 0.82, "drive": "memory_hygiene"},
    "system_observation": {"title": "Generate a read-only Orange Pi health observation report", "description": "Use read-only system context to report disk, memory, uptime, and failed-service status.", "goal_type": "system_observation", "novelty": 0.42, "utility": 0.90, "drive": "system_maintenance"},
    "workspace_experiment": {"title": "Create a workspace experiment report from recent Core activity", "description": "Create a small workspace report that summarizes recent actions, proposals, and observations.", "goal_type": "workspace_experiment", "novelty": 0.76, "utility": 0.70, "drive": "curiosity"},
    "project_incubation": {"title": "Draft a small project candidate from recent Core observations", "description": "Write a project candidate that could become a useful Core tool or report without executing changes.", "goal_type": "project_incubation", "novelty": 0.74, "utility": 0.64, "drive": "curiosity"},
    "self_improvement_proposal": {"title": "Propose a safe Core self-improvement plan", "description": "Write a proposal-only plan for improving Core behavior, policy coverage, tests, or Discord UX.", "goal_type": "self_improvement_proposal", "novelty": 0.70, "utility": 0.78, "drive": "skill_improvement", "risk": "medium"},
    "skill_growth": {"title": "Review recent action patterns for skill growth", "description": "Find repeated action or reflection patterns and propose a skill improvement note.", "goal_type": "skill_review", "novelty": 0.58, "utility": 0.72, "drive": "skill_improvement"},
    "research_loop": {"title": "Write a research note on autonomous goal quality", "description": "Create a safe research note about how Core should judge generated goals and observations.", "goal_type": "research_note", "novelty": 0.78, "utility": 0.62, "drive": "curiosity"},
}


def objective_types() -> set[str]:
    return {str(item["objective_type"]) for item in DEFAULT_OBJECTIVES}
