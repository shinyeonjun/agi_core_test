from __future__ import annotations

from typing import Any


def render(decision: dict[str, Any]) -> str:
    memories = decision.get("relevant_memories") or decision.get("selected_memories", [])
    drives = decision.get("drive_scores", {})
    top_drive = max(drives.items(), key=lambda item: item[1])[0] if drives else "unknown"
    policy = decision.get("policy_summary", {})
    skills = decision.get("relevant_skills", [])
    version = str(decision.get("version", "v0.16"))
    if not version.startswith("v"):
        version = f"v{version}"
    goal = decision.get("selected_goal") or {"id": decision.get("selected_goal_id"), "title": "unknown"}
    lines = [
        f"{version} Core fallback renderer response.",
        "Core saved the input as an event and used memory/skill/goal/state for this answer.",
        "",
        f"- goal: #{goal.get('id')} {goal.get('title')}",
        f"- renderer: {decision.get('renderer', 'fallback')}",
        f"- top_drive: {top_drive}",
        f"- policy: {policy.get('risk_level', decision.get('risk_level'))}, approval={policy.get('requires_approval', False)}",
        f"- related_memories: {len(memories)}",
        f"- related_skills: {len(skills)}",
        "",
    ]
    if memories:
        lines.append("Relevant memories:")
        for memory in memories[:3]:
            score = memory.get("score")
            suffix = f" score={score}" if score is not None else ""
            lines.append(f"- #{memory['id']} {memory['title']}{suffix}")
        lines.append("")
    if skills:
        lines.append("Relevant skills:")
        for skill in skills[:3]:
            lines.append(f"- {skill['name']} confidence={skill.get('confidence')}")
        lines.append("")
    if policy.get("denied"):
        lines.append("The request matched a risky pattern, so Core separated it into policy/approval flow and did not execute it.")
    else:
        lines.append("This Core is a stateful digital-agent runtime. It does not claim AGI; it grows testable components for memory, goals, policy, reflection, skills, and evaluation.")
    return "\n".join(lines)
