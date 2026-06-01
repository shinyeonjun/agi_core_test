from __future__ import annotations

from typing import Any


def render(decision: dict[str, Any]) -> str:
    memories = decision.get("selected_memories", [])
    drives = decision.get("drive_scores", {})
    top_drive = max(drives.items(), key=lambda item: item[1])[0] if drives else "unknown"

    lines = [
        "v0.1 fallback renderer \uc751\ub2f5\uc774\uc57c.",
        "\uc785\ub825\uc740 event\ub85c \uc800\uc7a5\ud588\uace0, answer_user goal\uc744 \uc0dd\uc131\ud574\uc11c \ucc98\ub9ac\ud588\uc5b4.",
        "",
        f"- goal: #{decision.get('selected_goal_id')}",
        f"- renderer: {decision.get('renderer')}",
        f"- top_drive: {top_drive}",
        f"- related_memories: {len(memories)}\uac1c",
        "",
    ]
    if memories:
        lines.append("\uad00\ub828 \uae30\uc5b5:")
        for memory in memories[:3]:
            lines.append(f"- #{memory['id']} {memory['title']}")
        lines.append("")
    lines.append("\ud604\uc7ac Core\ub294 Codex \uc5c6\uc774\ub3c4 state, event, memory, goal \ud750\ub984\uc744 \uac80\uc99d\ud558\ub294 \ub2e8\uacc4\uc57c.")
    return "\n".join(lines)
