from __future__ import annotations

from typing import Any

from agent.core.drives import compute_drives
from agent.core.goals import create_goal
from agent.memory.store import search_memories


def build_talk_decision(user_message: str) -> dict[str, Any]:
    memories = search_memories(user_message, limit=5) if user_message.strip() else []
    goal_id = create_goal(
        title="\uc0ac\uc6a9\uc790 \uc785\ub825\uc5d0 \ub2f5\ubcc0 \uc0dd\uc131",
        description=user_message,
        goal_type="answer_user",
        status="done",
        priority=0.95,
        risk_level="low",
        metadata={"renderer": "fallback"},
        dedupe=False,
    )
    drives = compute_drives()
    return {
        "version": "0.1",
        "kind": "talk_response",
        "user_message": user_message,
        "selected_goal_id": goal_id,
        "selected_memories": memories,
        "drive_scores": drives,
        "decision_confidence": 0.72,
        "risk_level": "low",
        "renderer": "fallback",
        "main_answer": "\uc785\ub825\uc744 event\ub85c \uc800\uc7a5\ud558\uace0, \uad00\ub828 memory\uc640 goal\uc744 \uae30\uc900\uc73c\ub85c fallback \ub2f5\ubcc0\uc744 \uc0dd\uc131\ud55c\ub2e4.",
        "must_include": ["v0.1", "fallback renderer", "event", "goal"],
        "must_not_include": ["\uc790\ub3d9 sudo \uc2e4\ud589", "\uc758\uc2dd\uc774 \uc0dd\uae40", "\uc2b9\uc778 \uc5c6\uc774 OS \ubcc0\uacbd"],
    }
