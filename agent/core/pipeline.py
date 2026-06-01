from __future__ import annotations

import json
from typing import Any

from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.decisions import record_decision
from agent.core.events import log_event
from agent.core.learner import update_after_turn
from agent.core.state import mark_user_interaction
from agent.renderer.engine import render_response
from agent.renderer.fallback_renderer import render as fallback_render
from agent.renderer.validator import validate_output


def run_talk(message: str, source: str = "user", source_event_id: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    init_db()
    mark_user_interaction()
    user_event_id = source_event_id or log_event(source, "user_message", message, metadata or {}, importance=0.8)
    decision = build_talk_decision(message, source_event_id=user_event_id)
    decision_row_id = record_decision(decision)
    log_event("core", "decision_created", json.dumps(decision, ensure_ascii=False), {"goal_id": decision["selected_goal_id"], "decision_id": decision_row_id}, 0.7)
    output = render_response(decision)
    validation = validate_output(output, decision.get("must_include"), decision.get("must_not_include"))
    if not validation["ok"]:
        output = fallback_render({**decision, "renderer": "fallback"})
    assistant_event_id = log_event("core", "assistant_output", output, {"validation": validation}, 0.7)
    learner = update_after_turn(message, user_event_id, decision.get("selected_goal_id"), decision)
    return {"text": output, "decision": decision, "validation": validation, "user_event_id": user_event_id, "assistant_event_id": assistant_event_id, "decision_id": decision_row_id, "learner": learner}
