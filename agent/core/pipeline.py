from __future__ import annotations

import json
from typing import Any

from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.decisions import record_decision
from agent.core.events import log_event
from agent.core.learner import update_after_turn
from agent.core.pipeline_kernel import CorePipelineTrace
from agent.core.state import mark_user_interaction
from agent.renderer.engine import render_response
from agent.renderer.fallback_renderer import render as fallback_render
from agent.renderer.validator import validate_output


def run_talk(message: str, source: str = "user", source_event_id: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    init_db()
    trace = CorePipelineTrace()
    mark_user_interaction()
    user_event_id = source_event_id or log_event(source, "user_message", message, metadata or {}, importance=0.8)
    trace.add("observe", "done", "사용자 입력을 이벤트로 저장하고 인터랙션 상태를 갱신했다.", {"event_id": user_event_id, "source": source})
    decision = build_talk_decision(message, source_event_id=user_event_id)
    trace.add("interpret", "done", "언어 해석과 정책 판단을 완료했다.", {"intent": (decision.get("language_interpretation") or {}).get("intent"), "risk": decision.get("risk_level")})
    trace.add("retrieve", "done", "관련 기억과 스킬 후보를 조회했다.", {"memories": len(decision.get("relevant_memories") or []), "skills": len(decision.get("relevant_skills") or [])})
    trace.add("plan", "done", "선택 목표, 라우팅, decision schema를 구성했다.", {"goal_id": decision.get("selected_goal_id"), "user_goal_created": decision.get("user_goal_created")})
    if decision.get("user_goal_created"):
        trace.add("act_or_defer", "deferred", "사용자 지시는 user queue 또는 프로젝트 계획으로 분리했다.", {"task_id": (decision.get("user_directed_goal") or {}).get("task_id")})
    else:
        trace.add("act_or_defer", "deferred", "대화 응답만 필요한 입력이라 실행 작업을 만들지 않았다.", {})
    decision["pipeline_trace"] = trace.to_dict()
    decision_row_id = record_decision(decision)
    log_event("core", "decision_created", json.dumps(decision, ensure_ascii=False), {"goal_id": decision["selected_goal_id"], "decision_id": decision_row_id}, 0.7)
    output = render_response(decision)
    validation = validate_output(output, decision.get("must_include"), decision.get("must_not_include"))
    trace.add("verify", "done" if validation["ok"] else "failed", "렌더러 출력 검증을 수행했다.", {"validation": validation})
    if not validation["ok"]:
        output = fallback_render({**decision, "renderer": "fallback"})
    assistant_event_id = log_event("core", "assistant_output", output, {"validation": validation}, 0.7)
    learner = update_after_turn(message, user_event_id, decision.get("selected_goal_id"), decision)
    trace.add("reflect", "done", "대화 결과를 학습/회고 루틴에 반영했다.", {"learner": learner})
    trace.add("report", "done", "최종 응답을 assistant_output 이벤트로 저장했다.", {"event_id": assistant_event_id})
    decision["pipeline_trace"] = trace.to_dict()
    log_event("core_pipeline", "pipeline_trace_completed", json.dumps(decision["pipeline_trace"], ensure_ascii=False), {"decision_id": decision_row_id, "goal_id": decision.get("selected_goal_id")}, 0.55)
    return {"text": output, "decision": decision, "validation": validation, "user_event_id": user_event_id, "assistant_event_id": assistant_event_id, "decision_id": decision_row_id, "learner": learner}
