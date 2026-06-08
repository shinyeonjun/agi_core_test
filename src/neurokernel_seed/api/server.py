from __future__ import annotations

from pathlib import Path
from typing import Any

from neurokernel_seed.harness.activation import ActivationError
from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.language.codex_harness import CodexLanguageHarness


def create_app(*, db_path: str | Path = "data/harness.db", project_root: str | Path = "."):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("fastapi is required to serve the Core API. Install with: pip install fastapi uvicorn") from exc

    service = HarnessService(db_path=db_path, project_root=project_root)
    language = CodexLanguageHarness()
    app = FastAPI(title="NeuroKernel AGI Seed Harness", version="0.1.0")

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/status")
    def status():
        return service.status()

    @app.get("/actions")
    def actions():
        return service.actions()

    @app.get("/capability-gaps")
    def capability_gaps(limit: int = 20, status: str | None = None):
        return service.capability_gaps(limit=limit, status=status)

    @app.get("/capability-proposals")
    def capability_proposals(limit: int = 20, status: str | None = None):
        return service.capability_proposals(limit=limit, status=status)

    @app.get("/capability-proposals/{proposal_id}")
    def capability_proposal(proposal_id: str):
        return service.capability_proposal(proposal_id)

    @app.get("/work-items")
    def work_items(limit: int = 20, status: str | None = None, work_type: str | None = None):
        return service.work_items(limit=limit, status=status, work_type=work_type)

    @app.get("/work-items/{work_id}")
    def work_item(work_id: str):
        return service.work_item(work_id)

    @app.get("/work-jobs")
    def work_jobs(limit: int = 20, work_id: str | None = None, status: str | None = None, queue_name: str | None = None):
        return service.work_jobs(limit=limit, work_id=work_id, status=status, queue_name=queue_name)

    @app.get("/work-pipeline/status")
    def work_pipeline_status(limit: int = 20, stale_after_seconds: int = 300):
        return service.work_pipeline_status(limit=limit, stale_after_seconds=stale_after_seconds)

    @app.get("/improvements/analyze")
    def improvements_analyze(min_gap_count: int = 2, lookback: int = 200):
        return service.analyze_improvements(min_gap_count=min_gap_count, lookback=lookback)

    @app.post("/improvements/propose")
    def improvements_propose(payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.propose_improvements(
            min_gap_count=int(payload.get("min_gap_count") or 2),
            lookback=int(payload.get("lookback") or 200),
            actor=str(payload.get("actor") or "api"),
        )

    @app.get("/model-improvements/analyze")
    def model_improvements_analyze(
        min_known_runtime_candidates: int = 100,
        min_new_known_runtime_candidates: int = 50,
        min_runtime_ranking_groups: int = 10,
        target_runtime_top1: float = 0.65,
    ):
        return service.analyze_model_improvements(
            min_known_runtime_candidates=min_known_runtime_candidates,
            min_new_known_runtime_candidates=min_new_known_runtime_candidates,
            min_runtime_ranking_groups=min_runtime_ranking_groups,
            target_runtime_top1=target_runtime_top1,
        )

    @app.post("/model-improvements/propose")
    def model_improvements_propose(payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.propose_model_improvements(
            min_known_runtime_candidates=int(payload.get("min_known_runtime_candidates") or 100),
            min_new_known_runtime_candidates=int(payload.get("min_new_known_runtime_candidates") or 50),
            min_runtime_ranking_groups=int(payload.get("min_runtime_ranking_groups") or 10),
            target_runtime_top1=float(payload.get("target_runtime_top1") or 0.65),
            actor=str(payload.get("actor") or "api"),
        )

    @app.get("/queue/health")
    def queue_health():
        return service.queue_health()

    @app.post("/work-items/{work_id}/enqueue")
    def work_item_enqueue(work_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.enqueue_work_item(work_id, actor=str(payload.get("actor") or "api"), max_attempts=int(payload.get("max_attempts") or 3))

    @app.post("/work-items/{work_id}/retry")
    def work_item_retry(work_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.retry_work_item(work_id, actor=str(payload.get("actor") or "api"), max_attempts=int(payload.get("max_attempts") or 3))

    @app.post("/work-items/{work_id}/promote-self-patch")
    def work_item_promote_self_patch(work_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.promote_work_item_to_self_patch(work_id, actor=str(payload.get("actor") or "api"), max_attempts=int(payload.get("max_attempts") or 3))

    @app.post("/work-items/{work_id}/note")
    def work_item_note(work_id: str, payload: dict[str, Any]):
        return service.add_work_note(work_id, actor=str(payload.get("actor") or "api"), note=str(payload.get("note") or ""))

    @app.post("/work-items/{work_id}/discord-notified")
    def work_item_discord_notified(work_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        actor = str(payload.get("actor") or "discord-work-notifier")
        event_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
        return service.mark_work_discord_notified(work_id, actor=actor, payload=event_payload)

    @app.post("/work-items/{work_id}/status")
    def work_item_status(work_id: str, payload: dict[str, Any]):
        return service.transition_work_item(
            work_id,
            str(payload.get("status") or payload.get("next_status") or ""),
            actor=str(payload.get("actor") or "api"),
            reason=payload.get("reason"),
        )

    @app.post("/work-items/{work_id}/activate")
    def work_item_activate(work_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        try:
            return service.activate_work_item(work_id, actor=str(payload.get("actor") or "api"))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail={"error_type": "not_found", "message": str(exc), "work_id": work_id}) from exc
        except ActivationError as exc:
            raise HTTPException(status_code=409, detail={"error_type": "activation_failed", "message": str(exc), "work_id": work_id}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error_type": "invalid_request", "message": str(exc), "work_id": work_id}) from exc

    @app.post("/work/route")
    def work_route(payload: dict[str, Any]):
        user_text = str(payload.get("user_text") or payload.get("text") or "").strip()
        if not user_text:
            return {"created": False, "route": "clarify", "reply": "조금 더 구체적으로 말해줘."}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_id = str(context.get("user_id") or payload.get("user_id") or "").strip()
        channel_id = str(context.get("channel_id") or payload.get("channel_id") or "").strip()
        if user_id:
            context = {**context, "memory": service.memory_context(user_id, channel_id=channel_id or None)}
        language_intent = payload.get("language_intent") if isinstance(payload.get("language_intent"), dict) else {}
        route_decision = language.route_work(
            user_text,
            context={**context, "language_intent": language_intent, "action_catalog": service.actions()},
        )
        route = str(route_decision.get("route") or "clarify")
        if route == "self_patch":
            capability_intent = language.propose_capability(
                user_text,
                context={**context, "language_intent": language_intent, "route_decision": route_decision, "action_catalog": service.actions()},
            )
            result = service.create_capability_proposal_from_intent(
                user_text=user_text,
                capability_intent=capability_intent,
                user_id=user_id or None,
                channel_id=channel_id or None,
                language_intent=language_intent,
            )
            return {**result, "route": route, "route_decision": route_decision}
        if route == "external_work":
            result = service.create_work_item_from_route(
                user_text=user_text,
                route_decision=route_decision,
                user_id=user_id or None,
                channel_id=channel_id or None,
            )
            return result
        return {"created": False, "route": route, "route_decision": route_decision}

    @app.post("/capability-proposals/from-request")
    def capability_proposal_from_request(payload: dict[str, Any]):
        user_text = str(payload.get("user_text") or payload.get("text") or "").strip()
        if not user_text:
            return {"created": False, "kind": "none", "reply": ""}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_id = str(context.get("user_id") or payload.get("user_id") or "").strip()
        channel_id = str(context.get("channel_id") or payload.get("channel_id") or "").strip()
        if user_id:
            context = {**context, "memory": service.memory_context(user_id, channel_id=channel_id or None)}
        language_intent = payload.get("language_intent") if isinstance(payload.get("language_intent"), dict) else {}
        capability_intent = language.propose_capability(
            user_text,
            context={**context, "language_intent": language_intent, "action_catalog": service.actions()},
        )
        return service.create_capability_proposal_from_intent(
            user_text=user_text,
            capability_intent=capability_intent,
            user_id=user_id or None,
            channel_id=channel_id or None,
            language_intent=language_intent,
        )

    @app.post("/capability-proposals/{proposal_id}/approve-dev")
    def capability_proposal_approve_dev(proposal_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.transition_capability_proposal(
            proposal_id,
            "approved_for_dev",
            actor=str(payload.get("actor") or "api"),
            reason=payload.get("reason"),
        )

    @app.post("/capability-proposals/{proposal_id}/reject")
    def capability_proposal_reject(proposal_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.transition_capability_proposal(
            proposal_id,
            "rejected",
            actor=str(payload.get("actor") or "api"),
            reason=payload.get("reason"),
        )

    @app.post("/capability-proposals/{proposal_id}/defer")
    def capability_proposal_defer(proposal_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.transition_capability_proposal(
            proposal_id,
            "deferred",
            actor=str(payload.get("actor") or "api"),
            reason=payload.get("reason"),
        )

    @app.post("/tasks")
    def create_task(payload: dict[str, Any]):
        return service.create_task(payload, source="api")

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        return service.get_task(task_id)

    @app.post("/tasks/{task_id}/dry-run")
    def dry_run(task_id: str):
        return service.dry_run(task_id)

    @app.post("/tasks/{task_id}/run")
    def run(task_id: str):
        return service.run(task_id)

    @app.post("/tasks/{task_id}/approve")
    def approve(task_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.approve(task_id, approved_by=str(payload.get("approved_by") or "api"), reason=payload.get("reason"))

    @app.post("/tasks/{task_id}/reject")
    def reject(task_id: str, payload: dict[str, Any] | None = None):
        payload = payload or {}
        return service.reject(task_id, rejected_by=str(payload.get("rejected_by") or "api"), reason=payload.get("reason"))

    @app.get("/tasks/{task_id}/trace")
    def task_trace(task_id: str):
        return {"task": service.get_task(task_id), "recent": service.recent_trace(20)}

    @app.get("/trace/recent")
    def recent_trace(limit: int = 5):
        return service.recent_trace(limit)

    @app.get("/memory/preferences")
    def memory_preferences(user_id: str, scope: str | None = None):
        return service.list_preferences(user_id, scope=scope)

    @app.post("/memory/preferences")
    def memory_set_preference(payload: dict[str, Any]):
        return service.set_preference(payload)

    @app.post("/memory/preferences/delete")
    def memory_delete_preference(payload: dict[str, Any]):
        return service.delete_preference(payload)

    @app.post("/memory/messages")
    def memory_add_message(payload: dict[str, Any]):
        return service.add_conversation_message(payload)

    @app.post("/memory/messages/link-task")
    def memory_link_message_to_task(payload: dict[str, Any]):
        return service.link_message_to_task(payload)

    @app.get("/memory/recent")
    def memory_recent(user_id: str, channel_id: str | None = None, limit: int = 20):
        return service.recent_conversation(user_id, channel_id=channel_id, limit=limit)

    @app.post("/memory/task-references")
    def memory_add_task_reference(payload: dict[str, Any]):
        return service.add_task_reference(payload)

    @app.post("/memory/interaction-outcomes")
    def memory_add_interaction_outcome(payload: dict[str, Any]):
        return service.add_interaction_outcome(payload)

    @app.get("/memory/context")
    def memory_context(user_id: str, channel_id: str | None = None, message_limit: int = 12):
        return service.memory_context(user_id, channel_id=channel_id, message_limit=message_limit)

    @app.post("/benchmark")
    def benchmark(payload: dict[str, Any] | None = None):
        payload = payload or {}
        task = {
            "goal": "run safe benchmark",
            "target": "orangepi5",
            "context": {"params": payload},
            "allowed_actions": ["run_safe_benchmark"],
            "blocked_actions": [],
            "success_criteria": ["benchmark completes"],
            "risk_level": "low",
            "requires_approval": False,
            "timeout_seconds": 300,
            "mode": "readonly",
        }
        created = service.create_task(task, source="api")
        return service.run(created["task"]["task_id"])

    @app.post("/language/to-core")
    def language_to_core(payload: dict[str, Any]):
        user_text = str(payload.get("user_text") or payload.get("text") or "").strip()
        if not user_text:
            return {
                "intent": "unknown",
                "reply": "무슨 작업인지 조금 더 말해줘.",
                "task_spec": None,
                "dev_task": None,
                "approval": None,
                "confidence": 0.0,
                "requires_confirmation": True,
                "clarifying_question": "무엇을 확인하거나 실행하면 될까?",
                "safety_notes": ["empty_input"],
            }
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_id = str(context.get("user_id") or payload.get("user_id") or "").strip()
        channel_id = str(context.get("channel_id") or payload.get("channel_id") or "").strip()
        if user_id:
            context = {**context, "memory": service.memory_context(user_id, channel_id=channel_id or None)}
        return language.to_core(user_text, context=context)

    @app.post("/language/preferences")
    def language_preferences(payload: dict[str, Any]):
        user_text = str(payload.get("user_text") or payload.get("text") or "").strip()
        if not user_text:
            return {"kind": "none", "reply": "", "candidates": [], "confidence": 0.0, "requires_confirmation": False, "clarifying_question": None, "safety_notes": ["empty_input"]}
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_id = str(context.get("user_id") or payload.get("user_id") or "").strip()
        channel_id = str(context.get("channel_id") or payload.get("channel_id") or "").strip()
        if user_id:
            context = {**context, "memory": service.memory_context(user_id, channel_id=channel_id or None)}
        return language.extract_preferences(user_text, context=context)

    @app.post("/language/to-human")
    def language_to_human(payload: dict[str, Any]):
        core_result = payload.get("core_result") if isinstance(payload.get("core_result"), dict) else payload
        style = str(payload.get("style") or "ko_short")
        context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        user_id = str(context.get("user_id") or payload.get("user_id") or "").strip()
        channel_id = str(context.get("channel_id") or payload.get("channel_id") or "").strip()
        if user_id:
            context = {**context, "memory": service.memory_context(user_id, channel_id=channel_id or None)}
        return language.to_human(core_result, style=style, context=context)

    return app


def serve(*, host: str = "127.0.0.1", port: int = 8765, db_path: str | Path = "data/harness.db", project_root: str | Path = ".") -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to serve the Core API. Install with: pip install fastapi uvicorn") from exc
    uvicorn.run(create_app(db_path=db_path, project_root=project_root), host=host, port=port)
