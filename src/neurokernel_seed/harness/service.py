from __future__ import annotations

from pathlib import Path
from typing import Any

from .action_catalog import ActionDefinition, build_action_catalog, public_catalog
from .activation import ActivationService, build_activation_config_from_env
from .capability_service import CapabilityProposalService
from .executors.benchmark import BenchmarkExecutor
from .executors.readonly_command import ReadOnlyCommandExecutor
from .executors.readonly_system import ReadOnlyExecutor
from .memory import HarnessMemory
from .preferences import preference_map
from .safety_gate import check_action_safety
from .task_spec import TaskSpec, task_spec_from_dict
from .trace import failure_trace
from .work_queue import WorkQueue, WorkQueueError, build_work_queue_from_env
from .work_service import WorkItemService


class HarnessService:
    """Core API facade.

    Runtime callers keep using this class, while domain-specific behavior lives
    in focused services. This keeps FastAPI/Discord stable and lets workers grow
    without turning the facade into a kitchen sink.
    """

    def __init__(
        self,
        *,
        db_path: str | Path = "data/harness.db",
        project_root: str | Path = ".",
        catalog: dict[str, ActionDefinition] | None = None,
        work_queue: WorkQueue | None = None,
    ):
        self.db_path = Path(db_path)
        self.project_root = Path(project_root)
        self.catalog = catalog or build_action_catalog()
        self.readonly_executor = ReadOnlyExecutor(project_root=self.project_root, memory_path=self.db_path)
        self.readonly_command_executor = ReadOnlyCommandExecutor(project_root=self.project_root)
        self.benchmark_executor = BenchmarkExecutor(project_root=self.project_root)
        self.work_queue, self.queue_error = _resolve_work_queue(work_queue)
        self.work_items_service = WorkItemService(db_path=self.db_path, work_queue=self.work_queue, queue_error=self.queue_error)
        self.capability_service = CapabilityProposalService(db_path=self.db_path, catalog=self.catalog, work_items=self.work_items_service)
        self.activation_service = ActivationService(build_activation_config_from_env(db_path=self.db_path, project_root=self.project_root))

    def health(self) -> dict[str, Any]:
        return {"ok": True, "service": "neurokernel-harness-v0"}

    def status(self) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            return {
                "health": self.health(),
                "queue": self.queue_health(),
                "recent_tasks": memory.list_tasks(10),
                "recent_traces": memory.recent_traces(5),
                "recent_work": memory.list_work_items(limit=5),
            }

    def queue_health(self) -> dict[str, Any]:
        return self.work_items_service.queue_health()

    def actions(self) -> list[dict[str, Any]]:
        return public_catalog(self.catalog)

    def capability_gaps(self, *, limit: int = 20, status: str | None = None) -> dict[str, Any]:
        return self.capability_service.list_gaps(limit=limit, status=status)

    def capability_proposals(self, *, limit: int = 20, status: str | None = None) -> dict[str, Any]:
        return self.capability_service.list_proposals(limit=limit, status=status)

    def capability_proposal(self, proposal_id: str) -> dict[str, Any]:
        return self.capability_service.get_proposal(proposal_id)

    def create_capability_proposal_from_intent(
        self,
        *,
        user_text: str,
        capability_intent: dict[str, Any],
        user_id: str | None = None,
        channel_id: str | None = None,
        language_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.capability_service.create_from_intent(
            user_text=user_text,
            capability_intent=capability_intent,
            user_id=user_id,
            channel_id=channel_id,
            language_intent=language_intent,
        )

    def transition_capability_proposal(self, proposal_id: str, next_status: str, *, actor: str = "api", reason: str | None = None) -> dict[str, Any]:
        return self.capability_service.transition(proposal_id, next_status, actor=actor, reason=reason)

    def work_items(self, *, limit: int = 20, status: str | None = None, work_type: str | None = None) -> dict[str, Any]:
        return self.work_items_service.list_items(limit=limit, status=status, work_type=work_type)

    def work_item(self, work_id: str) -> dict[str, Any]:
        return self.work_items_service.get_item(work_id)

    def work_jobs(self, *, limit: int = 20, work_id: str | None = None, status: str | None = None, queue_name: str | None = None) -> dict[str, Any]:
        return self.work_items_service.list_jobs(limit=limit, work_id=work_id, status=status, queue_name=queue_name)

    def add_work_note(self, work_id: str, *, actor: str = "api", note: str) -> dict[str, Any]:
        return self.work_items_service.add_note(work_id, actor=actor, note=note)

    def transition_work_item(self, work_id: str, next_status: str, *, actor: str = "api", reason: str | None = None) -> dict[str, Any]:
        return self.work_items_service.transition(work_id, next_status, actor=actor, reason=reason)

    def enqueue_work_item(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        return self.work_items_service.enqueue(work_id, actor=actor, max_attempts=max_attempts)

    def retry_work_item(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        return self.work_items_service.retry(work_id, actor=actor, max_attempts=max_attempts)

    def activate_work_item(self, work_id: str, *, actor: str = "api") -> dict[str, Any]:
        return self.activation_service.activate_work_item(work_id, actor=actor)

    def create_work_item_from_route(
        self,
        *,
        user_text: str,
        route_decision: dict[str, Any],
        user_id: str | None = None,
        channel_id: str | None = None,
        source_message_id: str | None = None,
    ) -> dict[str, Any]:
        return self.work_items_service.create_from_route(
            user_text=user_text,
            route_decision=route_decision,
            user_id=user_id,
            channel_id=channel_id,
            source_message_id=source_message_id,
        )

    def create_task(self, data: dict[str, Any], *, created_by: str = "system", source: str = "cli") -> dict[str, Any]:
        spec = task_spec_from_dict(data, catalog=self.catalog)
        with HarnessMemory(self.db_path) as memory:
            memory.create_task(spec, created_by=created_by, source=source)
            memory.transition_task(spec.task_id, "validated", {"validator": "accepted"})
            target_state = "waiting_approval" if spec.requires_approval else "ready"
            memory.transition_task(spec.task_id, target_state, {"requires_approval": spec.requires_approval})
            task = memory.get_task(spec.task_id)
        return {"accepted": True, "task": task}

    def get_task(self, task_id: str) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            return {"task": memory.get_task(task_id), "events": memory.events_for_task(task_id)}

    def dry_run(self, task_id: str) -> dict[str, Any]:
        task = self._load_task_spec(task_id)
        candidates = self._candidate_actions(task)
        decisions = []
        for action_id in candidates:
            decision = check_action_safety(task, action_id, catalog=self.catalog)
            decisions.append({"action_id": action_id, "safety": decision.as_dict()})
        chosen = next((item for item in decisions if item["safety"]["decision"] in {"allow", "dry_run_only"}), decisions[0] if decisions else None)
        with HarnessMemory(self.db_path) as memory:
            memory.add_action_decision(
                task_id,
                0,
                [{"action_id": item["action_id"]} for item in decisions],
                {"action_id": chosen["action_id"]} if chosen else None,
                chosen["safety"] if chosen else {"decision": "deny", "reason": "no candidate"},
            )
            memory.add_event(task_id, "dry_run", {"decisions": decisions, "chosen": chosen})
        return {"task_id": task_id, "mode": "dry_run", "candidate_decisions": decisions, "chosen": chosen}

    def run(self, task_id: str) -> dict[str, Any]:
        task = self._load_task_spec(task_id)
        candidates = self._candidate_actions(task)
        if not candidates:
            return self._fail_task(task_id, "no_candidate_actions", {"reason": "allowed_actions is empty"})

        with HarnessMemory(self.db_path) as memory:
            status = memory.get_task(task_id)["status"]
            if status == "waiting_approval":
                return {"task_id": task_id, "status": "waiting_approval", "requires_approval": True}
            if status not in {"ready", "validated"}:
                return {"task_id": task_id, "status": status, "error": "task is not runnable"}
            memory.transition_task(task_id, "running", {})
            memory.transition_task(task_id, "deciding", {})

        chosen_action, safety = self._choose_first_allowed(task, candidates)
        with HarnessMemory(self.db_path) as memory:
            memory.add_action_decision(
                task_id,
                0,
                [{"action_id": action_id} for action_id in candidates],
                {"action_id": chosen_action} if chosen_action else None,
                safety.as_dict() if safety else {"decision": "deny"},
            )
        if chosen_action is None or safety is None:
            return self._fail_task(task_id, "safety_denied_all_actions", {"candidates": candidates})
        if safety.decision == "requires_approval":
            with HarnessMemory(self.db_path) as memory:
                memory.transition_task(task_id, "waiting_approval", safety.as_dict())
            return {"task_id": task_id, "status": "waiting_approval", "safety": safety.as_dict()}
        if safety.decision == "dry_run_only":
            with HarnessMemory(self.db_path) as memory:
                memory.set_task_completed(task_id, {"dry_run_only": True, "chosen_action": chosen_action})
            return {"task_id": task_id, "status": "completed", "dry_run_only": True, "chosen_action": chosen_action, "safety": safety.as_dict()}

        with HarnessMemory(self.db_path) as memory:
            memory.transition_task(task_id, "executing", {"action_id": chosen_action})
        result = self._execute(chosen_action, task)
        self._record_execution(task_id, chosen_action, result)
        return {"task_id": task_id, "status": "completed" if result["success"] else "failed", "action": chosen_action, "safety": safety.as_dict(), "execution_result": result}

    def approve(self, task_id: str, *, approved_by: str = "user", reason: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            memory.add_approval(task_id, requested_by=None, approved_by=approved_by, decision="approved", scope="single_action", reason=reason)
            status = memory.get_task(task_id)["status"]
            if status == "waiting_approval":
                memory.transition_task(task_id, "ready", {"approved_by": approved_by})
            return {"task_id": task_id, "approved": True}

    def reject(self, task_id: str, *, rejected_by: str = "user", reason: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            memory.add_approval(task_id, requested_by=None, approved_by=rejected_by, decision="rejected", scope="single_action", reason=reason)
            status = memory.get_task(task_id)["status"]
            if status == "waiting_approval":
                memory.transition_task(task_id, "failed", {"rejected_by": rejected_by, "reason": reason})
            return {"task_id": task_id, "rejected": True}

    def recent_trace(self, limit: int = 5) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            return {"traces": memory.recent_traces(limit)}

    def list_preferences(self, user_id: str, *, scope: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            prefs = memory.list_preferences(user_id, scope=scope)
        return {"user_id": user_id, "preferences": prefs, "preference_map": preference_map(prefs)}

    def set_preference(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id") or "").strip()
        key = str(payload.get("key") or "").strip()
        value = payload.get("value")
        scope = str(payload.get("scope") or "global").strip()
        source = str(payload.get("source") or "explicit_user_request").strip()
        confidence = float(payload.get("confidence") if payload.get("confidence") is not None else 1.0)
        expires_at = payload.get("expires_at")
        with HarnessMemory(self.db_path) as memory:
            preference = memory.set_preference(user_id, key, value, scope=scope, source=source, confidence=confidence, expires_at=expires_at)
        return {"saved": True, "preference": preference}

    def delete_preference(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_id = str(payload.get("user_id") or "").strip()
        key = payload.get("key")
        scope = payload.get("scope")
        with HarnessMemory(self.db_path) as memory:
            deleted = memory.delete_preference(user_id, str(key).strip() if key else None, scope=str(scope).strip() if scope else None)
        return {"deleted": deleted, "user_id": user_id, "key": key, "scope": scope}

    def add_conversation_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            message = memory.add_conversation_message(
                user_id=str(payload.get("user_id") or "").strip(),
                channel_id=str(payload.get("channel_id")).strip() if payload.get("channel_id") is not None else None,
                message_id=str(payload.get("message_id")).strip() if payload.get("message_id") is not None else None,
                role=str(payload.get("role") or "").strip(),
                content=str(payload.get("content") or ""),
                linked_task_id=str(payload.get("linked_task_id")).strip() if payload.get("linked_task_id") else None,
            )
        return {"saved": True, "message": message}

    def recent_conversation(self, user_id: str, *, channel_id: str | None = None, limit: int = 20) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            messages = memory.recent_conversation_messages(user_id, channel_id=channel_id, limit=limit)
            task_refs = memory.recent_task_references(user_id, limit=10)
        return {"user_id": user_id, "channel_id": channel_id, "messages": messages, "task_references": task_refs}

    def add_task_reference(self, payload: dict[str, Any]) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            ref = memory.add_task_reference(
                task_id=str(payload.get("task_id") or "").strip(),
                user_id=str(payload.get("user_id") or "").strip(),
                short_label=str(payload.get("short_label") or "").strip(),
                status=str(payload.get("status") or "").strip(),
                result_summary=str(payload.get("result_summary") or ""),
            )
        return {"saved": True, "task_reference": ref}

    def memory_context(self, user_id: str, *, channel_id: str | None = None, message_limit: int = 12) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            prefs = memory.list_preferences(user_id)
            messages = memory.recent_conversation_messages(user_id, channel_id=channel_id, limit=message_limit)
            task_refs = memory.recent_task_references(user_id, limit=8)
        return {
            "user_id": user_id,
            "channel_id": channel_id,
            "user_preferences": preference_map(prefs),
            "recent_messages": [
                {"role": item.get("role"), "content": item.get("content_redacted"), "created_at": item.get("created_at")}
                for item in messages[-message_limit:]
            ],
            "recent_task_refs": [
                {
                    "task_id": item.get("task_id"),
                    "short_label": item.get("short_label"),
                    "status": item.get("status"),
                    "result_summary": item.get("result_summary"),
                    "created_at": item.get("created_at"),
                }
                for item in task_refs
            ],
        }

    def _load_task_spec(self, task_id: str) -> TaskSpec:
        with HarnessMemory(self.db_path) as memory:
            task = memory.get_task(task_id)
        return task_spec_from_dict(task["task_spec_json"], catalog=self.catalog)

    def _candidate_actions(self, task: TaskSpec) -> list[str]:
        return [action_id for action_id in task.allowed_actions if action_id not in task.blocked_actions]

    def _choose_first_allowed(self, task: TaskSpec, candidates: list[str]):
        chosen_requires_approval = None
        for action_id in candidates:
            safety = check_action_safety(task, action_id, catalog=self.catalog)
            if safety.decision in {"allow", "dry_run_only"}:
                return action_id, safety
            if safety.decision == "requires_approval" and chosen_requires_approval is None:
                chosen_requires_approval = (action_id, safety)
        if chosen_requires_approval:
            return chosen_requires_approval
        return None, None

    def _execute(self, action_id: str, task: TaskSpec) -> dict[str, Any]:
        params = task.context.get("params", {}) if isinstance(task.context.get("params", {}), dict) else {}
        action = self.catalog[action_id]
        if action.executor == "benchmark":
            return self.benchmark_executor.execute(action_id, params, {"task_id": task.task_id}).as_dict()
        if action.executor == "readonly_command":
            return self.readonly_command_executor.execute(action, params, {"task_id": task.task_id}).as_dict()
        return self.readonly_executor.execute(action_id, params, {"task_id": task.task_id}).as_dict()

    def _record_execution(self, task_id: str, chosen_action: str, result: dict[str, Any]) -> None:
        with HarnessMemory(self.db_path) as memory:
            memory.add_execution_result(task_id, result)
            memory.transition_task(task_id, "evaluating", {"success": result["success"]})
            if result["success"]:
                memory.set_task_completed(task_id, {"result": result})
            else:
                memory.add_trace(
                    task_id,
                    failure_trace(task_id, "execution_failed", action={"action_id": chosen_action}, actual=result, analysis={"error_type": result.get("error_type")}),
                )
                memory.set_task_failed(task_id, {"result": result})

    def _fail_task(self, task_id: str, bucket: str, analysis: dict[str, Any]) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            memory.add_trace(task_id, failure_trace(task_id, bucket, analysis=analysis))
            memory.set_task_failed(task_id, analysis)
        return {"task_id": task_id, "status": "failed", "failure_bucket": bucket, "analysis": analysis}


def _resolve_work_queue(work_queue: WorkQueue | None) -> tuple[WorkQueue | None, str | None]:
    if work_queue is not None:
        return work_queue, None
    try:
        return build_work_queue_from_env(), None
    except WorkQueueError as exc:
        return None, str(exc)
