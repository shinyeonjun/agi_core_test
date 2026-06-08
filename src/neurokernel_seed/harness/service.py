from __future__ import annotations

from pathlib import Path
from typing import Any

from neurokernel_seed.language.contracts import ALLOWED_PREFERENCE_KEYS, ALLOWED_PREFERENCE_SCOPES

from .action_catalog import ActionDefinition, build_action_catalog, public_catalog
from .activation import ActivationService, build_activation_config_from_env
from .capability_service import CapabilityProposalService
from .executors.benchmark import BenchmarkExecutor
from .executors.readonly_command import ReadOnlyCommandExecutor
from .executors.readonly_system import ReadOnlyExecutor
from .improvement import ImprovementService
from .interaction_contract import interaction_contract_from_runtime
from .memory import HarnessMemory
from .model_improvement import ModelImprovementService
from .preferences import preference_map
from .runtime_policy import RuntimeActionPolicy
from .safety_gate import check_action_safety
from .self_improvement import SelfImprovementService
from .task_spec import TaskSpec, task_spec_from_dict
from .trace import failure_trace, redact_text
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
        runtime_policy: RuntimeActionPolicy | None = None,
    ):
        self.db_path = Path(db_path)
        self.project_root = Path(project_root)
        self.catalog = catalog or build_action_catalog()
        self.runtime_policy = runtime_policy or RuntimeActionPolicy.from_env(self.project_root)
        self.readonly_executor = ReadOnlyExecutor(project_root=self.project_root, memory_path=self.db_path)
        self.readonly_command_executor = ReadOnlyCommandExecutor(project_root=self.project_root)
        self.benchmark_executor = BenchmarkExecutor(project_root=self.project_root)
        self.work_queue, self.queue_error = _resolve_work_queue(work_queue)
        self.work_items_service = WorkItemService(db_path=self.db_path, work_queue=self.work_queue, queue_error=self.queue_error)
        self.capability_service = CapabilityProposalService(db_path=self.db_path, catalog=self.catalog, work_items=self.work_items_service)
        self.improvement_service = ImprovementService(db_path=self.db_path, catalog=self.catalog, work_items=self.work_items_service)
        self.model_improvement_service = ModelImprovementService(db_path=self.db_path, work_items=self.work_items_service)
        self.self_improvement_service = SelfImprovementService(
            db_path=self.db_path,
            project_root=self.project_root,
            improvements=self.improvement_service,
            model_improvements=self.model_improvement_service,
            work_items=self.work_items_service,
        )
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

    def work_pipeline_status(self, *, limit: int = 20, stale_after_seconds: int = 300) -> dict[str, Any]:
        return self.work_items_service.pipeline_status(limit=limit, stale_after_seconds=stale_after_seconds)

    def add_work_note(self, work_id: str, *, actor: str = "api", note: str) -> dict[str, Any]:
        return self.work_items_service.add_note(work_id, actor=actor, note=note)

    def mark_work_discord_notified(self, work_id: str, *, actor: str = "discord-work-notifier", payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.work_items_service.mark_discord_notified(work_id, actor=actor, payload=payload)

    def transition_work_item(self, work_id: str, next_status: str, *, actor: str = "api", reason: str | None = None) -> dict[str, Any]:
        return self.work_items_service.transition(work_id, next_status, actor=actor, reason=reason)

    def enqueue_work_item(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        return self.work_items_service.enqueue(work_id, actor=actor, max_attempts=max_attempts)

    def retry_work_item(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        return self.work_items_service.retry(work_id, actor=actor, max_attempts=max_attempts)

    def promote_work_item_to_self_patch(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        return self.work_items_service.promote_to_self_patch(work_id, actor=actor, max_attempts=max_attempts)

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

    def analyze_improvements(self, *, min_gap_count: int = 2, lookback: int = 200) -> dict[str, Any]:
        return self.improvement_service.analyze(min_gap_count=min_gap_count, lookback=lookback)

    def propose_improvements(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        actor: str = "improvement_watchdog",
    ) -> dict[str, Any]:
        return self.improvement_service.propose_missing_output_gaps(
            min_gap_count=min_gap_count,
            lookback=lookback,
            actor=actor,
        )

    def analyze_model_improvements(
        self,
        *,
        min_known_runtime_candidates: int = 100,
        min_new_known_runtime_candidates: int = 50,
        min_runtime_ranking_groups: int = 10,
        target_runtime_top1: float = 0.65,
    ) -> dict[str, Any]:
        return self.model_improvement_service.analyze(
            min_known_runtime_candidates=min_known_runtime_candidates,
            min_new_known_runtime_candidates=min_new_known_runtime_candidates,
            min_runtime_ranking_groups=min_runtime_ranking_groups,
            target_runtime_top1=target_runtime_top1,
        )

    def propose_model_improvements(
        self,
        *,
        min_known_runtime_candidates: int = 100,
        min_new_known_runtime_candidates: int = 50,
        min_runtime_ranking_groups: int = 10,
        target_runtime_top1: float = 0.65,
        actor: str = "model_improvement_watchdog",
    ) -> dict[str, Any]:
        return self.model_improvement_service.propose_training_work(
            min_known_runtime_candidates=min_known_runtime_candidates,
            min_new_known_runtime_candidates=min_new_known_runtime_candidates,
            min_runtime_ranking_groups=min_runtime_ranking_groups,
            target_runtime_top1=target_runtime_top1,
            actor=actor,
        )

    def analyze_self_improvement(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        max_code_candidates: int = 5,
        min_code_score: int = 60,
    ) -> dict[str, Any]:
        return self.self_improvement_service.analyze(
            min_gap_count=min_gap_count,
            lookback=lookback,
            max_code_candidates=max_code_candidates,
            min_code_score=min_code_score,
        )

    def propose_self_improvement(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        max_code_candidates: int = 5,
        min_code_score: int = 60,
        max_proposals_per_cycle: int = 3,
        actor: str = "self_improvement_watchdog",
    ) -> dict[str, Any]:
        return self.self_improvement_service.propose(
            min_gap_count=min_gap_count,
            lookback=lookback,
            max_code_candidates=max_code_candidates,
            min_code_score=min_code_score,
            max_proposals_per_cycle=max_proposals_per_cycle,
            actor=actor,
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
        decisions, chosen, policy = self._evaluate_candidates(task, candidates)
        with HarnessMemory(self.db_path) as memory:
            memory.add_action_decision(
                task_id,
                0,
                [{"action_id": item["action_id"]} for item in decisions],
                {"action_id": chosen["action_id"]} if chosen else None,
                chosen["safety"] if chosen else {"decision": "deny", "reason": "no candidate"},
                model_score=_policy_decision_summary(policy),
                gate_trace={"runtime_policy": policy},
            )
            memory.record_experience(
                task_id=task_id,
                phase="dry_run",
                status="completed",
                decision_policy=_decision_policy_name(policy),
                model_used=bool(policy.get("model_used")),
                model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                before_state=self._experience_before_state(task, candidates, "dry_run"),
                after_state={"task_status": memory.get_task(task_id)["status"], "dry_run": True},
                outcome={"chosen_action": chosen["action_id"] if chosen else None, "executed": False},
                learning_masks={"candidate_outcomes_known": False, "safety_known": True, "selection_known": True},
                candidates=self._experience_candidates(task, decisions, chosen["action_id"] if chosen else None),
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

        decisions, chosen, policy = self._evaluate_candidates(task, candidates)
        chosen_action = chosen["action_id"] if chosen else None
        safety_payload = chosen["safety"] if chosen else None
        with HarnessMemory(self.db_path) as memory:
            memory.add_action_decision(
                task_id,
                0,
                [{"action_id": action_id} for action_id in candidates],
                {"action_id": chosen_action} if chosen_action else None,
                safety_payload if safety_payload else {"decision": "deny"},
                model_score=_policy_decision_summary(policy),
                gate_trace={"runtime_policy": policy},
            )
        if chosen_action is None or safety_payload is None:
            with HarnessMemory(self.db_path) as memory:
                memory.record_experience(
                    task_id=task_id,
                    phase="run",
                    status="failed",
                    decision_policy=_decision_policy_name(policy),
                    model_used=bool(policy.get("model_used")),
                    model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                    before_state=self._experience_before_state(task, candidates, "deciding"),
                    after_state={"task_status": "failed"},
                    outcome={"failure_bucket": "safety_denied_all_actions"},
                    learning_masks={"candidate_outcomes_known": False, "safety_known": True, "selection_known": True},
                    candidates=self._experience_candidates(task, decisions, None),
                )
            return self._fail_task(task_id, "safety_denied_all_actions", {"candidates": candidates})
        if safety_payload.get("decision") == "requires_approval":
            with HarnessMemory(self.db_path) as memory:
                memory.transition_task(task_id, "waiting_approval", safety_payload)
                memory.record_experience(
                    task_id=task_id,
                    phase="run",
                    status="waiting_approval",
                    decision_policy=_decision_policy_name(policy),
                    model_used=bool(policy.get("model_used")),
                    model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                    before_state=self._experience_before_state(task, candidates, "deciding"),
                    after_state={"task_status": "waiting_approval"},
                    outcome={"chosen_action": chosen_action, "approval_required": True},
                    learning_masks={"candidate_outcomes_known": False, "approval_known": True, "safety_known": True, "selection_known": True},
                    candidates=self._experience_candidates(task, decisions, chosen_action),
                )
            return {"task_id": task_id, "status": "waiting_approval", "safety": safety_payload}
        if safety_payload.get("decision") == "dry_run_only":
            with HarnessMemory(self.db_path) as memory:
                memory.set_task_completed(task_id, {"dry_run_only": True, "chosen_action": chosen_action})
                memory.record_experience(
                    task_id=task_id,
                    phase="run",
                    status="completed",
                    decision_policy=_decision_policy_name(policy),
                    model_used=bool(policy.get("model_used")),
                    model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                    before_state=self._experience_before_state(task, candidates, "deciding"),
                    after_state={"task_status": "completed", "dry_run_only": True},
                    outcome={"chosen_action": chosen_action, "dry_run_only": True},
                    learning_masks={"candidate_outcomes_known": False, "safety_known": True, "selection_known": True},
                    candidates=self._experience_candidates(task, decisions, chosen_action),
                )
            return {"task_id": task_id, "status": "completed", "dry_run_only": True, "chosen_action": chosen_action, "safety": safety_payload}

        with HarnessMemory(self.db_path) as memory:
            memory.transition_task(task_id, "executing", {"action_id": chosen_action})
        result = self._execute(chosen_action, task)
        self._record_execution(task_id, chosen_action, result)
        final_status = "completed" if result["success"] else "failed"
        with HarnessMemory(self.db_path) as memory:
            memory.record_experience(
                task_id=task_id,
                phase="run",
                status=final_status,
                decision_policy=_decision_policy_name(policy),
                model_used=bool(policy.get("model_used")),
                model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                before_state=self._experience_before_state(task, candidates, "executing"),
                after_state={"task_status": final_status, "action_id": chosen_action, "success": bool(result["success"])},
                outcome=self._experience_outcome(result),
                learning_masks={"candidate_outcomes_known": True, "safety_known": True, "selection_known": True},
                candidates=self._experience_candidates(task, decisions, chosen_action, executed_action=chosen_action, result=result),
            )
        return {"task_id": task_id, "status": final_status, "action": chosen_action, "safety": safety_payload, "execution_result": result}

    def probe_counterfactual_candidates(self, task_id: str, *, max_candidates: int = 4) -> dict[str, Any]:
        task = self._load_task_spec(task_id)
        candidates = self._candidate_actions(task)
        if not candidates:
            return self._fail_task(task_id, "no_candidate_actions", {"reason": "allowed_actions is empty"})

        decisions, chosen, policy = self._evaluate_candidates(task, candidates)
        probe_results: dict[str, dict[str, Any]] = {}
        skipped: list[dict[str, Any]] = []
        limit = max(1, int(max_candidates))
        for decision in decisions:
            if len(probe_results) >= limit:
                break
            action_id = str(decision.get("action_id") or "")
            safety = decision.get("safety") if isinstance(decision.get("safety"), dict) else {}
            action = self.catalog.get(action_id)
            if safety.get("decision") not in {"allow", "dry_run_only"}:
                skipped.append({"action_id": action_id, "reason": "safety_not_probeable", "safety_decision": safety.get("decision")})
                continue
            if action is None or action.side_effect or action.requires_approval or action.executor not in {"readonly_system", "readonly_command"}:
                skipped.append({"action_id": action_id, "reason": "not_readonly_probeable"})
                continue
            result = self._execute(action_id, task)
            probe_results[action_id] = result
            with HarnessMemory(self.db_path) as memory:
                memory.add_execution_result(task_id, result)

        chosen_action = str(chosen.get("action_id")) if chosen else None
        if chosen_action not in probe_results and probe_results:
            chosen_action = next(iter(probe_results))
        status = "completed" if probe_results else "failed"
        with HarnessMemory(self.db_path) as memory:
            memory.add_action_decision(
                task_id,
                0,
                [{"action_id": action_id} for action_id in candidates],
                {"action_id": chosen_action} if chosen_action else None,
                chosen.get("safety") if chosen else {"decision": "deny", "reason": "no probeable candidate"},
                model_score=_policy_decision_summary(policy),
                gate_trace={"runtime_policy": policy, "counterfactual_probe": {"max_candidates": limit}},
            )
            memory.record_experience(
                task_id=task_id,
                phase="counterfactual_probe",
                status=status,
                decision_policy="counterfactual_probe_safety_gated",
                model_used=bool(policy.get("model_used")),
                model_unavailable_reason=None if policy.get("model_used") else str(policy.get("reason") or "runtime_policy_unavailable"),
                before_state=self._experience_before_state(task, candidates, "counterfactual_probe"),
                after_state={"task_status": status, "probed_actions": list(probe_results)},
                outcome={
                    "chosen_action": chosen_action,
                    "probed_actions": list(probe_results),
                    "known_candidate_count": len(probe_results),
                    "skipped": skipped,
                },
                learning_masks={"candidate_outcomes_known": bool(probe_results), "safety_known": True, "selection_known": True},
                candidates=self._experience_counterfactual_candidates(task, decisions, chosen_action, probe_results),
            )
            if status == "completed":
                memory.set_task_completed(task_id, {"counterfactual_probe": True, "probed_actions": list(probe_results)})
            else:
                memory.set_task_failed(task_id, {"counterfactual_probe": True, "skipped": skipped})

        return {
            "task_id": task_id,
            "status": status,
            "chosen_action": chosen_action,
            "probed_actions": list(probe_results),
            "known_candidate_count": len(probe_results),
            "skipped": skipped,
        }

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
        if key not in ALLOWED_PREFERENCE_KEYS:
            raise ValueError(f"unknown preference key: {key}")
        if scope not in ALLOWED_PREFERENCE_SCOPES:
            raise ValueError(f"unknown preference scope: {scope}")
        if source != "explicit_user_request":
            raise ValueError("preference source must be explicit_user_request")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("preference confidence must be between 0 and 1")
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

    def link_message_to_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            updated = memory.link_conversation_message_to_task(
                user_id=str(payload.get("user_id") or "").strip(),
                channel_id=str(payload.get("channel_id")).strip() if payload.get("channel_id") is not None else None,
                message_id=str(payload.get("message_id")).strip() if payload.get("message_id") is not None else None,
                role=str(payload.get("role")).strip() if payload.get("role") is not None else None,
                task_id=str(payload.get("task_id") or "").strip(),
            )
        return {"updated": updated}

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

    def add_interaction_outcome(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = payload.get("task") if isinstance(payload.get("task"), dict) else None
        core_result = payload.get("core_result") if isinstance(payload.get("core_result"), dict) else None
        contract = interaction_contract_from_runtime(
            request_text=str(payload.get("request_text") or ""),
            response_text=str(payload.get("response_text") or ""),
            task=task,
            core_result=core_result,
        )
        with HarnessMemory(self.db_path) as memory:
            outcome = memory.add_interaction_outcome(
                source=str(payload.get("source") or "discord"),
                user_id=str(payload.get("user_id")).strip() if payload.get("user_id") is not None else None,
                channel_id=str(payload.get("channel_id")).strip() if payload.get("channel_id") is not None else None,
                user_message_id=str(payload.get("user_message_id")).strip() if payload.get("user_message_id") is not None else None,
                assistant_message_id=str(payload.get("assistant_message_id")).strip() if payload.get("assistant_message_id") is not None else None,
                task_id=str(payload.get("task_id")).strip() if payload.get("task_id") is not None else None,
                request_text=contract["request_text"],
                response_text=contract["response_text"],
                required_outputs=contract["required_outputs"],
                answered_outputs=contract["answered_outputs"],
                missing_outputs=contract["missing_outputs"],
                answer_quality=contract["answer_quality"],
                task_status=contract["task_status"],
                action_id=contract["action_id"],
                success=contract["success"],
            )
        return {"saved": True, "interaction_outcome": outcome}

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

    def _evaluate_candidates(self, task: TaskSpec, candidates: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
        decisions = []
        chosen_requires_approval = None
        deterministic_chosen = None
        for action_id in candidates:
            safety = check_action_safety(task, action_id, catalog=self.catalog)
            item = {"action_id": action_id, "safety": safety.as_dict()}
            decisions.append(item)
            if deterministic_chosen is None and safety.decision in {"allow", "dry_run_only"}:
                deterministic_chosen = item
            if safety.decision == "requires_approval" and chosen_requires_approval is None:
                chosen_requires_approval = item
        policy = self.runtime_policy.rank(task=task, decisions=decisions)
        for item in decisions:
            action_id = str(item["action_id"])
            item["model_score"] = (policy.get("scores") or {}).get(action_id, {"model_used": False, "reason": policy.get("reason")})
        if policy.get("model_used"):
            by_action = {str(item["action_id"]): item for item in decisions}
            for action_id in policy.get("ranked_actions") or []:
                item = by_action.get(str(action_id))
                safety = item.get("safety") if item else {}
                if isinstance(safety, dict) and safety.get("decision") in {"allow", "dry_run_only"}:
                    return decisions, item, policy
        return decisions, deterministic_chosen or chosen_requires_approval, policy

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

    def _experience_before_state(self, task: TaskSpec, candidates: list[str], phase: str) -> dict[str, Any]:
        params = task.context.get("params", {}) if isinstance(task.context.get("params", {}), dict) else {}
        return {
            "phase": phase,
            "task_status": phase,
            "task_id": task.task_id,
            "target": task.target,
            "mode": task.mode,
            "risk_level": task.risk_level,
            "requires_approval": task.requires_approval,
            "allowed_actions": list(task.allowed_actions),
            "blocked_actions": list(task.blocked_actions),
            "candidate_actions": list(candidates),
            "params_redacted": redact_text(str(params), max_chars=800),
        }

    def _experience_candidates(
        self,
        task: TaskSpec,
        decisions: list[dict[str, Any]],
        chosen_action: str | None,
        *,
        executed_action: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        params = task.context.get("params", {}) if isinstance(task.context.get("params", {}), dict) else {}
        items = []
        for index, decision in enumerate(decisions):
            action_id = str(decision["action_id"])
            executed = bool(executed_action and action_id == executed_action)
            known = bool(executed and result is not None)
            items.append(
                {
                    "candidate_index": index,
                    "action_id": action_id,
                    "params": params if action_id == chosen_action else {},
                    "safety_decision": decision.get("safety") or {},
                    "model_score": decision.get("model_score") or {"model_used": False, "reason": "runtime_policy_not_evaluated"},
                    "selected": action_id == chosen_action,
                    "executed": executed,
                    "execution_result_known": known,
                    "outcome": self._experience_outcome(result) if known and result else {},
                    "target_mask": {
                        "success": known,
                        "reward": known,
                        "duration_seconds": known,
                        "failure_present": known,
                    },
                }
            )
        return items

    def _experience_counterfactual_candidates(
        self,
        task: TaskSpec,
        decisions: list[dict[str, Any]],
        chosen_action: str | None,
        results_by_action: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        params = task.context.get("params", {}) if isinstance(task.context.get("params", {}), dict) else {}
        items = []
        for index, decision in enumerate(decisions):
            action_id = str(decision["action_id"])
            result = results_by_action.get(action_id)
            known = result is not None
            items.append(
                {
                    "candidate_index": index,
                    "action_id": action_id,
                    "params": params if known or action_id == chosen_action else {},
                    "safety_decision": decision.get("safety") or {},
                    "model_score": decision.get("model_score") or {"model_used": False, "reason": "runtime_policy_not_evaluated"},
                    "selected": action_id == chosen_action,
                    "executed": known,
                    "execution_result_known": known,
                    "outcome": self._experience_outcome(result) if known else {},
                    "target_mask": {
                        "success": known,
                        "reward": known,
                        "duration_seconds": known,
                        "failure_present": known,
                    },
                }
            )
        return items

    def _experience_outcome(self, result: dict[str, Any] | None) -> dict[str, Any]:
        if not result:
            return {}
        success = bool(result.get("success"))
        return {
            "action_id": result.get("action_id"),
            "success": success,
            "reward": 1.0 if success else -1.0,
            "duration_seconds": _duration_seconds(result.get("started_at"), result.get("ended_at")),
            "failure_present": not success or bool(result.get("error_type")),
            "error_type": result.get("error_type"),
        }


def _resolve_work_queue(work_queue: WorkQueue | None) -> tuple[WorkQueue | None, str | None]:
    if work_queue is not None:
        return work_queue, None
    try:
        return build_work_queue_from_env(), None
    except WorkQueueError as exc:
        return None, str(exc)


def _decision_policy_name(policy: dict[str, Any]) -> str:
    if policy.get("model_used"):
        return "runtime_model_ranked_safety_gated"
    return "deterministic_safety_first"


def _policy_decision_summary(policy: dict[str, Any]) -> dict[str, Any]:
    keys = ("model_used", "reason", "detail", "model_path", "schema_version", "top_action", "ranked_actions")
    return {key: policy.get(key) for key in keys if key in policy}


def _duration_seconds(started_at: Any, ended_at: Any) -> float:
    from datetime import datetime

    try:
        if not started_at or not ended_at:
            return 0.0
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
        return max(0.0, (end - start).total_seconds())
    except ValueError:
        return 0.0
