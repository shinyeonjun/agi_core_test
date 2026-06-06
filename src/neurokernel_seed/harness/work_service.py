from __future__ import annotations

from pathlib import Path
from typing import Any

from .ids import new_id
from .memory import HarnessMemory
from .work_queue import WorkQueue, queue_name_for_work_type


class WorkItemService:
    def __init__(self, *, db_path: str | Path, work_queue: WorkQueue | None = None, queue_error: str | None = None):
        self.db_path = Path(db_path)
        self.work_queue = work_queue
        self.queue_error = queue_error

    def queue_health(self) -> dict[str, Any]:
        if self.work_queue is None:
            return {"available": False, "reason": self.queue_error or "queue not configured"}
        try:
            return self.work_queue.health()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}

    def list_items(self, *, limit: int = 20, status: str | None = None, work_type: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            items = memory.list_work_items(limit=limit, status=status, work_type=work_type)
        return {"work_items": items}

    def get_item(self, work_id: str) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            item = memory.get_work_item(work_id)
            events = memory.work_events(work_id)
            notes = memory.work_notes(work_id)
            jobs = memory.list_work_jobs(work_id=work_id, limit=20)
            children = memory.child_work_items(work_id, limit=20)
        return {"work_item": item, "events": events, "notes": notes, "jobs": jobs, "child_work_items": children}

    def list_jobs(
        self,
        *,
        limit: int = 20,
        work_id: str | None = None,
        status: str | None = None,
        queue_name: str | None = None,
    ) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            jobs = memory.list_work_jobs(limit=limit, work_id=work_id, status=status, queue_name=queue_name)
        return {"jobs": jobs}

    def add_note(self, work_id: str, *, actor: str = "api", note: str) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            note_row = memory.add_work_note(work_id, actor=actor, note=note)
            item = memory.get_work_item(work_id)
        return {"work_item": item, "note": note_row}

    def transition(self, work_id: str, next_status: str, *, actor: str = "api", reason: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            item = memory.transition_work_item(work_id, next_status, actor=actor, payload={"reason": reason})
            events = memory.work_events(work_id)
        payload: dict[str, Any] = {"work_item": item, "events": events}
        if next_status == "accepted":
            payload["queue"] = self.enqueue(work_id, actor=actor)
        return payload

    def enqueue(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            work = memory.get_work_item(work_id)
            existing = memory.find_active_work_job(work_id)
            if existing:
                return {"queued": False, "reason": "active_job_exists", "job": existing}

            work_type = str(work.get("type") or "external_work")
            queue_name = queue_name_for_work_type(work_type)
            job_id = new_id("job", work_type)
            payload = _work_job_payload(job_id=job_id, work_id=work_id, work=work, work_type=work_type, queue_name=queue_name)
            memory.create_work_job(
                job_id=job_id,
                work_id=work_id,
                queue_name=queue_name,
                status="created",
                priority=str(work.get("priority") or "medium"),
                max_attempts=max_attempts,
                payload=payload,
                actor=actor,
            )
            if self.work_queue is None:
                memory.mark_work_job_enqueue_failed(job_id, error=self.queue_error or "queue not configured", actor=actor)
                return {"queued": False, "reason": self.queue_error or "queue not configured", "job": memory.get_work_job(job_id)}
            try:
                message_id = self.work_queue.enqueue(queue_name, payload)
                job = memory.mark_work_job_queued(job_id, redis_message_id=message_id, actor=actor)
                return {"queued": True, "queue_name": queue_name, "message_id": message_id, "job": job}
            except Exception as exc:
                job = memory.mark_work_job_enqueue_failed(job_id, error=str(exc), actor=actor)
                return {"queued": False, "reason": str(exc), "job": job}

    def retry(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            work = memory.get_work_item(work_id)
            if str(work.get("type") or "") != "self_patch":
                raise ValueError("only self_patch work items can be retried")
            status = str(work.get("status") or "")
            if status not in {"reviewing", "blocked", "failed"}:
                raise ValueError(f"work item is not retryable from status: {status}")
            existing = memory.find_active_work_job(work_id)
            if existing:
                return {"queued": False, "reason": "active_job_exists", "job": existing, "work_item": work}

            events = memory.work_events(work_id)
            previous_result = _latest_self_patch_result(events)
            work_type = str(work.get("type") or "self_patch")
            queue_name = queue_name_for_work_type(work_type)
            job_id = new_id("job", work_type)
            payload = _work_job_payload(
                job_id=job_id,
                work_id=work_id,
                work=work,
                work_type=work_type,
                queue_name=queue_name,
                extra={
                    "retry": {
                        "requested_by": actor,
                        "previous_status": status,
                        "previous_result": previous_result,
                    }
                },
            )
            memory.create_work_job(
                job_id=job_id,
                work_id=work_id,
                queue_name=queue_name,
                status="created",
                priority=str(work.get("priority") or "medium"),
                max_attempts=max_attempts,
                payload=payload,
                actor=actor,
            )
            memory.transition_work_item(work_id, "running", actor=actor, payload={"reason": "retry", "job_id": job_id, "previous_result": previous_result})
            if self.work_queue is None:
                job = memory.mark_work_job_enqueue_failed(job_id, error=self.queue_error or "queue not configured", actor=actor)
                _transition_back_after_retry_enqueue_failure(memory, work_id, status, actor=actor, job_id=job_id)
                return {"queued": False, "reason": self.queue_error or "queue not configured", "job": job, "work_item": memory.get_work_item(work_id)}
            try:
                message_id = self.work_queue.enqueue(queue_name, payload)
                job = memory.mark_work_job_queued(job_id, redis_message_id=message_id, actor=actor)
                return {"queued": True, "queue_name": queue_name, "message_id": message_id, "job": job, "work_item": memory.get_work_item(work_id)}
            except Exception as exc:
                job = memory.mark_work_job_enqueue_failed(job_id, error=str(exc), actor=actor)
                _transition_back_after_retry_enqueue_failure(memory, work_id, status, actor=actor, job_id=job_id)
                return {"queued": False, "reason": str(exc), "job": job, "work_item": memory.get_work_item(work_id)}

    def promote_to_self_patch(self, work_id: str, *, actor: str = "api", max_attempts: int = 3) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            parent = memory.get_work_item(work_id)
            parent_type = str(parent.get("type") or "")
            parent_status = str(parent.get("status") or "")
            if parent_type != "external_work":
                raise ValueError("only external_work items can be promoted to self_patch")
            if parent_status in {"rejected", "cancelled", "completed", "archived"}:
                raise ValueError(f"work item is not promotable from status: {parent_status}")

            existing_children = memory.child_work_items(work_id, work_type="self_patch", limit=10)
            reusable_child = next(
                (child for child in existing_children if str(child.get("status") or "") not in {"rejected", "cancelled", "failed", "archived"}),
                None,
            )
            if reusable_child:
                existing_job = memory.find_active_work_job(str(reusable_child["work_id"]))
                return {
                    "promoted": False,
                    "reason": "child_self_patch_exists",
                    "parent_work_item": parent,
                    "child_work_item": reusable_child,
                    "queue": {"queued": False, "reason": "active_job_exists", "job": existing_job} if existing_job else None,
                }

            metadata = parent.get("metadata_json") if isinstance(parent.get("metadata_json"), dict) else {}
            child = memory.create_work_item(
                work_id=new_id("work", str(parent.get("title") or "self_patch")),
                work_type="self_patch",
                title=str(parent.get("title") or "self_patch"),
                goal=str(parent.get("goal") or ""),
                status="accepted",
                priority=str(parent.get("priority") or "medium"),
                risk_level=str(parent.get("risk_level") or "low"),
                owner_user_id=parent.get("owner_user_id"),
                channel_id=parent.get("channel_id"),
                source_message_id=parent.get("source_message_id"),
                parent_work_id=work_id,
                linked_entity_type="promoted_external_work",
                linked_entity_id=work_id,
                route_reason=f"promoted from external_work {work_id}",
                confidence=float(parent.get("confidence") or 0.0),
                metadata={
                    "promoted_from_work_id": work_id,
                    "parent_status_at_promotion": parent_status,
                    "parent_metadata": metadata,
                    "deliverables": metadata.get("deliverables") or ["implementation_patch", "tests", "activation_candidate"],
                    "open_questions": metadata.get("open_questions") or [],
                },
                actor=actor,
            )
            memory.add_work_event(work_id, "promoted_to_self_patch", actor=actor, payload={"child_work_id": child["work_id"]})
            memory.add_work_note(work_id, actor=actor, note=f"개발 작업으로 전환됨: {child['work_id']}")
            parent_after = memory.get_work_item(work_id)

        queue_result = self.enqueue(str(child["work_id"]), actor=actor, max_attempts=max_attempts)
        return {
            "promoted": True,
            "parent_work_item": parent_after,
            "child_work_item": child,
            "queue": queue_result,
        }

    def create_from_route(
        self,
        *,
        user_text: str,
        route_decision: dict[str, Any],
        user_id: str | None = None,
        channel_id: str | None = None,
        source_message_id: str | None = None,
    ) -> dict[str, Any]:
        route = str(route_decision.get("route") or "clarify")
        if route in {"runtime_task", "unsafe", "clarify"}:
            return {"created": False, "route": route, "route_decision": route_decision}
        work_item = route_decision.get("work_item")
        if not isinstance(work_item, dict):
            return {"created": False, "route": "clarify", "route_decision": route_decision}
        with HarnessMemory(self.db_path) as memory:
            row = memory.create_work_item(
                work_id=new_id("work", str(work_item.get("title") or route)),
                work_type=str(work_item.get("type") or route),
                title=str(work_item.get("title") or route),
                goal=str(work_item.get("goal") or user_text),
                status="proposed",
                priority=str(work_item.get("priority") or "medium"),
                risk_level=str(work_item.get("risk_level") or "low"),
                owner_user_id=user_id,
                channel_id=channel_id,
                source_message_id=source_message_id,
                route_reason=str(route_decision.get("reason") or ""),
                confidence=float(route_decision.get("confidence") or 0.0),
                metadata={
                    "user_text": user_text[:1000],
                    "route_decision": route_decision,
                    "deliverables": work_item.get("deliverables") or [],
                    "open_questions": work_item.get("open_questions") or [],
                },
                actor=user_id or "language_organ",
            )
        return {"created": True, "route": route, "work_item": row, "route_decision": route_decision}


def _work_job_payload(*, job_id: str, work_id: str, work: dict[str, Any], work_type: str, queue_name: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "work_id": work_id,
        "work_type": work_type,
        "queue_name": queue_name,
        "priority": work.get("priority") or "medium",
        "risk_level": work.get("risk_level") or "low",
        "title": work.get("title") or "",
    }
    if extra:
        payload.update(extra)
    return payload


def _latest_self_patch_result(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event_type") not in {"job_completed", "self_patch_failed"}:
            continue
        payload = event.get("payload_json")
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            return _trim_self_patch_result(result)
    return {}


def _trim_self_patch_result(result: dict[str, Any]) -> dict[str, Any]:
    trimmed: dict[str, Any] = {
        "status": result.get("status"),
        "job_id": result.get("job_id"),
        "patch_path": result.get("patch_path"),
        "changed_files": result.get("changed_files") or [],
        "error": result.get("error"),
        "next_required_action": result.get("next_required_action"),
    }
    failure_analysis = result.get("failure_analysis")
    if isinstance(failure_analysis, dict):
        trimmed["failure_analysis"] = {
            "schema_version": failure_analysis.get("schema_version"),
            "primary_failure": failure_analysis.get("primary_failure"),
            "summary": failure_analysis.get("summary"),
            "retryable": failure_analysis.get("retryable"),
            "next_step": failure_analysis.get("next_step"),
            "signals": failure_analysis.get("signals") if isinstance(failure_analysis.get("signals"), dict) else {},
        }
    for key in ("test", "diff_check", "codex"):
        value = result.get(key)
        if isinstance(value, dict):
            trimmed[key] = {
                "returncode": value.get("returncode"),
                "stdout_tail": str(value.get("stdout_tail") or "")[-2_000:],
                "stderr_tail": str(value.get("stderr_tail") or "")[-2_000:],
            }
    return trimmed


def _transition_back_after_retry_enqueue_failure(memory: HarnessMemory, work_id: str, previous_status: str, *, actor: str, job_id: str) -> None:
    try:
        memory.transition_work_item(work_id, previous_status, actor=actor, payload={"reason": "retry enqueue failed", "job_id": job_id})
    except ValueError:
        memory.add_work_event(work_id, "retry_status_restore_failed", actor=actor, payload={"previous_status": previous_status, "job_id": job_id})
