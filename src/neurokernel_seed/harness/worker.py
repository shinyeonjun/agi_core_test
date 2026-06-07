from __future__ import annotations

import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .memory import HarnessMemory
from .self_patch import CodexSelfPatchWorker, build_self_patch_config_from_env
from .work_queue import WorkQueue, build_work_queue_from_env


class SelfPatchRunner(Protocol):
    def run(self, *, job_id: str, work: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class WorkDispatcherConfig:
    db_path: Path = Path("data/harness.db")
    project_root: Path = Path(".")
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}-{int(time.time())}")
    queues: tuple[str, ...] = ("self_patch", "external_work")
    block_ms: int = 0
    once: bool = False
    idle_sleep_seconds: float = 0.0
    error_sleep_seconds: float = 5.0


class WorkDispatcher:
    def __init__(self, *, config: WorkDispatcherConfig, work_queue: WorkQueue | None = None, self_patch_runner: SelfPatchRunner | None = None):
        self.config = config
        self.work_queue = work_queue or build_work_queue_from_env()
        if self.work_queue is None:
            raise RuntimeError("work queue is not configured")
        self.self_patch_runner = self_patch_runner or CodexSelfPatchWorker(
            build_self_patch_config_from_env(project_root=self.config.project_root)
        )

    def run_forever(self) -> None:
        while True:
            try:
                count = self.run_once()
            except Exception as exc:
                if self.config.once:
                    raise
                _log_worker_error(self.config.worker_id, exc)
                time.sleep(self.config.error_sleep_seconds)
                continue
            if self.config.once:
                return
            if count == 0:
                time.sleep(self.config.idle_sleep_seconds)

    def run_once(self) -> int:
        messages = self.work_queue.read(list(self.config.queues), worker_id=self.config.worker_id, block_ms=self.config.block_ms, count=1)
        processed = 0
        for message in messages:
            self._process_message(message.queue_name, message.message_id, message.payload)
            processed += 1
        return processed

    def _process_message(self, queue_name: str, message_id: str, payload: dict[str, Any]) -> None:
        job_id = str(payload.get("job_id") or "")
        work_id = str(payload.get("work_id") or "")
        if not job_id or not work_id:
            self.work_queue.dead_letter({"payload": payload, "error": "missing job_id or work_id"})
            self.work_queue.ack(queue_name, message_id)
            return
        try:
            with HarnessMemory(self.config.db_path) as memory:
                job = memory.mark_work_job_running(job_id, worker_id=self.config.worker_id, actor=self.config.worker_id)
                work = memory.get_work_item(work_id)
                runnable = _mark_runnable_work_started(memory, work, job_id=job_id, queue_name=queue_name, actor=self.config.worker_id)
            if runnable:
                with HarnessMemory(self.config.db_path) as memory:
                    memory.mark_work_job_heartbeat(job_id, worker_id=self.config.worker_id, actor=self.config.worker_id, message="runner_started")
                result = self._run_work(job_id=job_id, queue_name=queue_name, work=work, payload=payload)
                with HarnessMemory(self.config.db_path) as memory:
                    memory.mark_work_job_heartbeat(job_id, worker_id=self.config.worker_id, actor=self.config.worker_id, message="runner_finished")
            else:
                result = {"status": "ignored", "reason": "work item is not in a runnable state", "job_status_before": job.get("status")}
            with HarnessMemory(self.config.db_path) as memory:
                _record_work_result(memory, work_id=work_id, work_type=str(work.get("type") or ""), job_id=job_id, result=result, actor=self.config.worker_id)
                memory.mark_work_job_completed(job_id, actor=self.config.worker_id, result=result)
            self.work_queue.ack(queue_name, message_id)
        except Exception as exc:
            with HarnessMemory(self.config.db_path) as memory:
                try:
                    is_self_patch = str(payload.get("work_type") or "") == "self_patch"
                    job = memory.mark_work_job_failed(job_id, error=str(exc), actor=self.config.worker_id, retryable=not is_self_patch)
                    if is_self_patch:
                        result = {
                            "status": "codex_failed",
                            "job_id": job_id,
                            "work_id": work_id,
                            "error": str(exc),
                            "next_required_action": "retry_self_patch_after_error",
                        }
                        memory.add_work_event(work_id, "self_patch_failed", actor=self.config.worker_id, payload={"job_id": job_id, "result": result})
                        memory.add_work_note(work_id, actor=self.config.worker_id, note=_result_note("self_patch", job_id, result))
                        _transition_if_allowed(memory, work_id, "reviewing", actor=self.config.worker_id, payload={"job_id": job_id, "result": result})
                    if job.get("status") == "dead_letter":
                        self.work_queue.dead_letter({"job_id": job_id, "work_id": work_id, "queue_name": queue_name, "error": str(exc), "payload": payload})
                        self.work_queue.ack(queue_name, message_id)
                except Exception:
                    self.work_queue.dead_letter({"job_id": job_id, "work_id": work_id, "queue_name": queue_name, "error": str(exc), "payload": payload})
                    self.work_queue.ack(queue_name, message_id)

    def _run_work(self, *, job_id: str, queue_name: str, work: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        work_type = str(work.get("type") or "")
        if work_type == "self_patch":
            return self.self_patch_runner.run(job_id=job_id, work=work, payload=payload)
        return {"status": "dispatched", "queue_name": queue_name, "note": _dispatch_note(work_type, job_id)}


def _dispatch_note(work_type: str, job_id: str) -> str:
    if work_type == "self_patch":
        return f"Self-patch job {job_id} reached the dispatcher. Next worker should implement, test, and attach the approved capability."
    if work_type == "external_work":
        return f"External work job {job_id} reached the dispatcher. Next worker should expand it into a project/research execution plan."
    return f"Work job {job_id} reached the dispatcher."


def _log_worker_error(worker_id: str, exc: Exception) -> None:
    print(f"[work-worker] worker_id={worker_id} transient_loop_error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _mark_runnable_work_started(memory: HarnessMemory, work: dict[str, Any], *, job_id: str, queue_name: str, actor: str) -> bool:
    work_id = str(work.get("work_id") or "")
    work_type = str(work.get("type") or "")
    status = str(work.get("status") or "")
    if work_type != "self_patch":
        if status == "accepted":
            memory.transition_work_item(work_id, "planned", actor=actor, payload={"job_id": job_id, "queue_name": queue_name})
            return True
        if status in {"planned", "running", "blocked", "waiting_approval"}:
            memory.add_work_event(work_id, "worker_observed_planned_work", actor=actor, payload={"job_id": job_id, "queue_name": queue_name, "status": status})
            return True
        memory.add_work_event(work_id, "worker_observed_non_runnable_status", actor=actor, payload={"job_id": job_id, "status": status})
        return False
    if status in {"accepted", "planned"}:
        memory.transition_work_item(work_id, "running", actor=actor, payload={"job_id": job_id, "queue_name": queue_name})
        return True
    if status == "running":
        memory.add_work_event(work_id, "worker_resumed_running_work", actor=actor, payload={"job_id": job_id, "queue_name": queue_name})
        return True
    memory.add_work_event(work_id, "worker_observed_non_runnable_status", actor=actor, payload={"job_id": job_id, "status": status})
    return False


def _record_work_result(memory: HarnessMemory, *, work_id: str, work_type: str, job_id: str, result: dict[str, Any], actor: str) -> None:
    status = str(result.get("status") or "unknown")
    memory.add_work_note(work_id, actor=actor, note=_result_note(work_type, job_id, result))
    if work_type != "self_patch":
        return
    if status == "patch_ready":
        _transition_if_allowed(memory, work_id, "waiting_approval", actor=actor, payload={"job_id": job_id, "result": result})
    elif status in {"test_failed", "diff_check_failed", "codex_failed_no_patch"}:
        _transition_if_allowed(memory, work_id, "reviewing", actor=actor, payload={"job_id": job_id, "result": result})
    elif status in {"no_patch", "ignored"}:
        _transition_if_allowed(memory, work_id, "blocked", actor=actor, payload={"job_id": job_id, "result": result})
    else:
        _transition_if_allowed(memory, work_id, "blocked", actor=actor, payload={"job_id": job_id, "result": result})


def _transition_if_allowed(memory: HarnessMemory, work_id: str, next_status: str, *, actor: str, payload: dict[str, Any]) -> None:
    try:
        memory.transition_work_item(work_id, next_status, actor=actor, payload=payload)
    except ValueError:
        memory.add_work_event(work_id, "worker_transition_skipped", actor=actor, payload={"next_status": next_status, "reason": "invalid_transition"})


def _result_note(work_type: str, job_id: str, result: dict[str, Any]) -> str:
    status = str(result.get("status") or "unknown")
    if work_type == "self_patch":
        patch_path = result.get("patch_path") or ""
        changed = ", ".join(result.get("changed_files") or [])
        analysis = result.get("failure_analysis") if isinstance(result.get("failure_analysis"), dict) else {}
        primary_failure = analysis.get("primary_failure")
        summary = analysis.get("summary")
        failure = f" primary_failure={primary_failure}." if primary_failure else ""
        summary_text = f" summary={summary}" if summary else ""
        return f"Self-patch job {job_id} finished with status={status}.{failure} patch={patch_path}. changed_files={changed or 'none'}.{summary_text}"
    return str(result.get("note") or _dispatch_note(work_type, job_id))


def serve_worker(*, db_path: str | Path = "data/harness.db", project_root: str | Path = ".", worker_id: str | None = None, queues: list[str] | None = None, block_ms: int | None = None, once: bool = False) -> None:
    config = WorkDispatcherConfig(
        db_path=Path(db_path),
        project_root=Path(project_root),
        worker_id=worker_id or f"worker-{socket.gethostname()}-{int(time.time())}",
        queues=tuple(queues or ["self_patch", "external_work"]),
        block_ms=int(block_ms if block_ms is not None else os.environ.get("NEUROKERNEL_WORKER_BLOCK_MS", "0")),
        once=once,
    )
    WorkDispatcher(config=config).run_forever()
