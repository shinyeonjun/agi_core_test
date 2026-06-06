from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .memory import HarnessMemory
from .work_queue import WorkQueue, build_work_queue_from_env


@dataclass(frozen=True)
class WorkDispatcherConfig:
    db_path: Path = Path("data/harness.db")
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}-{int(time.time())}")
    queues: tuple[str, ...] = ("self_patch", "external_work")
    block_ms: int = 0
    once: bool = False
    idle_sleep_seconds: float = 0.0


class WorkDispatcher:
    def __init__(self, *, config: WorkDispatcherConfig, work_queue: WorkQueue | None = None):
        self.config = config
        self.work_queue = work_queue or build_work_queue_from_env()
        if self.work_queue is None:
            raise RuntimeError("work queue is not configured")

    def run_forever(self) -> None:
        while True:
            count = self.run_once()
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
                status = str(work.get("status") or "")
                if status == "accepted":
                    memory.transition_work_item(work_id, "planned", actor=self.config.worker_id, payload={"job_id": job_id, "queue_name": queue_name})
                elif status not in {"planned", "running", "blocked", "waiting_approval"}:
                    memory.add_work_event(work_id, "worker_observed_non_runnable_status", actor=self.config.worker_id, payload={"job_id": job_id, "status": status})
                memory.add_work_note(
                    work_id,
                    actor=self.config.worker_id,
                    note=_dispatch_note(str(work.get("type") or ""), job_id),
                )
                memory.mark_work_job_completed(job_id, actor=self.config.worker_id, result={"dispatched": True, "job_status_before": job.get("status")})
            self.work_queue.ack(queue_name, message_id)
        except Exception as exc:
            with HarnessMemory(self.config.db_path) as memory:
                try:
                    job = memory.mark_work_job_failed(job_id, error=str(exc), actor=self.config.worker_id, retryable=True)
                    if job.get("status") == "dead_letter":
                        self.work_queue.dead_letter({"job_id": job_id, "work_id": work_id, "queue_name": queue_name, "error": str(exc), "payload": payload})
                        self.work_queue.ack(queue_name, message_id)
                except Exception:
                    self.work_queue.dead_letter({"job_id": job_id, "work_id": work_id, "queue_name": queue_name, "error": str(exc), "payload": payload})
                    self.work_queue.ack(queue_name, message_id)


def _dispatch_note(work_type: str, job_id: str) -> str:
    if work_type == "self_patch":
        return f"Self-patch job {job_id} reached the dispatcher. Next worker should implement, test, and attach the approved capability."
    if work_type == "external_work":
        return f"External work job {job_id} reached the dispatcher. Next worker should expand it into a project/research execution plan."
    return f"Work job {job_id} reached the dispatcher."


def serve_worker(*, db_path: str | Path = "data/harness.db", worker_id: str | None = None, queues: list[str] | None = None, block_ms: int | None = None, once: bool = False) -> None:
    config = WorkDispatcherConfig(
        db_path=Path(db_path),
        worker_id=worker_id or f"worker-{socket.gethostname()}-{int(time.time())}",
        queues=tuple(queues or ["self_patch", "external_work"]),
        block_ms=int(block_ms if block_ms is not None else os.environ.get("NEUROKERNEL_WORKER_BLOCK_MS", "0")),
        once=once,
    )
    WorkDispatcher(config=config).run_forever()
