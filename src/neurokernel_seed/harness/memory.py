from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .memory_rows import required_text as _required_text
from .memory_rows import row_to_dict as _row
from .memory_rows import to_json as _json
from .memory_schema import init_schema
from .ids import new_id
from .state_machine import assert_transition
from .task_spec import TaskSpec
from .trace import redact_text
from .work_states import allowed_proposal_next_statuses, allowed_work_next_statuses


class HarnessMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HarnessMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_schema(self) -> None:
        init_schema(self.conn)

    def create_task(self, spec: TaskSpec, *, created_by: str = "system", source: str = "cli") -> None:
        self.conn.execute(
            """
            INSERT INTO tasks(task_id, created_by, source, goal, target, status, risk_level, requires_approval, task_spec_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec.task_id,
                created_by,
                source,
                spec.goal,
                spec.target,
                "created",
                spec.risk_level,
                int(spec.requires_approval),
                _json(spec.as_dict()),
            ),
        )
        self.add_event(spec.task_id, "created", spec.as_dict())
        self.conn.commit()

    def transition_task(self, task_id: str, next_status: str, payload: dict[str, Any] | None = None) -> None:
        current = self.get_task(task_id)["status"]
        assert_transition(str(current), next_status)
        self.conn.execute("UPDATE tasks SET status=? WHERE task_id=?", (next_status, task_id))
        self.add_event(task_id, next_status, payload or {})
        self.conn.commit()

    def set_task_failed(self, task_id: str, reason: dict[str, Any]) -> None:
        self.conn.execute("UPDATE tasks SET status='failed' WHERE task_id=?", (task_id,))
        self.add_event(task_id, "failed", reason)
        self.conn.commit()

    def set_task_completed(self, task_id: str, payload: dict[str, Any]) -> None:
        self.conn.execute("UPDATE tasks SET status='completed' WHERE task_id=?", (task_id,))
        self.add_event(task_id, "completed", payload)
        self.conn.commit()

    def add_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO task_events(task_id, event_type, payload_json) VALUES (?, ?, ?)", (task_id, event_type, _json(payload)))
        self._insert_agent_event(
            event_type=f"task.{event_type}",
            source="task_lifecycle",
            payload=payload,
            task_id=task_id,
        )

    def add_action_decision(self, task_id: str, step: int, candidate_actions: list[dict[str, Any]], chosen_action: dict[str, Any] | None, safety_decision: dict[str, Any], model_score: dict[str, Any] | None = None, gate_trace: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            """
            INSERT INTO action_decisions(task_id, step, candidate_actions_json, chosen_action_json, model_score_json, safety_decision_json, gate_trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, step, _json(candidate_actions), _json(chosen_action) if chosen_action else None, _json(model_score or {}), _json(safety_decision), _json(gate_trace or {})),
        )
        self.conn.commit()

    def add_execution_result(self, task_id: str, result: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO execution_results(task_id, action_id, started_at, ended_at, success, stdout_redacted, stderr_redacted, result_json, error_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                result["action_id"],
                result["started_at"],
                result["ended_at"],
                int(bool(result["success"])),
                result.get("stdout_redacted", ""),
                result.get("stderr_redacted", ""),
                _json(result.get("result", {})),
                result.get("error_type"),
            ),
        )
        self.conn.commit()

    def add_approval(self, task_id: str, *, requested_by: str | None, approved_by: str | None, decision: str, scope: str, reason: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO approvals(task_id, requested_by, approved_by, decision, scope, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, requested_by, approved_by, decision, scope, reason),
        )
        self.add_event(task_id, "approval_" + decision, {"scope": scope, "reason": reason})
        self.conn.commit()

    def has_approval(self, task_id: str, *, scope: str = "single_action") -> bool:
        row = self.conn.execute("SELECT COUNT(*) FROM approvals WHERE task_id=? AND decision='approved' AND scope=?", (task_id, scope)).fetchone()
        return bool(row[0])

    def add_trace(self, task_id: str, trace: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO traces(task_id, failure_bucket, state_json, candidate_actions_json, chosen_action_json, expected_result_json, actual_result_json, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                trace.get("failure_bucket", "unknown"),
                _json(trace.get("state", {})),
                _json(trace.get("candidate_actions", [])),
                _json(trace.get("chosen_action", {})),
                _json(trace.get("expected_result", {})),
                _json(trace.get("actual_result", {})),
                _json(trace.get("analysis", {})),
            ),
        )
        self.conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        return _row(row)

    def list_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(row) for row in rows]

    def recent_traces(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(row) for row in rows]

    def events_for_task(self, task_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM task_events WHERE task_id=? ORDER BY event_id", (task_id,)).fetchall()
        return [_row(row) for row in rows]

    def append_agent_event(
        self,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
        actor_id: str | None = None,
        work_id: str | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
        proposal_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        idempotency_key: str | None = None,
        status: str = "recorded",
    ) -> dict[str, Any]:
        event_id = self._insert_agent_event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            actor_id=actor_id,
            work_id=work_id,
            job_id=job_id,
            task_id=task_id,
            proposal_id=proposal_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            idempotency_key=idempotency_key,
            status=status,
        )
        self.conn.commit()
        return self.get_agent_event(event_id)

    def get_agent_event(self, event_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM agent_events WHERE event_id=?", (_required_text(event_id, "event_id"),)).fetchone()
        if row is None:
            raise KeyError(f"unknown agent event: {event_id}")
        return _row(row)

    def recent_agent_events(
        self,
        *,
        limit: int = 50,
        work_id: str | None = None,
        job_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 300))
        clauses = []
        params: list[Any] = []
        if work_id:
            clauses.append("work_id=?")
            params.append(work_id)
        if job_id:
            clauses.append("job_id=?")
            params.append(job_id)
        if event_type:
            clauses.append("event_type=?")
            params.append(event_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM agent_events{where} ORDER BY created_at DESC, event_id DESC LIMIT ?", (*params, limit)).fetchall()
        return [_row(row) for row in rows]

    def _insert_agent_event(
        self,
        *,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        actor_id: str | None = None,
        work_id: str | None = None,
        job_id: str | None = None,
        task_id: str | None = None,
        proposal_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        idempotency_key: str | None = None,
        status: str = "recorded",
    ) -> str:
        event_id = new_id("evt", event_type)
        if idempotency_key:
            existing = self.conn.execute("SELECT event_id FROM agent_events WHERE idempotency_key=?", (idempotency_key,)).fetchone()
            if existing is not None:
                return str(existing["event_id"])
        self.conn.execute(
            """
            INSERT INTO agent_events(
              event_id, idempotency_key, event_type, source, actor_id, work_id, job_id,
              task_id, proposal_id, correlation_id, causation_id, payload_json, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                idempotency_key,
                _required_text(event_type, "event_type"),
                _required_text(source, "source"),
                actor_id,
                work_id,
                job_id,
                task_id,
                proposal_id,
                correlation_id,
                causation_id,
                _json(payload),
                _required_text(status, "status"),
            ),
        )
        return event_id

    def set_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        *,
        scope: str = "global",
        source: str = "explicit_user_request",
        confidence: float = 1.0,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        user_id = _required_text(user_id, "user_id")
        key = _required_text(key, "key")
        scope = _required_text(scope, "scope")
        self.conn.execute(
            """
            INSERT INTO user_preferences(user_id, key, value_json, scope, source, confidence, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, key, scope) DO UPDATE SET
              value_json=excluded.value_json,
              source=excluded.source,
              confidence=excluded.confidence,
              updated_at=CURRENT_TIMESTAMP,
              expires_at=excluded.expires_at
            """,
            (user_id, key, _json(value), scope, source, float(confidence), expires_at),
        )
        self.conn.commit()
        return self.get_preference(user_id, key, scope=scope)

    def get_preference(self, user_id: str, key: str, *, scope: str = "global") -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM user_preferences WHERE user_id=? AND key=? AND scope=?",
            (user_id, key, scope),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown preference: {user_id}/{scope}/{key}")
        return _row(row)

    def list_preferences(self, user_id: str, *, scope: str | None = None) -> list[dict[str, Any]]:
        if scope:
            rows = self.conn.execute(
                "SELECT * FROM user_preferences WHERE user_id=? AND scope=? ORDER BY key",
                (user_id, scope),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM user_preferences WHERE user_id=? ORDER BY scope, key",
                (user_id,),
            ).fetchall()
        return [_row(row) for row in rows]

    def delete_preference(self, user_id: str, key: str | None = None, *, scope: str | None = None) -> int:
        if key and scope:
            cur = self.conn.execute("DELETE FROM user_preferences WHERE user_id=? AND key=? AND scope=?", (user_id, key, scope))
        elif key:
            cur = self.conn.execute("DELETE FROM user_preferences WHERE user_id=? AND key=?", (user_id, key))
        elif scope:
            cur = self.conn.execute("DELETE FROM user_preferences WHERE user_id=? AND scope=?", (user_id, scope))
        else:
            cur = self.conn.execute("DELETE FROM user_preferences WHERE user_id=?", (user_id,))
        self.conn.commit()
        return int(cur.rowcount)

    def add_conversation_message(
        self,
        *,
        user_id: str,
        role: str,
        content: str,
        channel_id: str | None = None,
        message_id: str | None = None,
        linked_task_id: str | None = None,
    ) -> dict[str, Any]:
        user_id = _required_text(user_id, "user_id")
        role = _required_text(role, "role")
        content_redacted = redact_text(str(content), max_chars=2_000)
        cur = self.conn.execute(
            """
            INSERT INTO conversation_messages(user_id, channel_id, message_id, role, content_redacted, linked_task_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, channel_id, message_id, role, content_redacted, linked_task_id),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM conversation_messages WHERE id=?", (cur.lastrowid,)).fetchone()
        return _row(row)

    def recent_conversation_messages(self, user_id: str, *, channel_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 300))
        if channel_id:
            rows = self.conn.execute(
                """
                SELECT * FROM conversation_messages
                WHERE user_id=? AND channel_id=?
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, channel_id, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM conversation_messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return list(reversed([_row(row) for row in rows]))

    def add_task_reference(
        self,
        *,
        task_id: str,
        user_id: str,
        short_label: str,
        status: str,
        result_summary: str = "",
    ) -> dict[str, Any]:
        cur = self.conn.execute(
            """
            INSERT INTO task_references(task_id, user_id, short_label, status, result_summary)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _required_text(task_id, "task_id"),
                _required_text(user_id, "user_id"),
                _required_text(short_label, "short_label")[:120],
                _required_text(status, "status"),
                redact_text(str(result_summary), max_chars=1_000),
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM task_references WHERE id=?", (cur.lastrowid,)).fetchone()
        return _row(row)

    def recent_task_references(self, user_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM task_references WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, max(1, min(int(limit), 50))),
        ).fetchall()
        return [_row(row) for row in rows]

    def create_work_item(
        self,
        *,
        work_id: str,
        work_type: str,
        title: str,
        goal: str,
        status: str = "proposed",
        priority: str = "medium",
        risk_level: str = "low",
        owner_user_id: str | None = None,
        channel_id: str | None = None,
        source_message_id: str | None = None,
        parent_work_id: str | None = None,
        linked_entity_type: str | None = None,
        linked_entity_id: str | None = None,
        route_reason: str = "",
        confidence: float = 0.0,
        metadata: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO work_items(
              work_id, type, title, goal, status, priority, risk_level,
              owner_user_id, channel_id, source_message_id, parent_work_id,
              linked_entity_type, linked_entity_id, route_reason, confidence, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_text(work_id, "work_id"),
                _required_text(work_type, "work_type"),
                _required_text(title, "title")[:200],
                _required_text(goal, "goal")[:2_000],
                _required_text(status, "status"),
                _required_text(priority, "priority"),
                _required_text(risk_level, "risk_level"),
                owner_user_id,
                channel_id,
                source_message_id,
                parent_work_id,
                linked_entity_type,
                linked_entity_id,
                redact_text(route_reason, max_chars=1_000),
                float(confidence),
                _json(metadata or {}),
            ),
        )
        self.add_work_event(work_id, "created", actor=actor, payload={"type": work_type, "status": status})
        self.conn.commit()
        return self.get_work_item(work_id)

    def add_work_event(self, work_id: str, event_type: str, *, actor: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO work_events(work_id, event_type, actor, payload_json) VALUES (?, ?, ?, ?)",
            (_required_text(work_id, "work_id"), _required_text(event_type, "event_type"), _required_text(actor, "actor"), _json(payload)),
        )
        self._insert_agent_event(
            event_type=f"work.{event_type}",
            source="work_ledger",
            actor_id=actor,
            work_id=work_id,
            job_id=str(payload.get("job_id")) if payload.get("job_id") else None,
            payload=payload,
        )

    def add_work_note(self, work_id: str, *, actor: str, note: str) -> dict[str, Any]:
        cur = self.conn.execute(
            "INSERT INTO work_notes(work_id, actor, note_redacted) VALUES (?, ?, ?)",
            (_required_text(work_id, "work_id"), _required_text(actor, "actor"), redact_text(str(note), max_chars=2_000)),
        )
        self.add_work_event(work_id, "note_added", actor=actor, payload={"note_id": cur.lastrowid})
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM work_notes WHERE note_id=?", (cur.lastrowid,)).fetchone()
        return _row(row)

    def get_work_item(self, work_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM work_items WHERE work_id=?", (work_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown work item: {work_id}")
        return _row(row)

    def list_work_items(self, *, limit: int = 20, status: str | None = None, work_type: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        clauses = []
        params: list[Any] = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if work_type:
            clauses.append("type=?")
            params.append(work_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM work_items{where} ORDER BY updated_at DESC, created_at DESC LIMIT ?", (*params, limit)).fetchall()
        return [_row(row) for row in rows]

    def child_work_items(self, parent_work_id: str, *, work_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        clauses = ["parent_work_id=?"]
        params: list[Any] = [_required_text(parent_work_id, "parent_work_id")]
        if work_type:
            clauses.append("type=?")
            params.append(work_type)
        where = " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(f"SELECT * FROM work_items{where} ORDER BY updated_at DESC, created_at DESC LIMIT ?", (*params, limit)).fetchall()
        return [_row(row) for row in rows]

    def transition_work_item(self, work_id: str, next_status: str, *, actor: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.get_work_item(work_id)
        current_status = str(current.get("status") or "")
        if next_status not in allowed_work_next_statuses(current_status):
            raise ValueError(f"invalid work transition: {current_status} -> {next_status}")
        self.conn.execute("UPDATE work_items SET status=?, updated_at=CURRENT_TIMESTAMP WHERE work_id=?", (_required_text(next_status, "next_status"), work_id))
        self.add_work_event(work_id, next_status, actor=actor, payload=payload or {})
        self.conn.commit()
        return self.get_work_item(work_id)

    def work_events(self, work_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM work_events WHERE work_id=? ORDER BY event_id", (work_id,)).fetchall()
        return [_row(row) for row in rows]

    def work_notes(self, work_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM work_notes WHERE work_id=? ORDER BY note_id", (work_id,)).fetchall()
        return [_row(row) for row in rows]

    def create_work_job(
        self,
        *,
        job_id: str,
        work_id: str,
        queue_name: str,
        status: str = "created",
        priority: str = "medium",
        max_attempts: int = 3,
        payload: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO work_jobs(job_id, work_id, queue_name, status, priority, max_attempts, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_text(job_id, "job_id"),
                _required_text(work_id, "work_id"),
                _required_text(queue_name, "queue_name"),
                _required_text(status, "status"),
                _required_text(priority, "priority"),
                max(1, int(max_attempts)),
                _json(payload or {}),
            ),
        )
        self.add_work_event(work_id, "job_created", actor=actor, payload={"job_id": job_id, "queue_name": queue_name, "status": status})
        self.conn.commit()
        return self.get_work_job(job_id)

    def get_work_job(self, job_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM work_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown work job: {job_id}")
        return _row(row)

    def list_work_jobs(
        self,
        *,
        limit: int = 20,
        work_id: str | None = None,
        status: str | None = None,
        queue_name: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        clauses = []
        params: list[Any] = []
        if work_id:
            clauses.append("work_id=?")
            params.append(work_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        if queue_name:
            clauses.append("queue_name=?")
            params.append(queue_name)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM work_jobs{where} ORDER BY updated_at DESC, created_at DESC LIMIT ?", (*params, limit)).fetchall()
        return [_row(row) for row in rows]

    def find_active_work_job(self, work_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM work_jobs
            WHERE work_id=? AND status IN ('created', 'queued', 'running', 'retry_wait')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (work_id,),
        ).fetchone()
        return _row(row) if row is not None else None

    def mark_work_job_queued(self, job_id: str, *, redis_message_id: str | None, actor: str = "queue") -> dict[str, Any]:
        job = self.get_work_job(job_id)
        self.conn.execute(
            """
            UPDATE work_jobs
            SET status='queued', redis_message_id=?, enqueued_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
            WHERE job_id=?
            """,
            (redis_message_id, job_id),
        )
        self.conn.execute("UPDATE work_items SET queued_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP WHERE work_id=?", (job["work_id"],))
        self.add_work_event(str(job["work_id"]), "job_queued", actor=actor, payload={"job_id": job_id, "redis_message_id": redis_message_id})
        self.conn.commit()
        return self.get_work_job(job_id)

    def mark_work_job_enqueue_failed(self, job_id: str, *, error: str, actor: str = "queue") -> dict[str, Any]:
        job = self.get_work_job(job_id)
        self.conn.execute(
            """
            UPDATE work_jobs
            SET status='enqueue_failed', last_error=?, updated_at=CURRENT_TIMESTAMP
            WHERE job_id=?
            """,
            (redact_text(error, max_chars=1_000), job_id),
        )
        self.add_work_event(str(job["work_id"]), "job_enqueue_failed", actor=actor, payload={"job_id": job_id, "error": redact_text(error, max_chars=1_000)})
        self.conn.commit()
        return self.get_work_job(job_id)

    def mark_work_job_running(self, job_id: str, *, worker_id: str, actor: str = "worker") -> dict[str, Any]:
        job = self.get_work_job(job_id)
        self.conn.execute(
            """
            UPDATE work_jobs
            SET status='running', worker_id=?, attempts=attempts + 1, started_at=CURRENT_TIMESTAMP, heartbeat_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP
            WHERE job_id=?
            """,
            (_required_text(worker_id, "worker_id"), job_id),
        )
        self.add_work_event(str(job["work_id"]), "job_running", actor=actor, payload={"job_id": job_id, "worker_id": worker_id})
        self.conn.commit()
        return self.get_work_job(job_id)

    def mark_work_job_heartbeat(self, job_id: str, *, worker_id: str, message: str = "", actor: str = "worker") -> dict[str, Any]:
        job = self.get_work_job(job_id)
        self.conn.execute(
            """
            UPDATE work_jobs
            SET heartbeat_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP, worker_id=COALESCE(worker_id, ?)
            WHERE job_id=?
            """,
            (_required_text(worker_id, "worker_id"), job_id),
        )
        self.add_work_event(
            str(job["work_id"]),
            "job_heartbeat",
            actor=actor,
            payload={"job_id": job_id, "worker_id": worker_id, "message": redact_text(message, max_chars=300)},
        )
        self.conn.commit()
        return self.get_work_job(job_id)

    def mark_work_job_completed(self, job_id: str, *, actor: str = "worker", result: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.get_work_job(job_id)
        self.conn.execute(
            """
            UPDATE work_jobs
            SET status='completed', completed_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP, last_error=NULL
            WHERE job_id=?
            """,
            (job_id,),
        )
        self.add_work_event(str(job["work_id"]), "job_completed", actor=actor, payload={"job_id": job_id, "result": result or {}})
        self.conn.commit()
        return self.get_work_job(job_id)

    def mark_work_job_failed(self, job_id: str, *, error: str, actor: str = "worker", retryable: bool = True) -> dict[str, Any]:
        job = self.get_work_job(job_id)
        attempts = int(job.get("attempts") or 0)
        max_attempts = int(job.get("max_attempts") or 1)
        next_status = "retry_wait" if retryable and attempts < max_attempts else "dead_letter"
        self.conn.execute(
            """
            UPDATE work_jobs
            SET status=?, last_error=?, updated_at=CURRENT_TIMESTAMP
            WHERE job_id=?
            """,
            (next_status, redact_text(error, max_chars=1_000), job_id),
        )
        self.add_work_event(str(job["work_id"]), "job_failed", actor=actor, payload={"job_id": job_id, "status": next_status, "error": redact_text(error, max_chars=1_000)})
        self.conn.commit()
        return self.get_work_job(job_id)

    def stale_running_work_jobs(self, *, stale_after_seconds: int = 300, limit: int = 50) -> list[dict[str, Any]]:
        stale_after_seconds = max(1, int(stale_after_seconds))
        limit = max(1, min(int(limit), 200))
        rows = self.conn.execute(
            """
            SELECT *,
              CAST(strftime('%s','now') - strftime('%s', COALESCE(heartbeat_at, updated_at, started_at, created_at)) AS INTEGER) AS stale_for_seconds
            FROM work_jobs
            WHERE status='running'
              AND (strftime('%s','now') - strftime('%s', COALESCE(heartbeat_at, updated_at, started_at, created_at))) >= ?
            ORDER BY stale_for_seconds DESC
            LIMIT ?
            """,
            (stale_after_seconds, limit),
        ).fetchall()
        return [_row(row) for row in rows]

    def create_capability_gap(
        self,
        *,
        gap_id: str,
        user_id: str | None,
        channel_id: str | None,
        request_text: str,
        normalized_request: str,
        gap_type: str,
        requested_capability: str,
        matched_actions: list[dict[str, Any]],
        confidence: float,
        status: str = "detected",
        linked_task_id: str | None = None,
    ) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO capability_gaps(
              gap_id, user_id, channel_id, request_text, normalized_request, gap_type,
              requested_capability, matched_actions_json, confidence, status, linked_task_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_text(gap_id, "gap_id"),
                user_id,
                channel_id,
                redact_text(str(request_text), max_chars=2_000),
                _required_text(normalized_request, "normalized_request"),
                _required_text(gap_type, "gap_type"),
                _required_text(requested_capability, "requested_capability"),
                _json(matched_actions),
                float(confidence),
                _required_text(status, "status"),
                linked_task_id,
            ),
        )
        self.conn.commit()
        return self.get_capability_gap(gap_id)

    def create_capability_proposal(self, *, proposal_id: str, gap_id: str, proposal: dict[str, Any], status: str = "proposed", actor: str = "system", work_id: str | None = None) -> dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO capability_proposals(
              proposal_id, gap_id, status, action_id, capability_name, purpose, target,
              risk_level, side_effect, requires_approval, inputs_json, outputs_json,
              implementation_hint_json, test_plan_json, safety_notes_json, confidence,
              approval_required_for_implementation, activation_requires_tests, work_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _required_text(proposal_id, "proposal_id"),
                _required_text(gap_id, "gap_id"),
                _required_text(status, "status"),
                _required_text(str(proposal.get("action_id") or ""), "action_id"),
                _required_text(str(proposal.get("capability_name") or ""), "capability_name"),
                _required_text(str(proposal.get("purpose") or ""), "purpose"),
                _required_text(str(proposal.get("target") or ""), "target"),
                _required_text(str(proposal.get("risk_level") or ""), "risk_level"),
                int(bool(proposal.get("side_effect"))),
                int(bool(proposal.get("requires_approval"))),
                _json(proposal.get("inputs") or {}),
                _json(proposal.get("outputs") or {}),
                _json(proposal.get("implementation_hint") or {}),
                _json(proposal.get("test_plan") or []),
                _json(proposal.get("safety_notes") or []),
                float(proposal.get("confidence") or 0.0),
                int(bool(proposal.get("approval_required_for_implementation"))),
                int(bool(proposal.get("activation_requires_tests"))),
                work_id,
            ),
        )
        self.add_proposal_event(proposal_id, "created", actor=actor, payload={"gap_id": gap_id, "status": status})
        self.conn.commit()
        return self.get_capability_proposal(proposal_id)

    def add_proposal_event(self, proposal_id: str, event_type: str, *, actor: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO proposal_events(proposal_id, event_type, actor, payload_json) VALUES (?, ?, ?, ?)",
            (_required_text(proposal_id, "proposal_id"), _required_text(event_type, "event_type"), _required_text(actor, "actor"), _json(payload)),
        )
        self._insert_agent_event(
            event_type=f"proposal.{event_type}",
            source="proposal_ledger",
            actor_id=actor,
            proposal_id=proposal_id,
            work_id=str(payload.get("work_id")) if payload.get("work_id") else None,
            payload=payload,
        )

    def get_capability_gap(self, gap_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM capability_gaps WHERE gap_id=?", (gap_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown capability gap: {gap_id}")
        return _row(row)

    def get_capability_proposal(self, proposal_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM capability_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown capability proposal: {proposal_id}")
        return _row(row)

    def find_capability_proposal_by_work_id(self, work_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM capability_proposals WHERE work_id=? ORDER BY created_at DESC LIMIT 1",
            (_required_text(work_id, "work_id"),),
        ).fetchone()
        return _row(row) if row is not None else None

    def list_capability_gaps(self, *, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        if status:
            rows = self.conn.execute("SELECT * FROM capability_gaps WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM capability_gaps ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(row) for row in rows]

    def list_capability_proposals(self, *, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        if status:
            rows = self.conn.execute("SELECT * FROM capability_proposals WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM capability_proposals ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_row(row) for row in rows]

    def find_open_capability_proposal(self, *, action_id: str, target: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM capability_proposals
            WHERE action_id=? AND target=? AND status IN ('proposed', 'approved_for_dev', 'deferred')
            ORDER BY created_at DESC LIMIT 1
            """,
            (action_id, target),
        ).fetchone()
        return _row(row) if row is not None else None

    def transition_capability_proposal(self, proposal_id: str, next_status: str, *, actor: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.get_capability_proposal(proposal_id)
        current_status = str(current.get("status") or "")
        if next_status not in allowed_proposal_next_statuses(current_status):
            raise ValueError(f"invalid proposal transition: {current_status} -> {next_status}")
        self.conn.execute(
            "UPDATE capability_proposals SET status=?, updated_at=CURRENT_TIMESTAMP WHERE proposal_id=?",
            (_required_text(next_status, "next_status"), proposal_id),
        )
        self.add_proposal_event(proposal_id, next_status, actor=actor, payload=payload or {})
        self.conn.commit()
        return self.get_capability_proposal(proposal_id)

    def proposal_events(self, proposal_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM proposal_events WHERE proposal_id=? ORDER BY event_id", (proposal_id,)).fetchall()
        return [_row(row) for row in rows]
