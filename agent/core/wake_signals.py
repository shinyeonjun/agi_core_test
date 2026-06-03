from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event


PENDING_STATUS = "pending"


def _json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("payload_json")
    try:
        row["payload"] = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        row["payload"] = {"decode_error": True}
    return row


def _clamp_priority(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _result_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    result = result or {}
    nested = result.get("result") if isinstance(result.get("result"), dict) else {}
    return {
        "action": result.get("action"),
        "status": result.get("status") or nested.get("status"),
        "reason": result.get("reason") or nested.get("reason"),
        "task_id": result.get("task_id") or nested.get("task_id"),
        "goal_id": result.get("goal_id") or nested.get("goal_id"),
        "executed": result.get("executed") if result.get("executed") is not None else nested.get("executed"),
    }


def _dedupe_key(signal_type: str, source: str, payload: dict[str, Any] | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    payload_view = payload or {}
    stable = {
        "signal_type": signal_type,
        "source": source,
        "task_id": payload_view.get("task_id"),
        "goal_id": payload_view.get("goal_id"),
        "approval_id": payload_view.get("approval_id"),
        "action_id": payload_view.get("action_id"),
        "message_id": payload_view.get("message_id"),
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _now_expired_clause(now: str) -> tuple[str, tuple[str, ...]]:
    return "(expires_at IS NULL OR expires_at > ?)", (now,)


def emit_wake_signal(
    signal_type: str,
    source: str,
    *,
    priority: float = 0.5,
    payload: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    not_before: str | None = None,
    expires_at: str | None = None,
) -> int:
    init_db()
    ts = now_kst()
    priority = _clamp_priority(priority)
    key = _dedupe_key(signal_type, source, payload, dedupe_key)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO wake_signals (
                created_at, updated_at, signal_type, source, priority, status,
                payload_json, dedupe_key, occurrence_count, not_before, expires_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, 1, ?, ?)
            ON CONFLICT(dedupe_key) WHERE status = 'pending' AND dedupe_key IS NOT NULL
            DO UPDATE SET
                updated_at = excluded.updated_at,
                priority = MAX(wake_signals.priority, excluded.priority),
                payload_json = excluded.payload_json,
                occurrence_count = wake_signals.occurrence_count + 1,
                not_before = COALESCE(excluded.not_before, wake_signals.not_before),
                expires_at = COALESCE(excluded.expires_at, wake_signals.expires_at)
            """,
            (ts, ts, signal_type, source, priority, _json(payload), key, not_before, expires_at),
        )
        if cur.lastrowid:
            signal_id = int(cur.lastrowid)
        else:
            row = conn.execute("SELECT id FROM wake_signals WHERE dedupe_key = ? AND status = 'pending'", (key,)).fetchone()
            signal_id = int(row["id"]) if row else 0
        conn.commit()
    log_event("reactor", "wake_signal_emitted", signal_type, {"signal_id": signal_id, "source": source, "priority": priority}, 0.55)
    return signal_id


def list_wake_signals(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM wake_signals"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY status = 'pending' DESC, priority DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_decode(dict(row)) for row in rows]


def claim_wake_signal() -> dict[str, Any] | None:
    init_db()
    ts = now_kst()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        expired = conn.execute(
            """
            UPDATE wake_signals
            SET status = 'expired', updated_at = ?
            WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (ts, ts),
        ).rowcount
        row = conn.execute(
            """
            SELECT * FROM wake_signals
            WHERE status = 'pending'
              AND (not_before IS NULL OR not_before <= ?)
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY priority DESC, occurrence_count DESC, id ASC
            LIMIT 1
            """,
            (ts, ts),
        ).fetchone()
        if not row:
            conn.commit()
            if expired:
                log_event("reactor", "wake_signals_expired", str(expired), {"count": expired}, 0.45)
            return None
        signal = dict(row)
        conn.execute(
            """
            UPDATE wake_signals
            SET status = 'claimed', updated_at = ?, claimed_at = ?
            WHERE id = ? AND status = 'pending'
            """,
            (ts, ts, signal["id"]),
        )
        conn.commit()
    return _decode(signal)


def complete_wake_signal(signal_id: int, *, status: str = "done", result: dict[str, Any] | None = None) -> bool:
    if status not in {"done", "skipped", "failed"}:
        raise ValueError(f"invalid wake signal completion status: {status}")
    init_db()
    ts = now_kst()
    payload = {"result": _result_summary(result)}
    with connect() as conn:
        cur = conn.execute(
            """
            UPDATE wake_signals
            SET status = ?, updated_at = ?, completed_at = ?, payload_json = ?
            WHERE id = ?
            """,
            (status, ts, ts, _json(payload), signal_id),
        )
        conn.commit()
        ok = cur.rowcount > 0
    if ok:
        log_event("reactor", "wake_signal_completed", str(signal_id), {"signal_id": signal_id, "status": status}, 0.5)
    return ok


def wake_signal_counts() -> dict[str, int]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM wake_signals GROUP BY status").fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def prune_wake_signals(*, done_older_than_hours: int = 168, expired_older_than_hours: int = 24) -> dict[str, int]:
    init_db()
    ts = now_kst()
    done_cutoff = (datetime.now(KST) - timedelta(hours=max(1, int(done_older_than_hours)))).isoformat(timespec="seconds")
    expired_cutoff = (datetime.now(KST) - timedelta(hours=max(1, int(expired_older_than_hours)))).isoformat(timespec="seconds")
    with connect() as conn:
        done = conn.execute(
            """
            DELETE FROM wake_signals
            WHERE status IN ('done', 'skipped', 'failed')
              AND updated_at < ?
            """,
            (done_cutoff,),
        ).rowcount
        expired = conn.execute(
            """
            DELETE FROM wake_signals
            WHERE status = 'expired'
              AND updated_at < ?
            """,
            (expired_cutoff,),
        ).rowcount
        conn.commit()
    result = {"done_pruned": int(done), "expired_pruned": int(expired)}
    if done or expired:
        log_event("reactor", "wake_signals_pruned", str(result), {"created_at": ts, **result}, 0.45)
    return result


def latest_signal_age_seconds() -> float | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT created_at FROM wake_signals ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None
    try:
        created = datetime.fromisoformat(str(row["created_at"]))
    except ValueError:
        return None
    return max(0.0, (datetime.now(created.tzinfo or KST) - created).total_seconds())
