from __future__ import annotations

from typing import Any

from agent.core.autonomy import get_autonomy_state
from agent.core.database import connect, init_db


def _count(conn, query: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(query, params).fetchone()
    return int(row["count"] if row else 0)


def _last_eval(conn) -> tuple[str | None, float | None]:
    row = conn.execute("SELECT result, score FROM eval_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None, None
    return str(row["result"]), float(row["score"]) if row["score"] is not None else None


def _renderer_rates(conn) -> tuple[float, float]:
    total = _count(conn, "SELECT COUNT(*) AS count FROM renderer_runs")
    if total == 0:
        return 0.0, 0.0
    success = _count(conn, "SELECT COUNT(*) AS count FROM renderer_runs WHERE success = 1")
    fallback = _count(conn, "SELECT COUNT(*) AS count FROM renderer_runs WHERE success = 0 OR validation_result_json LIKE '%\"fallback\": true%'")
    return round(success / total, 4), round(fallback / total, 4)


def _action_rates(conn) -> tuple[float, int, int]:
    total = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE created_at >= datetime('now', '-1 day')")
    if total == 0:
        return 0.0, 0, 0
    success = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'completed' AND created_at >= datetime('now', '-1 day')")
    timeout = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'timeout' AND created_at >= datetime('now', '-1 day')")
    blocked = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'blocked' AND created_at >= datetime('now', '-1 day')")
    return round(success / total, 4), timeout, blocked


def _last_action(conn) -> tuple[str | None, str | None]:
    row = conn.execute("SELECT created_at, status FROM action_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None, None
    return str(row["created_at"]), str(row["status"])


def collect_metrics() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        last_eval_result, last_eval_score = _last_eval(conn)
        renderer_success_rate, renderer_fallback_rate = _renderer_rates(conn)
        action_success_rate, action_timeout_count, action_blocked_count = _action_rates(conn)
        last_action_at, last_action_status = _last_action(conn)
        autonomy = get_autonomy_state()
        return {
            "events_count": _count(conn, "SELECT COUNT(*) AS count FROM events"),
            "memories_count": _count(conn, "SELECT COUNT(*) AS count FROM memories WHERE archived = 0"),
            "open_goals_count": _count(conn, "SELECT COUNT(*) AS count FROM goals WHERE status IN ('proposed', 'active', 'waiting_approval', 'blocked')"),
            "pending_approvals_count": _count(conn, "SELECT COUNT(*) AS count FROM approvals WHERE status = 'pending'"),
            "reflections_count": _count(conn, "SELECT COUNT(*) AS count FROM reflections"),
            "skills_count": _count(conn, "SELECT COUNT(*) AS count FROM skills WHERE archived = 0"),
            "last_eval_result": last_eval_result,
            "last_eval_score": last_eval_score,
            "renderer_success_rate": renderer_success_rate,
            "renderer_fallback_rate": renderer_fallback_rate,
            "policy_critical_count_24h": _count(conn, "SELECT COUNT(*) AS count FROM policy_decisions WHERE risk_level = 'critical' AND created_at >= datetime('now', '-1 day')"),
            "discord_messages_24h": _count(conn, "SELECT COUNT(*) AS count FROM discord_events WHERE created_at >= datetime('now', '-1 day')"),
            "tick_count_24h": _count(conn, "SELECT COUNT(*) AS count FROM events WHERE event_type = 'idle_tick' AND ts >= datetime('now', '-1 day')"),
            "workspace_artifact_count": _count(conn, "SELECT COUNT(*) AS count FROM workspace_artifacts"),
            "action_runs_count": _count(conn, "SELECT COUNT(*) AS count FROM action_runs"),
            "action_proposals_count": _count(conn, "SELECT COUNT(*) AS count FROM action_proposals"),
            "action_success_rate_24h": action_success_rate,
            "action_timeout_count_24h": action_timeout_count,
            "action_blocked_count_24h": action_blocked_count,
            "full_device_lab_enabled": bool(autonomy.get("full_device_lab_enabled")),
            "current_autonomy_profile": autonomy.get("autonomy_profile"),
            "last_action_at": last_action_at,
            "last_action_status": last_action_status,
            "catastrophic_actions_count": _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE command_json LIKE '%rm%' AND command_json LIKE '%/%'"),
        }
