from __future__ import annotations

from typing import Any

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


def collect_metrics() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        last_eval_result, last_eval_score = _last_eval(conn)
        renderer_success_rate, renderer_fallback_rate = _renderer_rates(conn)
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
        }
