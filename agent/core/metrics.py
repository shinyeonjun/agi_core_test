from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from agent.core.autonomy import get_autonomy_state
from agent.core.database import connect, init_db
from agent.core.goals import is_noise_goal_record
from agent.config.defaults import KST
from agent.memory.sparse_vector import VECTOR_TYPE


def _count(conn, query: str, params: tuple[Any, ...] = ()) -> int:
    row = conn.execute(query, params).fetchone()
    return int(row["count"] if row else 0)


def _since_24h() -> str:
    return (datetime.now(KST) - timedelta(days=1)).isoformat(timespec="seconds")


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


def _action_rates(conn) -> dict[str, Any]:
    since = _since_24h()
    total = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE created_at >= ?", (since,))
    if total == 0:
        return {
            "success_rate": 0.0,
            "execution_success_rate": 0.0,
            "planned_block_rate": 0.0,
            "timeout_count": 0,
            "blocked_count": 0,
            "executed_count": 0,
            "successful_count": 0,
        }
    success = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'completed' AND created_at >= ?", (since,))
    timeout = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'timeout' AND created_at >= ?", (since,))
    blocked = _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE status = 'blocked' AND created_at >= ?", (since,))
    planned_blocks = _count(
        conn,
        """
        SELECT COUNT(*) AS count FROM action_runs
        WHERE status = 'blocked'
          AND result_summary IN ('profile_not_full_device_lab', 'approval_required')
          AND created_at >= ?
        """,
        (since,),
    )
    executed_total = max(0, total - planned_blocks)
    return {
        "success_rate": round(success / total, 4),
        "execution_success_rate": round(success / executed_total, 4) if executed_total else 0.0,
        "planned_block_rate": round(planned_blocks / total, 4),
        "timeout_count": timeout,
        "blocked_count": blocked,
        "executed_count": executed_total,
        "successful_count": success,
    }


def _last_action(conn) -> tuple[str | None, str | None]:
    row = conn.execute("SELECT created_at, status FROM action_runs ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return None, None
    return str(row["created_at"]), str(row["status"])


def collect_metrics() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        since = _since_24h()
        last_eval_result, last_eval_score = _last_eval(conn)
        renderer_success_rate, renderer_fallback_rate = _renderer_rates(conn)
        action_rates = _action_rates(conn)
        last_action_at, last_action_status = _last_action(conn)
        autonomy = get_autonomy_state()
        open_goal_rows = conn.execute("SELECT * FROM goals WHERE status IN ('proposed', 'active', 'waiting_approval', 'blocked')").fetchall()
        noise_open_goals = sum(1 for row in open_goal_rows if is_noise_goal_record(dict(row)))
        meaningful_open_goals = len(open_goal_rows) - noise_open_goals
        return {
            "events_count": _count(conn, "SELECT COUNT(*) AS count FROM events"),
            "memories_count": _count(conn, "SELECT COUNT(*) AS count FROM memories WHERE archived = 0"),
            "memory_vector_count": _count(
                conn,
                """
                SELECT COUNT(*) AS count
                FROM memory_vectors v
                JOIN memories m ON m.id = v.memory_id
                WHERE m.archived = 0 AND v.vector_type = ?
                """,
                (VECTOR_TYPE,),
            ),
            "memory_vector_coverage": round(
                _count(
                    conn,
                    """
                    SELECT COUNT(*) AS count
                    FROM memory_vectors v
                    JOIN memories m ON m.id = v.memory_id
                    WHERE m.archived = 0 AND v.vector_type = ?
                    """,
                    (VECTOR_TYPE,),
                ) / max(1, _count(conn, "SELECT COUNT(*) AS count FROM memories WHERE archived = 0")),
                4,
            ),
            "open_goals_count": _count(conn, "SELECT COUNT(*) AS count FROM goals WHERE status IN ('proposed', 'active', 'waiting_approval', 'blocked')"),
            "meaningful_open_goals_count": meaningful_open_goals,
            "noise_open_goals_count": noise_open_goals,
            "pending_approvals_count": _count(conn, "SELECT COUNT(*) AS count FROM approvals WHERE status = 'pending'"),
            "reflections_count": _count(conn, "SELECT COUNT(*) AS count FROM reflections"),
            "skills_count": _count(conn, "SELECT COUNT(*) AS count FROM skills WHERE archived = 0"),
            "last_eval_result": last_eval_result,
            "last_eval_score": last_eval_score,
            "renderer_success_rate": renderer_success_rate,
            "renderer_fallback_rate": renderer_fallback_rate,
            "policy_critical_count_24h": _count(conn, "SELECT COUNT(*) AS count FROM policy_decisions WHERE risk_level = 'critical' AND created_at >= ?", (since,)),
            "discord_messages_24h": _count(conn, "SELECT COUNT(*) AS count FROM discord_events WHERE created_at >= ?", (since,)),
            "tick_count_24h": _count(conn, "SELECT COUNT(*) AS count FROM events WHERE event_type = 'idle_tick' AND ts >= ?", (since,)),
            "workspace_artifact_count": _count(conn, "SELECT COUNT(*) AS count FROM workspace_artifacts"),
            "action_runs_count": _count(conn, "SELECT COUNT(*) AS count FROM action_runs"),
            "action_proposals_count": _count(conn, "SELECT COUNT(*) AS count FROM action_proposals"),
            "repeated_action_suppressed_count": _count(conn, "SELECT COUNT(*) AS count FROM action_proposals WHERE reason = 'duplicate_recent_action'"),
            "self_map_count": _count(conn, "SELECT COUNT(*) AS count FROM self_maps"),
            "queued_user_tasks_count": _count(conn, "SELECT COUNT(*) AS count FROM task_queue WHERE queue_type = 'user' AND status = 'queued'"),
            "queued_autonomous_tasks_count": _count(conn, "SELECT COUNT(*) AS count FROM task_queue WHERE queue_type = 'autonomous' AND status = 'queued'"),
            "action_success_rate_24h": action_rates["success_rate"],
            "action_execution_success_rate_24h": action_rates["execution_success_rate"],
            "action_planned_block_rate_24h": action_rates["planned_block_rate"],
            "action_executed_count_24h": action_rates["executed_count"],
            "action_successful_count_24h": action_rates["successful_count"],
            "action_timeout_count_24h": action_rates["timeout_count"],
            "action_blocked_count_24h": action_rates["blocked_count"],
            "full_device_lab_enabled": bool(autonomy.get("full_device_lab_enabled")),
            "current_autonomy_profile": autonomy.get("autonomy_profile"),
            "last_action_at": last_action_at,
            "last_action_status": last_action_status,
            "catastrophic_actions_count": _count(conn, "SELECT COUNT(*) AS count FROM action_runs WHERE command_json LIKE '%rm%' AND command_json LIKE '%/%'"),
        }
