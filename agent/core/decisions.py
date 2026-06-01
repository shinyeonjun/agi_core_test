from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db


def record_decision(decision: dict[str, Any]) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO decisions (
                ts, kind, user_input, selected_goal_id, decision_json,
                confidence, risk_level, renderer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst(), decision.get("kind", "unknown"), decision.get("user_input") or decision.get("user_message"),
                decision.get("selected_goal_id"), json.dumps(decision, ensure_ascii=False),
                decision.get("confidence", decision.get("decision_confidence")), decision.get("risk_level", "low"),
                decision.get("renderer", "fallback"),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_decisions(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
