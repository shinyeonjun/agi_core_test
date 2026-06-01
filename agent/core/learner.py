from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goals import create_goal
from agent.language.engine import interpret_user_message
from agent.language.fallback_rule import detect_feedback_rule
from agent.memory.store import add_memory


def detect_feedback(user_text: str, *, interpretation: dict[str, Any] | None = None) -> str:
    interpretation = interpretation or interpret_user_message(user_text, {"purpose": "feedback_detection"}, log=False)
    if interpretation.get("intent") in {"feedback", "style_feedback"} and interpretation.get("sentiment") in {"positive", "negative"}:
        return str(interpretation["sentiment"])
    return detect_feedback_rule(user_text)


def create_reflection(summary: str, source_event_id: int | None = None, goal_id: int | None = None, learned: dict[str, Any] | None = None, followup_goal: dict[str, Any] | None = None, confidence: float = 0.7) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO reflections (created_at, source_event_id, goal_id, summary, learned_json, followup_goal_json, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), source_event_id, goal_id, summary, json.dumps(learned or {}, ensure_ascii=False), json.dumps(followup_goal or {}, ensure_ascii=False), confidence),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_reflections(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM reflections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def upsert_skill(name: str, trigger_description: str, procedure: list[str], tags: list[str] | None = None) -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO skills (created_at, updated_at, name, trigger_description, procedure_json, tags_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                updated_at = excluded.updated_at,
                trigger_description = excluded.trigger_description,
                procedure_json = excluded.procedure_json,
                tags_json = excluded.tags_json,
                archived = 0
            """,
            (ts, ts, name, trigger_description, json.dumps(procedure, ensure_ascii=False), json.dumps(tags or [], ensure_ascii=False)),
        )
        row = conn.execute("SELECT id FROM skills WHERE name = ?", (name,)).fetchone()
        conn.commit()
        return int(row["id"])


def list_skills(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM skills WHERE archived = 0 ORDER BY confidence DESC, id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def retrieve_skills(intent: str, tags: list[str] | None = None, limit: int = 3) -> list[dict[str, Any]]:
    init_db()
    terms = {term for term in intent.lower().split() if len(term) >= 2}
    wanted_tags = {tag.lower() for tag in (tags or [])}
    with connect() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM skills WHERE archived = 0").fetchall()]
    for row in rows:
        trigger = f"{row.get('name', '')} {row.get('trigger_description', '')}".lower()
        row_tags = set(json.loads(row.get("tags_json") or "[]"))
        tag_match = len(wanted_tags & {str(tag).lower() for tag in row_tags}) / max(1, len(wanted_tags))
        text_match = sum(1 for term in terms if term in trigger) / max(1, len(terms))
        row["score"] = round(float(row.get("confidence") or 0.5) * 0.5 + tag_match * 0.3 + text_match * 0.2, 4)
    selected = sorted(rows, key=lambda item: item.get("score", 0.0), reverse=True)[:limit]
    if selected:
        with connect() as conn:
            conn.executemany("UPDATE skills SET last_used_at = ? WHERE id = ?", [(now_kst(), item["id"]) for item in selected])
            conn.commit()
    return selected


def update_skill_feedback(skill_name: str, feedback: str) -> dict[str, Any]:
    upsert_skill(skill_name, "Core talk pipeline quality improvement", ["retrieve relevant memory", "build Decision Object", "return validated renderer output"], ["talk", "core"])
    column = "success_count" if feedback == "positive" else "failure_count" if feedback == "negative" else None
    with connect() as conn:
        if column:
            conn.execute(f"UPDATE skills SET {column} = {column} + 1, updated_at = ? WHERE name = ?", (now_kst(), skill_name))
        conn.execute("UPDATE skills SET confidence = CAST(success_count + 1 AS REAL) / CAST(success_count + failure_count + 2 AS REAL) WHERE name = ?", (skill_name,))
        row = conn.execute("SELECT * FROM skills WHERE name = ?", (skill_name,)).fetchone()
        conn.commit()
    return dict(row)


def update_preference_ema(key: str, observed_value: float, alpha: float = 0.1) -> None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
        old = float(row["value"]) if row else 0.5
        new_value = old * (1 - alpha) + observed_value * alpha
        conn.execute(
            """
            INSERT INTO preferences (key, value, confidence, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, confidence = excluded.confidence, updated_at = excluded.updated_at
            """,
            (key, f"{new_value:.4f}", min(1.0, alpha + 0.6), now_kst()),
        )
        conn.commit()


def update_after_turn(user_text: str, source_event_id: int | None, goal_id: int | None, decision: dict[str, Any]) -> dict[str, Any]:
    feedback = detect_feedback(user_text, interpretation=decision.get("language_interpretation"))
    skill = update_skill_feedback("core_talk_pipeline", feedback)
    learned = {"feedback": feedback, "skill": skill.get("name"), "confidence": skill.get("confidence")}
    if feedback == "positive":
        update_preference_ema("talk_style_alignment", 1.0)
    elif feedback == "negative":
        update_preference_ema("talk_style_alignment", 0.0)
        add_memory("negative feedback", user_text, memory_type="failure", tags=["feedback", "talk"], importance=0.75, confidence=0.8, source_event_id=source_event_id)
        create_goal("Apply user negative feedback", user_text, goal_type="improvement", status="active", priority=0.7, risk_level="low", metadata={"feedback": feedback})
    reflection_id = create_reflection("Recorded talk feedback, selected goal, and retrieved context.", source_event_id, goal_id, learned, confidence=0.72)
    log_event("learner", "reflection_created", str(reflection_id), learned, 0.5)
    return {"feedback": feedback, "skill": skill, "reflection_id": reflection_id}
