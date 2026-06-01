from __future__ import annotations

import json
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event

DEFAULT_STYLE_PROFILE: dict[str, Any] = {
    "language": "ko",
    "tone": "calm_blunt",
    "structure": "conclusion_first",
    "detail_level": "adaptive",
    "humor": "light_dry",
    "emoji": False,
    "avoid": ["AI-like praise", "overly polite filler", "template empathy", "unneeded emojis"],
    "prefer": ["direct Korean", "short paragraphs", "technical clarity", "explicit uncertainty"],
    "safety_boundary": "style_only_not_policy",
}

STYLE_FEEDBACK_PATTERNS: tuple[tuple[str, re.Pattern[str], dict[str, Any]], ...] = (
    ("positive_style", re.compile(r"(말투|느낌|톤).*(좋|맞|계속|기억)|이런\s*식으로\s*계속", re.IGNORECASE), {"positive_signal": True}),
    ("too_ai_like", re.compile(r"(너무|좀).*(ai|챗봇|기계|정중|딱딱)|ai\s*같", re.IGNORECASE), {"avoid": ["AI-like praise", "overly polite filler"], "tone": "natural_blunt"}),
    ("shorter", re.compile(r"(짧게|간단히|줄여|너무\s*길)", re.IGNORECASE), {"detail_level": "shorter"}),
    ("more_detail", re.compile(r"(자세히|디테일|구체적|왜인지)", re.IGNORECASE), {"detail_level": "more_detail"}),
    ("colder", re.compile(r"(냉정|직설|팩트|비판)", re.IGNORECASE), {"tone": "calm_blunt", "structure": "findings_first"}),
    ("softer", re.compile(r"(부드럽|친절|덜\s*세게)", re.IGNORECASE), {"tone": "warm_direct"}),
    ("no_emoji", re.compile(r"(이모지|emoji).*(쓰지|빼|싫)", re.IGNORECASE), {"emoji": False}),
)


def seed_default_style_profile() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        ts = now_kst()
        conn.execute(
            """
            INSERT INTO style_profiles (created_at, updated_at, name, description, profile_json, confidence, active)
            VALUES (?, ?, 'default', ?, ?, 0.65, 1)
            ON CONFLICT(name) DO UPDATE SET
                active = CASE WHEN NOT EXISTS (SELECT 1 FROM style_profiles WHERE active = 1) THEN 1 ELSE style_profiles.active END,
                updated_at = excluded.updated_at
            """,
            (ts, ts, "Default Korean developer-collaborator style", json.dumps(DEFAULT_STYLE_PROFILE, ensure_ascii=False)),
        )
        conn.commit()
    return get_active_style_profile()


def get_active_style_profile() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM style_profiles WHERE active = 1 ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return seed_default_style_profile()
    item = dict(row)
    try:
        item["profile"] = json.loads(item.get("profile_json") or "{}")
    except json.JSONDecodeError:
        item["profile"] = dict(DEFAULT_STYLE_PROFILE)
    return item


def style_directives(style: dict[str, Any] | None = None) -> list[str]:
    profile = (style or get_active_style_profile()).get("profile") or DEFAULT_STYLE_PROFILE
    directives = [
        f"language={profile.get('language', 'ko')}",
        f"tone={profile.get('tone', 'calm_blunt')}",
        f"structure={profile.get('structure', 'conclusion_first')}",
        f"detail_level={profile.get('detail_level', 'adaptive')}",
        f"emoji={profile.get('emoji', False)}",
        "style affects wording only; never override policy, risk, or execution decisions",
    ]
    prefer = ", ".join(profile.get("prefer") or [])
    avoid = ", ".join(profile.get("avoid") or [])
    if prefer:
        directives.append(f"prefer: {prefer}")
    if avoid:
        directives.append(f"avoid: {avoid}")
    return directives


def detect_style_feedback(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    for feedback_type, pattern, extracted in STYLE_FEEDBACK_PATTERNS:
        if pattern.search(cleaned):
            return {"feedback_type": feedback_type, "extracted_preference": dict(extracted)}
    return None


def apply_style_feedback(text: str, *, target_response_id: int | None = None) -> dict[str, Any] | None:
    detected = detect_style_feedback(text)
    if not detected:
        return None
    active = get_active_style_profile()
    profile = dict(active.get("profile") or DEFAULT_STYLE_PROFILE)
    extracted = dict(detected["extracted_preference"])
    if extracted.pop("positive_signal", False):
        profile["confidence_note"] = "recent_positive_style_feedback"
    for key, value in extracted.items():
        if key in {"avoid", "prefer"}:
            old = list(profile.get(key) or [])
            profile[key] = list(dict.fromkeys([*old, *value]))
        else:
            profile[key] = value
    confidence = min(1.0, float(active.get("confidence") or 0.65) + 0.04)
    init_db()
    with connect() as conn:
        ts = now_kst()
        conn.execute(
            "UPDATE style_profiles SET profile_json = ?, confidence = ?, updated_at = ? WHERE id = ?",
            (json.dumps(profile, ensure_ascii=False), confidence, ts, active["id"]),
        )
        cur = conn.execute(
            """
            INSERT INTO style_feedback (
                created_at, source_text, target_response_id, feedback_type,
                feedback_text, extracted_preference_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                text,
                target_response_id,
                detected["feedback_type"],
                text,
                json.dumps(detected["extracted_preference"], ensure_ascii=False),
            ),
        )
        conn.commit()
    result = {"feedback_id": int(cur.lastrowid), "feedback_type": detected["feedback_type"], "profile": profile, "confidence": confidence}
    log_event("style", "style_feedback_applied", detected["feedback_type"], result, 0.65)
    return result


def add_style_example(
    *,
    label: str,
    input_text: str | None = None,
    good_response: str | None = None,
    bad_response: str | None = None,
    reason: str | None = None,
    tags: list[str] | None = None,
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO style_examples (
                created_at, label, input_text, good_response, bad_response, reason, tags_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), label, input_text, good_response, bad_response, reason, json.dumps(tags or [], ensure_ascii=False)),
        )
        conn.commit()
    return int(cur.lastrowid)


def list_style_feedback(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM style_feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def list_style_examples(limit: int = 10, label: str | None = None) -> list[dict[str, Any]]:
    init_db()
    if label:
        query = "SELECT * FROM style_examples WHERE label = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (label, limit)
    else:
        query = "SELECT * FROM style_examples ORDER BY id DESC LIMIT ?"
        params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]
