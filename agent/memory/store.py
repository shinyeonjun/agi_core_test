from __future__ import annotations

import json
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db


def add_memory(
    title: str,
    content: str,
    memory_type: str = "fact",
    tags: list[str] | None = None,
    importance: float = 0.5,
    confidence: float = 0.7,
    source_event_id: int | None = None,
) -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO memories (
                created_at, updated_at, memory_type, title, content, tags_json,
                importance, confidence, source_event_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                ts,
                memory_type,
                title,
                content,
                json.dumps(tags or [], ensure_ascii=False),
                importance,
                confidence,
                source_event_id,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def _query_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"\w+", query.lower()):
        if len(term) >= 2 and term not in terms:
            terms.append(term)
    return terms[:8]


def search_memories(query: str, limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    terms = _query_terms(query)
    if not terms:
        where = "archived = 0"
        params: list[Any] = []
    else:
        parts = []
        params = []
        for term in terms:
            like = f"%{term}%"
            parts.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(tags_json) LIKE ?)" )
            params.extend([like, like, like])
        where = "archived = 0 AND (" + " OR ".join(parts) + ")"
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {where}
            ORDER BY importance DESC, confidence DESC, updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        ids = [row["id"] for row in rows]
        if ids:
            conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now_kst(), memory_id) for memory_id in ids],
            )
            conn.commit()
    return [dict(row) for row in rows]


def list_memories(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM memories WHERE archived = 0 ORDER BY importance DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
