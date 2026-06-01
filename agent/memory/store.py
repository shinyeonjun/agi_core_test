from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

from agent.config.defaults import KST, now_kst
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
            (ts, ts, memory_type, title, content, json.dumps(tags or [], ensure_ascii=False), importance, confidence, source_event_id),
        )
        conn.commit()
        return int(cur.lastrowid)


def _query_terms(query: str) -> list[str]:
    terms = []
    for term in re.findall(r"[0-9A-Za-z_\uac00-\ud7a3]+", query.lower()):
        if len(term) >= 2 and term not in terms:
            terms.append(term)
    return terms[:8]


def _parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return [str(item) for item in value]
    except json.JSONDecodeError:
        return []


def _recency_score(row: dict[str, Any]) -> float:
    raw = row.get("last_used_at") or row.get("updated_at") or row.get("created_at")
    try:
        used = datetime.fromisoformat(str(raw))
    except ValueError:
        return 0.1
    days = max(0.0, (datetime.now(KST) - used).total_seconds() / 86400)
    if days <= 1:
        return 1.0
    if days <= 7:
        return 0.7
    if days <= 30:
        return 0.4
    return 0.1


def _score_memory(row: dict[str, Any], query: str, terms: list[str], fts_rank: float | None = None) -> float:
    haystack = f"{row.get('title', '')} {row.get('content', '')}".lower()
    tags = _parse_tags(row.get("tags_json"))
    keyword_match = sum(1 for term in terms if term in haystack) / max(1, len(terms))
    tag_match = sum(1 for term in terms if any(term in tag.lower() for tag in tags)) / max(1, len(terms))
    use_count_bonus = min(1.0, math.log1p(int(row.get("use_count") or 0)) / 5)
    project_relevance = 1.0 if any(tag in {"core", "project_context", "orangepi", "digital_agi"} for tag in tags) else 0.2
    rank_bonus = 0.2 if fts_rank is not None else 0.0
    score = (
        keyword_match * 0.30 + tag_match * 0.20 + float(row.get("importance") or 0.0) * 0.20
        + _recency_score(row) * 0.10 + use_count_bonus * 0.10 + project_relevance * 0.10 + rank_bonus
    )
    row["score"] = round(score, 4)
    return float(row["score"])


def _fts_query(terms: list[str]) -> str:
    return " OR ".join(term.replace('"', ' ') for term in terms)


def _search_fts(query: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
    if not terms:
        return []
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT m.*, bm25(memories_fts) AS bm25_score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.id
            WHERE memories_fts MATCH ? AND m.archived = 0
            ORDER BY bm25_score ASC, m.importance DESC, m.updated_at DESC
            LIMIT ?
            """,
            (_fts_query(terms), limit),
        ).fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        _score_memory(row, query, terms, float(row.get("bm25_score") or 0.0))
    return sorted(result, key=lambda row: row.get("score", 0.0), reverse=True)[:limit]


def _search_like(query: str, terms: list[str], limit: int) -> list[dict[str, Any]]:
    if not terms:
        where = "archived = 0"
        params: list[Any] = []
    else:
        parts = []
        params = []
        for term in terms:
            like = f"%{term}%"
            parts.append("(LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(tags_json) LIKE ?)")
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
    result = [dict(row) for row in rows]
    for row in result:
        _score_memory(row, query, terms)
    return sorted(result, key=lambda row: row.get("score", 0.0), reverse=True)[:limit]


def search_memories(query: str, limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    terms = _query_terms(query)
    try:
        rows = _search_fts(query, terms, limit)
    except Exception:
        rows = []
    if not rows:
        rows = _search_like(query, terms, limit)
    ids = [row["id"] for row in rows]
    if ids:
        with connect() as conn:
            conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now_kst(), memory_id) for memory_id in ids],
            )
            conn.commit()
    return rows


def list_memories(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM memories WHERE archived = 0 ORDER BY importance DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def rebuild_memory_fts() -> None:
    init_db()
    with connect() as conn:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except Exception:
            pass
        conn.commit()
