from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db
from agent.memory.sparse_vector import search_memory_vectors, upsert_memory_vector


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
        memory_id = int(cur.lastrowid)
    upsert_memory_vector({
        "id": memory_id,
        "title": title,
        "content": content,
        "memory_type": memory_type,
        "tags_json": json.dumps(tags or [], ensure_ascii=False),
    })
    return memory_id


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


def _score_memory(row: dict[str, Any], query: str, terms: list[str], fts_rank: float | None = None, sparse_similarity: float | None = None) -> float:
    haystack = f"{row.get('title', '')} {row.get('content', '')}".lower()
    tags = _parse_tags(row.get("tags_json"))
    keyword_match = sum(1 for term in terms if term in haystack) / max(1, len(terms))
    tag_match = sum(1 for term in terms if any(term in tag.lower() for tag in tags)) / max(1, len(terms))
    use_count_bonus = min(1.0, math.log1p(int(row.get("use_count") or 0)) / 5)
    project_relevance = 1.0 if any(tag in {"core", "project_context", "orangepi", "digital_agi"} for tag in tags) else 0.2
    summary_bonus = 0.16 if row.get("memory_type") in {"summary", "preference_summary", "failure_summary"} or any(tag in {"memory_summary", "compacted"} for tag in tags) else 0.0
    rank_bonus = 0.16 if fts_rank is not None else 0.0
    sparse_bonus = max(0.0, min(1.0, float(sparse_similarity if sparse_similarity is not None else row.get("sparse_similarity") or 0.0)))
    score = (
        keyword_match * 0.24 + tag_match * 0.16 + sparse_bonus * 0.22
        + float(row.get("importance") or 0.0) * 0.16 + _recency_score(row) * 0.08
        + use_count_bonus * 0.08 + project_relevance * 0.06 + summary_bonus + rank_bonus
    )
    row["score"] = round(score, 4)
    row["score_components"] = {
        "keyword": round(keyword_match, 4),
        "tag": round(tag_match, 4),
        "importance": float(row.get("importance") or 0.0),
        "recency": round(_recency_score(row), 4),
        "use_count": round(use_count_bonus, 4),
        "project": round(project_relevance, 4),
        "summary": round(summary_bonus, 4),
        "fts": round(rank_bonus, 4),
        "sparse": round(sparse_bonus, 4),
    }
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


def _merge_candidates(*candidate_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for group in candidate_groups:
        for row in group:
            memory_id = int(row["id"])
            existing = merged.get(memory_id)
            if not existing:
                merged[memory_id] = dict(row)
                continue
            existing["sparse_similarity"] = max(float(existing.get("sparse_similarity") or 0.0), float(row.get("sparse_similarity") or 0.0))
            if row.get("bm25_score") is not None:
                existing["bm25_score"] = row.get("bm25_score")
    return list(merged.values())


def search_memories(query: str, limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    terms = _query_terms(query)
    candidate_limit = max(limit * 5, 50)
    try:
        fts_rows = _search_fts(query, terms, candidate_limit)
    except Exception:
        fts_rows = []
    like_rows = _search_like(query, terms, candidate_limit) if not fts_rows else []
    vector_rows = search_memory_vectors(query, limit=candidate_limit)
    rows = _merge_candidates(fts_rows, like_rows, vector_rows)
    for row in rows:
        _score_memory(row, query, terms, row.get("bm25_score"), row.get("sparse_similarity"))
    rows = sorted(rows, key=lambda row: row.get("score", 0.0), reverse=True)[:limit]
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


def archive_memories(memory_ids: list[int], *, reason: str = "compacted") -> int:
    ids = [int(memory_id) for memory_id in memory_ids if memory_id is not None]
    if not ids:
        return 0
    init_db()
    placeholders = ",".join("?" for _ in ids)
    with connect() as conn:
        rows = conn.execute(f"SELECT id, tags_json FROM memories WHERE id IN ({placeholders}) AND archived = 0", tuple(ids)).fetchall()
        count = 0
        ts = now_kst()
        for row in rows:
            tags = _parse_tags(row["tags_json"])
            for tag in ["archived", reason]:
                if tag not in tags:
                    tags.append(tag)
            cur = conn.execute(
                "UPDATE memories SET archived = 1, updated_at = ?, tags_json = ? WHERE id = ? AND archived = 0",
                (ts, json.dumps(tags, ensure_ascii=False), int(row["id"])),
            )
            count += int(cur.rowcount)
        conn.commit()
        return count


def count_archived_memories() -> int:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM memories WHERE archived = 1").fetchone()
    return int(row["count"] if row else 0)


def rebuild_memory_fts() -> None:
    init_db()
    with connect() as conn:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        except Exception:
            pass
        conn.commit()
