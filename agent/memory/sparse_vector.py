from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from agent.config.defaults import env_bool, env_int, now_kst
from agent.core.database import connect, init_db

VECTOR_TYPE = "hashed_char_ngram"
DEFAULT_DIMENSIONS = 4096


def vector_enabled() -> bool:
    return env_bool("AGENT_SPARSE_VECTOR_ENABLED", True)


def vector_dimensions() -> int:
    return max(256, env_int("AGENT_SPARSE_VECTOR_DIMENSIONS", DEFAULT_DIMENSIONS))


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _hash_feature(feature: str, dimensions: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def _features(text: str) -> list[str]:
    normalized = _normalize_text(text)
    compact = re.sub(r"\s+", "", normalized)
    tokens = re.findall(r"[0-9a-z_\uac00-\ud7a3]+", normalized)
    features: list[str] = []
    for n in range(2, 5):
        if len(compact) >= n:
            features.extend(f"c{n}:{compact[i:i+n]}" for i in range(len(compact) - n + 1))
    for token in tokens:
        if len(token) >= 2:
            features.append(f"w:{token}")
    for left, right in zip(tokens, tokens[1:]):
        features.append(f"b:{left}_{right}")
    return features


def sparse_vector(text: str, *, dimensions: int | None = None) -> dict[int, float]:
    dims = dimensions or vector_dimensions()
    counts: dict[int, float] = {}
    for feature in _features(text):
        index = _hash_feature(feature, dims)
        counts[index] = counts.get(index, 0.0) + 1.0
    if not counts:
        return {}
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm <= 0:
        return {}
    return {index: round(value / norm, 6) for index, value in counts.items()}


def cosine_similarity(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return round(sum(value * right.get(index, 0.0) for index, value in left.items()), 6)


def memory_vector_text(memory: dict[str, Any]) -> str:
    tags = memory.get("tags_json")
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except json.JSONDecodeError:
            tags = []
    tag_text = " ".join(str(tag) for tag in (tags or []))
    return " ".join([
        str(memory.get("title") or ""),
        str(memory.get("content") or ""),
        tag_text,
        str(memory.get("memory_type") or ""),
    ]).strip()


def _encode_vector(vector: dict[int, float]) -> str:
    return json.dumps({str(index): value for index, value in sorted(vector.items())}, separators=(",", ":"))


def _decode_vector(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return {int(index): float(value) for index, value in payload.items()}


def upsert_memory_vector(memory: dict[str, Any]) -> bool:
    if not vector_enabled():
        return False
    init_db()
    text = memory_vector_text(memory)
    if not text:
        return False
    dims = vector_dimensions()
    digest = _content_hash(text)
    vector = sparse_vector(text, dimensions=dims)
    if not vector:
        return False
    memory_id = int(memory["id"])
    ts = now_kst()
    with connect() as conn:
        row = conn.execute(
            "SELECT content_hash FROM memory_vectors WHERE memory_id = ? AND vector_type = ?",
            (memory_id, VECTOR_TYPE),
        ).fetchone()
        if row and row["content_hash"] == digest:
            return False
        conn.execute(
            """
            INSERT INTO memory_vectors (
                memory_id, vector_type, dimensions, content_hash, vector_json,
                nonzero_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id, vector_type) DO UPDATE SET
                dimensions = excluded.dimensions,
                content_hash = excluded.content_hash,
                vector_json = excluded.vector_json,
                nonzero_count = excluded.nonzero_count,
                updated_at = excluded.updated_at
            """,
            (memory_id, VECTOR_TYPE, dims, digest, _encode_vector(vector), len(vector), ts),
        )
        conn.commit()
    return True


def backfill_memory_vectors(*, limit: int = 100, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(
            """
            SELECT m.*
            FROM memories m
            LEFT JOIN memory_vectors v ON v.memory_id = m.id AND v.vector_type = ?
            WHERE m.archived = 0
              AND (? = 1 OR v.memory_id IS NULL)
            ORDER BY m.id ASC
            LIMIT ?
            """,
            (VECTOR_TYPE, 1 if force else 0, max(1, int(limit))),
        ).fetchall()]
    if dry_run:
        return {"dry_run": True, "candidate_count": len(rows), "updated": 0, "vector_type": VECTOR_TYPE, "dimensions": vector_dimensions()}
    updated = 0
    for row in rows:
        if upsert_memory_vector(row):
            updated += 1
    return {"dry_run": False, "candidate_count": len(rows), "updated": updated, "vector_type": VECTOR_TYPE, "dimensions": vector_dimensions()}


def vector_status() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        active = conn.execute("SELECT COUNT(*) AS count FROM memories WHERE archived = 0").fetchone()["count"]
        vectorized = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memory_vectors v
            JOIN memories m ON m.id = v.memory_id
            WHERE m.archived = 0 AND v.vector_type = ?
            """,
            (VECTOR_TYPE,),
        ).fetchone()["count"]
        stale = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM memories m
            JOIN memory_vectors v ON v.memory_id = m.id AND v.vector_type = ?
            WHERE m.archived = 0 AND m.updated_at > v.updated_at
            """,
            (VECTOR_TYPE,),
        ).fetchone()["count"]
    coverage = round(float(vectorized) / float(active), 4) if active else 0.0
    return {
        "enabled": vector_enabled(),
        "vector_type": VECTOR_TYPE,
        "dimensions": vector_dimensions(),
        "active_memories": int(active),
        "vectorized_memories": int(vectorized),
        "stale_vectors": int(stale),
        "coverage": coverage,
    }


def search_memory_vectors(query: str, *, limit: int = 20, min_similarity: float = 0.05) -> list[dict[str, Any]]:
    if not vector_enabled():
        return []
    query_vector = sparse_vector(query)
    if not query_vector:
        return []
    init_db()
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(
            """
            SELECT m.*, v.vector_json
            FROM memory_vectors v
            JOIN memories m ON m.id = v.memory_id
            WHERE m.archived = 0 AND v.vector_type = ?
            """,
            (VECTOR_TYPE,),
        ).fetchall()]
    scored: list[dict[str, Any]] = []
    for row in rows:
        similarity = cosine_similarity(query_vector, _decode_vector(row.pop("vector_json", None)))
        if similarity >= min_similarity:
            row["sparse_similarity"] = similarity
            scored.append(row)
    return sorted(scored, key=lambda item: item.get("sparse_similarity", 0.0), reverse=True)[:limit]

