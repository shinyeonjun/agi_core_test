from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from typing import Any

from agent.config.defaults import KST, env_int, now_kst
from agent.core.database import connect, init_db
from agent.language.schemas import Interpretation, normalize_interpretation

CACHE_VERSION = "language-cache-v1"


def normalize_cache_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def cache_key(text: str, *, engine: str = "codex") -> str:
    normalized = normalize_cache_text(text)
    digest = hashlib.sha256(f"{CACHE_VERSION}:{engine}:{normalized}".encode("utf-8")).hexdigest()
    return digest


def cache_ttl_seconds() -> int:
    return env_int("AGENT_LANGUAGE_CACHE_TTL_SECONDS", 604800)


def _is_fresh(updated_at: str, ttl_seconds: int) -> bool:
    if ttl_seconds <= 0:
        return False
    try:
        timestamp = datetime.fromisoformat(updated_at)
    except ValueError:
        return False
    return datetime.now(timestamp.tzinfo or KST) <= timestamp + timedelta(seconds=ttl_seconds)


def get_cached_interpretation(text: str, *, engine: str = "codex") -> Interpretation | None:
    init_db()
    key = cache_key(text, engine=engine)
    with connect() as conn:
        row = conn.execute("SELECT * FROM language_interpretation_cache WHERE cache_key = ?", (key,)).fetchone()
        if not row:
            return None
        if not _is_fresh(str(row["updated_at"]), cache_ttl_seconds()):
            return None
        conn.execute(
            "UPDATE language_interpretation_cache SET hit_count = hit_count + 1, last_used_at = ? WHERE cache_key = ?",
            (now_kst(), key),
        )
        conn.commit()
    try:
        result = json.loads(row["result_json"] or "{}")
    except json.JSONDecodeError:
        return None
    result["engine"] = f"{engine}_cache"
    return normalize_interpretation(result, engine=f"{engine}_cache")


def store_cached_interpretation(text: str, result: Interpretation, *, engine: str = "codex") -> None:
    if result.engine != engine or result.fallback_reason:
        return
    init_db()
    key = cache_key(text, engine=engine)
    now = now_kst()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO language_interpretation_cache (
                cache_key, created_at, updated_at, engine, normalized_input, result_json,
                hit_count, last_used_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                updated_at = excluded.updated_at,
                engine = excluded.engine,
                normalized_input = excluded.normalized_input,
                result_json = excluded.result_json
            """,
            (
                key,
                now,
                now,
                engine,
                normalize_cache_text(text)[:500],
                json.dumps(result.to_dict(), ensure_ascii=False),
                now,
            ),
        )
        conn.commit()


def cache_stats() -> dict[str, Any]:
    init_db()
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM language_interpretation_cache").fetchone()["count"]
        hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) AS count FROM language_interpretation_cache").fetchone()["count"]
        rows = conn.execute(
            """
            SELECT engine, COUNT(*) AS entries, COALESCE(SUM(hit_count), 0) AS hits
            FROM language_interpretation_cache
            GROUP BY engine
            ORDER BY entries DESC
            """
        ).fetchall()
    return {"entries": int(total), "hits": int(hits), "by_engine": [dict(row) for row in rows]}
