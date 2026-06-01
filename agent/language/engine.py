from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.language.cache import cache_stats, get_cached_interpretation, store_cached_interpretation
from agent.language.codex_engine import CodexLanguageEngine
from agent.language.fallback_rule import FallbackRuleLanguageEngine
from agent.language.schemas import Interpretation, normalize_interpretation

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"authorization:\s*bearer\s+[^\s,;]+", re.IGNORECASE),
    re.compile(r"(?m)^\s*[A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=.*$", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^\s,'\"]+", re.IGNORECASE),
    re.compile(r"https://discord(?:app)?\.com/api/webhooks/[^\s]+", re.IGNORECASE),
    re.compile(r"\.env(?:\.[\w-]+)?", re.IGNORECASE),
]


def _json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _redact_text(value: str, max_length: int = 1000) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    if len(result) > max_length:
        result = result[:max_length] + "...[truncated]"
    return result


def _sanitize(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_sanitize(item) for item in value[:30]]
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    return value


def record_interpretation(
    *,
    source_event_id: int | None,
    engine: str,
    input_text: str,
    result: Interpretation,
    accepted: bool,
    fallback_reason: str | None = None,
) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO interpretation_logs (
                created_at, source_event_id, engine, input_text, result_json,
                confidence, accepted, fallback_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst(),
                source_event_id,
                engine,
                _redact_text(input_text),
                _json(_sanitize(result.to_dict())),
                result.confidence,
                1 if accepted else 0,
                fallback_reason,
            ),
        )
        conn.commit()
    return int(cur.lastrowid)


def list_interpretation_logs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM interpretation_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def language_cache_stats() -> dict[str, Any]:
    return cache_stats()


def get_language_engine() -> Any:
    mode = os.getenv("AGENT_LANGUAGE_ENGINE", "codex").strip().lower()
    fallback = FallbackRuleLanguageEngine()
    if mode in {"codex", "codex_cli"}:
        return CodexLanguageEngine(fallback=fallback)
    return fallback


def interpret_user_message(text: str, context: dict[str, Any] | None = None, *, source_event_id: int | None = None, log: bool = True) -> dict[str, Any]:
    engine = get_language_engine()
    accepted = True
    fallback_reason = None
    cache_hit = False
    try:
        if getattr(engine, "name", "") == "codex":
            result = get_cached_interpretation(text, engine="codex")
            cache_hit = result is not None
        else:
            result = None
        if result is None:
            result = engine.interpret_user_message(text, context or {})
            if getattr(engine, "name", "") == "codex":
                store_cached_interpretation(text, normalize_interpretation(result, engine="codex"), engine="codex")
        result = normalize_interpretation(result, engine=getattr(engine, "name", "unknown"))
    except Exception as exc:
        fallback_reason = f"engine_error:{type(exc).__name__}"
        fallback = FallbackRuleLanguageEngine()
        result = fallback.interpret_user_message(text, context or {})
        result = normalize_interpretation(result, engine=fallback.name, fallback_reason=fallback_reason)
        accepted = False

    if result.fallback_reason:
        fallback_reason = result.fallback_reason
        accepted = False
    if log:
        log_id = record_interpretation(
            source_event_id=source_event_id,
            engine=result.engine,
            input_text=text,
            result=result,
            accepted=accepted,
            fallback_reason=fallback_reason,
        )
    else:
        log_id = None
    data = result.to_dict()
    data["log_id"] = log_id
    data["cache_hit"] = cache_hit
    return data
