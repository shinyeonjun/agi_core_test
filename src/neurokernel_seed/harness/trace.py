from __future__ import annotations

import re
from typing import Any

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
]


def redact_text(value: str, *, max_chars: int = 8_000) -> str:
    text = value[:max_chars]
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    if len(value) > max_chars:
        text += "\n[TRUNCATED]"
    return text


def failure_trace(task_id: str, failure_bucket: str, *, state: dict[str, Any] | None = None, action: dict[str, Any] | None = None, expected: dict[str, Any] | None = None, actual: dict[str, Any] | None = None, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "failure_bucket": failure_bucket,
        "state": state or {},
        "chosen_action": action or {},
        "expected_result": expected or {},
        "actual_result": actual or {},
        "analysis": analysis or {},
    }
