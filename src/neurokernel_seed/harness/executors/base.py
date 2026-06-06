from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    action_id: str
    success: bool
    started_at: str
    ended_at: str
    duration_ms: float
    result: dict[str, Any] = field(default_factory=dict)
    stdout_redacted: str = ""
    stderr_redacted: str = ""
    error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "result": self.result,
            "stdout_redacted": self.stdout_redacted,
            "stderr_redacted": self.stderr_redacted,
            "error_type": self.error_type,
        }


class TimedExecution:
    def __init__(self):
        self.started_at = _now()
        self.start = time.perf_counter()

    def finish(self, action_id: str, *, success: bool, result: dict[str, Any] | None = None, stdout: str = "", stderr: str = "", error_type: str | None = None) -> ExecutionResult:
        ended_at = _now()
        return ExecutionResult(action_id, success, self.started_at, ended_at, (time.perf_counter() - self.start) * 1000.0, result or {}, stdout, stderr, error_type)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

