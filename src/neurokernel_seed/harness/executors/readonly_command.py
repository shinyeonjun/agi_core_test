from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from neurokernel_seed.harness.action_catalog import ActionDefinition
from neurokernel_seed.harness.executors.base import ExecutionResult, TimedExecution
from neurokernel_seed.harness.trace import redact_text


class ReadOnlyCommandExecutor:
    """Executes low-risk registry commands without a shell.

    This adapter is intentionally narrow: registry actions can call a Python
    read-only probe and return JSON/text, but they do not get shell expansion,
    inherited secrets, or arbitrary working directories.
    """

    def __init__(self, *, project_root: str | Path = "."):
        self.project_root = Path(project_root).resolve()

    def execute(self, action: ActionDefinition, params: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ExecutionResult:
        timer = TimedExecution()
        try:
            config = action.executor_config
            command = _string_list(config.get("command"), "command")
            timeout_seconds = int(config.get("timeout_seconds", 5))
            output_mode = str(config.get("output") or "json")
            result = subprocess.run(
                command,
                cwd=self.project_root,
                env=_minimal_env(),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            if result.returncode != 0:
                return timer.finish(
                    action.action_id,
                    success=False,
                    result={"returncode": result.returncode},
                    stdout=redact_text(result.stdout),
                    stderr=redact_text(result.stderr),
                    error_type="readonly_command_failed",
                )
            payload = _parse_output(output_mode, result.stdout)
            return timer.finish(action.action_id, success=True, result=payload, stdout=redact_text(result.stdout), stderr=redact_text(result.stderr))
        except Exception as exc:
            return timer.finish(action.action_id, success=False, result={"error": str(exc)}, stderr=redact_text(str(exc)), error_type=exc.__class__.__name__)


def _parse_output(output_mode: str, stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if output_mode == "text":
        return {"text": redact_text(text)}
    parsed = json.loads(text or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("readonly_command JSON output must be an object")
    return parsed


def _minimal_env() -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONIOENCODING": "utf-8",
    }
    for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = [str(item) for item in value]
    if any(not item for item in result):
        raise ValueError(f"{label} cannot contain empty values")
    return result
