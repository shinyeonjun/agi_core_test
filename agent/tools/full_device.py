from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from agent.config.defaults import project_root
from agent.core.autonomy import current_profile
from agent.core.policy import PolicyEngine
from agent.tools.action_log import create_action_run, finish_action_run, get_action_run, list_action_runs

MAX_CAPTURE_CHARS = 20000
DEFAULT_TIMEOUT_SECONDS = 15
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r'''(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,'"]+'''),
    re.compile(r"(?im)^\s*[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API[_-]?KEY)[A-Z0-9_]*\s*=.*$"),
)


def redact_action_output(text: str | None) -> str:
    result = text or ""
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    if "PRIVATE KEY" in result.upper():
        return "<unsafe output blocked>"
    if len(result) > MAX_CAPTURE_CHARS:
        result = result[:MAX_CAPTURE_CHARS] + "...[truncated]"
    return result


def lightweight_snapshot(cwd: str | None = None) -> dict[str, Any]:
    commands = {
        "disk": ["df", "-h", "/"],
        "memory": ["free", "-h"],
        "failed_services": ["systemctl", "--failed", "--no-pager"],
        "git_status": ["git", "status", "--short"],
    }
    snapshot: dict[str, Any] = {}
    for name, argv in commands.items():
        try:
            completed = subprocess.run(argv, cwd=cwd or project_root(), text=True, capture_output=True, timeout=5, check=False)
            snapshot[name] = {
                "returncode": completed.returncode,
                "stdout": redact_action_output(completed.stdout),
                "stderr": redact_action_output(completed.stderr),
            }
        except Exception as exc:
            snapshot[name] = {"returncode": 127, "stdout": "", "stderr": redact_action_output(str(exc))}
    return snapshot


def _resolve_cwd(cwd: str | None) -> str:
    path = Path(cwd).expanduser().resolve() if cwd else project_root()
    return str(path)


def _command_to_argv(command: str, use_shell: bool) -> list[str] | str:
    return command if use_shell else shlex.split(command)


def run_action(command: str, *, cwd: str | None = None, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, goal_id: int | None = None, use_shell: bool = False) -> dict[str, Any]:
    profile = current_profile()
    action_type = "shell_command" if use_shell else "local_command"
    policy = PolicyEngine(profile=profile).classify_text(command, action_type=action_type)
    resolved_cwd = _resolve_cwd(cwd)
    if profile != "full_device_lab":
        action_id = create_action_run(
            goal_id=goal_id,
            action_type=action_type,
            command=command,
            cwd=resolved_cwd,
            profile=profile,
            risk_level=policy.risk_level,
            status="blocked",
            before_snapshot=None,
            result_summary="profile_not_full_device_lab",
        )
        return {"id": action_id, "executed": False, "status": "blocked", "reason": "profile_not_full_device_lab", "policy": policy.to_dict()}
    if policy.denied or policy.requires_approval:
        reason = policy.denied_reason or "approval_required"
        action_id = create_action_run(
            goal_id=goal_id,
            action_type=action_type,
            command=command,
            cwd=resolved_cwd,
            profile=profile,
            risk_level=policy.risk_level,
            status="blocked",
            before_snapshot=None,
            result_summary=reason,
        )
        return {"id": action_id, "executed": False, "status": "blocked", "reason": reason, "policy": policy.to_dict()}

    before = lightweight_snapshot(resolved_cwd)
    argv = _command_to_argv(command, use_shell)
    action_id = create_action_run(
        goal_id=goal_id,
        action_type=action_type,
        command=argv,
        cwd=resolved_cwd,
        profile=profile,
        risk_level=policy.risk_level,
        status="running",
        before_snapshot=before,
    )
    try:
        completed = subprocess.run(argv, cwd=resolved_cwd, shell=use_shell, text=True, capture_output=True, timeout=timeout_seconds, check=False)
        stdout = redact_action_output(completed.stdout)
        stderr = redact_action_output(completed.stderr)
        returncode = completed.returncode
        status = "completed"
        summary = f"rc={returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout = redact_action_output(exc.stdout if isinstance(exc.stdout, str) else "")
        stderr = redact_action_output(exc.stderr if isinstance(exc.stderr, str) else "timeout")
        status = "timeout"
        returncode = 124
        summary = "timeout"
    except FileNotFoundError as exc:
        stdout = ""
        stderr = redact_action_output(str(exc))
        status = "failed"
        returncode = 127
        summary = "command_not_found"
    after = lightweight_snapshot(resolved_cwd)
    finish_action_run(action_id, status=status, returncode=returncode, stdout=stdout, stderr=stderr, after_snapshot=after, result_summary=summary)
    return {
        "id": action_id,
        "executed": True,
        "status": status,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "policy": policy.to_dict(),
    }


__all__ = ["run_action", "list_action_runs", "get_action_run", "redact_action_output", "lightweight_snapshot"]
