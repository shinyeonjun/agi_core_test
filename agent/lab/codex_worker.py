from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.codex_config import codex_exec_args, codex_exec_config
from agent.config.defaults import env_bool, project_root
from agent.core.autonomy import current_profile
from agent.core.capabilities import ALLOWED_CODEX_WORK_SANDBOXES, codex_work_sandbox, codex_worker_blockers, codex_worker_enabled
from agent.core.events import log_event
from agent.core.policy import PolicyEngine
from agent.tools.full_device import redact_action_output
from agent.workspace.executor import write_text_artifact

MAX_WORKER_OUTPUT_CHARS = 6000
UNSAFE_CHANGED_FILE_PATTERNS = (
    re.compile(r"(^|/)\.env(?:[.\w-]*)?$", re.IGNORECASE),
    re.compile(r"(^|/)\.ssh(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(id_rsa|id_ed25519|authorized_keys)$", re.IGNORECASE),
    re.compile(r"(^|/)(secrets?|credentials|token)([.\w-]*)?$", re.IGNORECASE),
    re.compile(r"(^|/)\.(npmrc|pypirc|netrc)$", re.IGNORECASE),
)


SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r'''(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,'"]+'''),
    re.compile(r"(?im)^\s*[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API[_-]?KEY)[A-Z0-9_]*\s*=.*$"),
)


def _redact(text: str | None, max_chars: int = MAX_WORKER_OUTPUT_CHARS) -> str:
    result = redact_action_output(text or "")
    for pattern in SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    if len(result) > max_chars:
        result = result[:max_chars] + "...[truncated]"
    return result


def _git_status(root: Path) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return [f"git_status_error:{type(exc).__name__}"]
    return [line[:240] for line in (completed.stdout or "").splitlines()[:80]]


def _changed_path(status_line: str) -> str:
    line = status_line.strip()
    if not line:
        return ""
    if " -> " in line:
        line = line.rsplit(" -> ", 1)[-1]
    if len(line) >= 4 and line[2] == " ":
        return line[3:].strip()
    parts = line.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _unsafe_changed_files(status_lines: list[str]) -> list[str]:
    unsafe: list[str] = []
    for line in status_lines:
        path = _changed_path(line).replace("\\", "/").strip()
        if path and any(pattern.search(path) for pattern in UNSAFE_CHANGED_FILE_PATTERNS):
            unsafe.append(path[:180])
    return sorted(set(unsafe))


def _blocked_result(reason: str, *, goal_id: int | None, task_id: int | None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "codex_work_blocked",
        "attempted": False,
        "executed": False,
        "reason": reason,
        "goal_id": goal_id,
        "task_id": task_id,
        "profile": current_profile(),
    }
    result.update(extra)
    log_event("lab", "codex_work_blocked", reason, result, 0.72)
    return result


def _worker_prompt(user_request: str, *, goal_id: int | None, task_id: int | None) -> str:
    return "\n".join(
        [
            "You are Agent Core's Codex work worker.",
            "Implement the user's code-change request inside the current repository when feasible.",
            "",
            "Hard rules:",
            "- Do not read, print, copy, or store .env files, tokens, passwords, private keys, or SSH keys.",
            "- Do not run destructive commands.",
            "- Do not install packages, change systemd, use sudo, or perform external network actions unless the user request explicitly requires it and the existing repo already supports it.",
            "- Keep changes scoped and run relevant local tests when feasible.",
            "- If the request is too broad, create a small safe implementation step and report the remaining work.",
            "- Do not claim AGI or unrestricted autonomy.",
            "",
            f"goal_id: {goal_id}",
            f"task_id: {task_id}",
            f"user_request: {user_request[:2000]}",
            "",
            "Return a concise Korean report with: what changed, tests run, remaining risk.",
        ]
    )


def run_codex_work(user_request: str, *, goal_id: int | None = None, task_id: int | None = None) -> dict[str, Any]:
    root = project_root()
    blockers = codex_worker_blockers()
    if blockers:
        return _blocked_result("codex_worker_not_available", goal_id=goal_id, task_id=task_id, blockers=blockers, sandbox=codex_work_sandbox())
    if not codex_worker_enabled():
        return _blocked_result("codex_worker_disabled", goal_id=goal_id, task_id=task_id)
    if not root.exists():
        return _blocked_result("project_root_missing", goal_id=goal_id, task_id=task_id, project_root=str(root))

    sandbox = codex_work_sandbox()
    if sandbox not in ALLOWED_CODEX_WORK_SANDBOXES:
        return _blocked_result("invalid_codex_work_sandbox", goal_id=goal_id, task_id=task_id, sandbox=sandbox)

    policy = PolicyEngine(profile="safe").classify_decision(user_request, action_type="codex_work")
    if policy.denied or policy.requires_approval:
        return _blocked_result(
            policy.reason,
            goal_id=goal_id,
            task_id=task_id,
            policy={"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "matched_rules": policy.matched_rules},
        )

    config = codex_exec_config("WORK", default_reasoning="high", default_timeout=120)
    output_path = Path(tempfile.gettempdir()) / f"agent_core_codex_work_{os.getpid()}_{uuid4().hex}.md"
    before_status = _git_status(root)
    start = time.monotonic()
    args = [
        *codex_exec_args(config),
        "--sandbox",
        sandbox,
        "--skip-git-repo-check",
        "--ephemeral",
        "--output-last-message",
        str(output_path),
        _worker_prompt(user_request, goal_id=goal_id, task_id=task_id),
    ]
    try:
        completed = subprocess.run(
            args,
            cwd=root,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=config.timeout_seconds,
            check=False,
        )
        report = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else (completed.stdout or "").strip()
        returncode = completed.returncode
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        report = exc.stdout if isinstance(exc.stdout, str) else ""
        returncode = 124
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
    except Exception as exc:
        report = ""
        returncode = 127
        stderr = type(exc).__name__
    finally:
        if output_path.exists():
            output_path.unlink()

    after_status = _git_status(root)
    duration_ms = int((time.monotonic() - start) * 1000)
    unsafe_files = _unsafe_changed_files(after_status)
    status = "codex_work_completed" if returncode == 0 and not unsafe_files else "codex_work_blocked" if unsafe_files else "codex_work_failed"
    result = {
        "status": status,
        "attempted": True,
        "executed": returncode == 0 and not unsafe_files,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "report": _redact(report),
        "stderr": _redact(stderr, 1200),
        "changed_files": after_status,
        "changed_files_before": before_status,
        "unsafe_changed_files": unsafe_files,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "profile": current_profile(),
        "sandbox": sandbox,
        "policy": {"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "matched_rules": policy.matched_rules},
    }
    artifact = write_text_artifact(
        "reports",
        f"codex-work-{task_id or 'manual'}-{uuid4().hex[:8]}.md",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        "codex_work_report",
        "Codex work report",
        {"goal_id": goal_id, "task_id": task_id, "returncode": returncode},
    )
    result["artifact_id"] = artifact.get("id")
    log_event("lab", status, str(task_id or goal_id or ""), result, 0.86 if status == "codex_work_completed" else 0.72)
    return result


def can_run_codex_work() -> bool:
    return env_bool("AGENT_CODEX_WORKER_AVAILABLE_OVERRIDE", True) and not codex_worker_blockers()
