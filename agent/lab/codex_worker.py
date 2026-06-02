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
from agent.core.capabilities import codex_worker_enabled
from agent.core.events import log_event
from agent.tools.full_device import redact_action_output
from agent.workspace.executor import write_text_artifact

MAX_WORKER_OUTPUT_CHARS = 6000


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
    if not codex_worker_enabled():
        return {"status": "blocked", "executed": False, "reason": "codex_worker_disabled"}
    if not root.exists():
        return {"status": "blocked", "executed": False, "reason": "project_root_missing", "project_root": str(root)}

    config = codex_exec_config("WORK", default_reasoning="high", default_timeout=120)
    output_path = Path(tempfile.gettempdir()) / f"agent_core_codex_work_{os.getpid()}_{uuid4().hex}.md"
    before_status = _git_status(root)
    start = time.monotonic()
    args = [
        *codex_exec_args(config),
        "--sandbox",
        os.getenv("AGENT_CODEX_WORK_SANDBOX", "workspace-write"),
        "--skip-git-repo-check",
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
    status = "codex_work_completed" if returncode == 0 else "codex_work_failed"
    result = {
        "status": status,
        "executed": True,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "report": _redact(report),
        "stderr": _redact(stderr, 1200),
        "changed_files": after_status,
        "changed_files_before": before_status,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
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
    log_event("lab", "codex_work_completed" if returncode == 0 else "codex_work_failed", str(task_id or goal_id or ""), result, 0.86 if returncode == 0 else 0.72)
    return result


def can_run_codex_work() -> bool:
    return codex_worker_enabled() and env_bool("AGENT_CODEX_WORKER_AVAILABLE_OVERRIDE", True)
