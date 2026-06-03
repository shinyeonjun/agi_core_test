from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.codex_config import codex_exec_args, codex_exec_config
from agent.config.defaults import env_bool, env_int, project_root, workspace_root
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import current_profile
from agent.core.capabilities import ALLOWED_CODEX_WORK_BACKENDS, ALLOWED_CODEX_WORK_SANDBOXES, codex_work_backend, codex_work_sandbox, codex_worker_blockers, codex_worker_enabled
from agent.core.events import log_event
from agent.core.policy import ActionProposal, PolicyEngine
from agent.core.self_code_review import review_codex_work_result
from agent.core.self_improvement_release import evaluate_release_candidate
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


def _git(args: list[str], root: Path, *, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


def _is_git_repo(root: Path) -> bool:
    try:
        completed = _git(["rev-parse", "--is-inside-work-tree"], root, timeout=5)
    except Exception:
        return False
    return completed.returncode == 0 and (completed.stdout or "").strip() == "true"


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


def _maybe_create_self_improvement_approval(result: dict[str, Any]) -> int | None:
    review = result.get("code_review") if isinstance(result.get("code_review"), dict) else {}
    verdict = str(review.get("verdict") or "")
    release_gate = review.get("release_gate") if isinstance(review.get("release_gate"), dict) else {}
    if verdict not in {"needs_approval", "ready_for_approval"}:
        return None
    proposal = ActionProposal(
        action_type="self_improvement_apply",
        description=f"자가개선 결과 main 반영 승인: {result.get('worktree_branch') or 'worktree'}",
        payload={
            "goal_id": result.get("goal_id"),
            "task_id": result.get("task_id"),
            "worktree": result.get("worktree"),
            "worktree_branch": result.get("worktree_branch"),
            "changed_files": result.get("changed_files") or [],
            "release_gate_status": release_gate.get("status"),
            "review_verdict": verdict,
            "rollback_plan": release_gate.get("rollback_plan"),
            "note": "승인 전 main에는 반영되지 않는다.",
        },
        risk_level=str(release_gate.get("effective_risk") or (result.get("policy") or {}).get("risk_level") or "medium"),
        requires_approval=True,
    )
    approval_id = ApprovalStore().create_approval(proposal)
    log_event(
        "approval",
        "self_improvement_apply_approval_created",
        str(approval_id),
        {"approval_id": approval_id, "goal_id": result.get("goal_id"), "task_id": result.get("task_id"), "verdict": verdict},
        0.82,
    )
    return approval_id


def _work_loop_timeout() -> int:
    return max(30, min(3600, env_int("AGENT_WORK_LOOP_TIMEOUT", env_int("AGENT_CODEX_WORK_TIMEOUT", 600))))


def _work_loop_iterations() -> int:
    return max(1, min(5, env_int("AGENT_WORK_LOOP_ITERATIONS", 2)))


def _work_loop_worktree_enabled() -> bool:
    return env_bool("AGENT_WORK_LOOP_WORKTREE", True)


def _work_loop_verify_commands() -> list[str]:
    value = os.getenv("AGENT_WORK_LOOP_VERIFY_COMMANDS")
    if value is None:
        return [f"{shlex.quote(sys.executable)} -m pytest -q"]
    stripped = value.strip()
    if stripped.lower() in {"", "0", "false", "off", "none", "skip"}:
        return []
    return [part.strip() for part in stripped.split(";") if part.strip()]


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


def _native_loop_prompt(
    user_request: str,
    *,
    goal_id: int | None,
    task_id: int | None,
    worktree: Path,
    iteration: int,
    max_iterations: int,
    previous_evidence: list[dict[str, Any]],
) -> str:
    previous = json.dumps(previous_evidence[-6:], ensure_ascii=False, indent=2) if previous_evidence else "[]"
    return "\n".join(
        [
            "Core Native Work Loop",
            "",
            "You are Agent Core's code-work executor. Core owns policy, memory, Discord reporting, task queues, and final safety checks.",
            "This is a user-triggered work run. Autonomous study/self-improvement loops are separate and must not be mixed into this response.",
            "",
            "Mission:",
            "- Finish the requested code work as far as safely possible inside the given repository/worktree.",
            "- Plan, implement, verify, and leave observable evidence for Core.",
            "- If verification fails, use the previous evidence to make one focused repair pass.",
            "- Prefer small, scoped changes over broad rewrites.",
            "",
            "Core safety boundary:",
            "- Do not read, print, copy, or store .env files, tokens, passwords, private keys, or SSH keys.",
            "- Do not modify .env, .ssh, key files, token files, credentials, package manager auth files, or deployment secrets.",
            "- Do not use sudo, systemd writes, package installation, or external network actions.",
            "- Do not claim AGI, consciousness, unrestricted autonomy, or policy bypass.",
            "- If the task needs a forbidden action, stop and report the blocker with evidence.",
            "",
            "Verification requirement:",
            "- Run relevant tests or checks when feasible, but Core may run its own verification after your pass.",
            "- If tests cannot run, explain the exact blocker and what should be run next.",
            "- Completion requires changed-file summary, verification commands, and remaining risks.",
            "",
            f"goal_id: {goal_id}",
            f"task_id: {task_id}",
            f"worktree: {worktree}",
            f"iteration: {iteration}/{max_iterations}",
            "",
            "Previous Core evidence:",
            previous,
            "",
            "User request:",
            user_request[:4000],
            "",
            "Return a concise Korean report with: plan, changes, tests/evidence, blocked items, remaining risk.",
        ]
    )


def _create_native_worktree(root: Path, task_id: int | None) -> tuple[Path, str | None, str | None]:
    if not _work_loop_worktree_enabled():
        return root, None, None
    if not _is_git_repo(root):
        raise RuntimeError("git_repo_required_for_native_work_loop")
    branch = f"codex/native-loop-task-{task_id or 'manual'}-{uuid4().hex[:8]}"
    path = workspace_root() / "tasks" / "native-worktrees" / branch.replace("/", "-")
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _git(["worktree", "add", "-b", branch, str(path), "HEAD"], root, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"git_worktree_add_failed:{_redact(completed.stderr, 500)}")
    return path.resolve(), branch, "created"


def _run_codex_exec_backend(root: Path, user_request: str, *, goal_id: int | None, task_id: int | None, sandbox: str) -> dict[str, Any]:
    config = codex_exec_config("WORK", default_reasoning="high", default_timeout=120)
    output_path = Path(tempfile.gettempdir()) / f"agent_core_codex_work_{os.getpid()}_{uuid4().hex}.md"
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
        return {"returncode": completed.returncode, "report": report, "stderr": completed.stderr or "", "model": config.model, "reasoning_effort": config.reasoning_effort, "timeout_seconds": config.timeout_seconds}
    except subprocess.TimeoutExpired as exc:
        report = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
        return {"returncode": 124, "report": report, "stderr": stderr, "model": config.model, "reasoning_effort": config.reasoning_effort, "timeout_seconds": config.timeout_seconds}
    except Exception as exc:
        return {"returncode": 127, "report": "", "stderr": type(exc).__name__, "model": config.model, "reasoning_effort": config.reasoning_effort, "timeout_seconds": config.timeout_seconds}
    finally:
        if output_path.exists():
            output_path.unlink()


def _run_verification_command(root: Path, command: str, *, timeout: int) -> dict[str, Any]:
    try:
        args = shlex.split(command)
    except ValueError as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": f"parse_error:{type(exc).__name__}"}
    if not args:
        return {"command": command, "returncode": 0, "stdout": "", "stderr": "skipped_empty_command"}
    try:
        completed = subprocess.run(
            args,
            cwd=root,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": _redact(completed.stdout or "", 1600),
            "stderr": _redact(completed.stderr or "", 1600),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": 124,
            "stdout": _redact(exc.stdout if isinstance(exc.stdout, str) else "", 1600),
            "stderr": _redact(exc.stderr if isinstance(exc.stderr, str) else "timeout", 1600),
        }
    except Exception as exc:
        return {"command": command, "returncode": 127, "stdout": "", "stderr": type(exc).__name__}


def _verification_passed(evidence: list[dict[str, Any]]) -> bool:
    latest_with_verification = None
    for item in reversed(evidence):
        verification = item.get("verification")
        if verification:
            latest_with_verification = verification
            break
    if not latest_with_verification:
        return False
    return all(record.get("returncode") == 0 for record in latest_with_verification)


def _run_native_loop_backend(root: Path, user_request: str, *, goal_id: int | None, task_id: int | None, sandbox: str) -> dict[str, Any]:
    worktree, branch, worktree_status = _create_native_worktree(root, task_id)
    config = codex_exec_config("WORK", default_reasoning="high", default_timeout=120)
    timeout_seconds = _work_loop_timeout()
    max_iterations = _work_loop_iterations()
    verify_commands = _work_loop_verify_commands()
    evidence: list[dict[str, Any]] = []
    last_report = ""
    last_stderr = ""
    last_returncode = 127
    completed_successfully = False

    for iteration in range(1, max_iterations + 1):
        output_path = Path(tempfile.gettempdir()) / f"agent_core_native_work_{os.getpid()}_{uuid4().hex}.md"
        args = [
            *codex_exec_args(config),
            "--sandbox",
            sandbox,
            "--skip-git-repo-check",
            "--ephemeral",
            "--output-last-message",
            str(output_path),
            _native_loop_prompt(
                user_request,
                goal_id=goal_id,
                task_id=task_id,
                worktree=worktree,
                iteration=iteration,
                max_iterations=max_iterations,
                previous_evidence=evidence,
            ),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                args,
                cwd=worktree,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
            last_report = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else (completed.stdout or "").strip()
            last_stderr = completed.stderr or ""
            last_returncode = completed.returncode
        except subprocess.TimeoutExpired as exc:
            last_report = exc.stdout if isinstance(exc.stdout, str) else ""
            last_stderr = exc.stderr if isinstance(exc.stderr, str) else "timeout"
            last_returncode = 124
        except Exception as exc:
            last_report = ""
            last_stderr = type(exc).__name__
            last_returncode = 127
        finally:
            if output_path.exists():
                output_path.unlink()

        status_lines = _git_status(worktree)
        unsafe_files = _unsafe_changed_files(status_lines)
        verification = [
            _run_verification_command(worktree, command, timeout=max(30, min(300, timeout_seconds // 2)))
            for command in verify_commands
        ]
        verification_failed = any(item.get("returncode") not in {0, None} for item in verification)
        iteration_evidence = {
            "phase": "iteration",
            "iteration": iteration,
            "codex_returncode": last_returncode,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "changed_files": status_lines,
            "unsafe_changed_files": unsafe_files,
            "verification": verification,
        }
        evidence.append(iteration_evidence)
        if unsafe_files:
            last_returncode = 126
            break
        if last_returncode == 0 and not verification_failed:
            completed_successfully = True
            break
        if last_returncode == 0 and verification_failed:
            last_returncode = 125

    return {
        "returncode": 0 if completed_successfully else last_returncode,
        "report": last_report,
        "stderr": last_stderr,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "timeout_seconds": timeout_seconds,
        "worktree": str(worktree),
        "worktree_branch": branch,
        "worktree_status": worktree_status,
        "mode": "native_loop",
        "iterations_used": len(evidence),
        "max_iterations": max_iterations,
        "verification_commands": verify_commands,
        "evidence_ledger": evidence,
        "integration_status": "worktree_pending_review" if branch else "direct_workspace_changes",
    }


def run_codex_work(user_request: str, *, goal_id: int | None = None, task_id: int | None = None, self_improvement: bool = False) -> dict[str, Any]:
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
    backend = codex_work_backend()
    if backend not in ALLOWED_CODEX_WORK_BACKENDS:
        return _blocked_result("invalid_codex_work_backend", goal_id=goal_id, task_id=task_id, backend=backend)
    if self_improvement and backend != "native_loop":
        return _blocked_result("self_improvement_requires_native_loop", goal_id=goal_id, task_id=task_id, backend=backend)

    policy_text = "Core self improvement code task in isolated worktree. Keep changes inside allowed repository files." if self_improvement else user_request
    policy = PolicyEngine(profile="safe").classify_decision(policy_text, action_type="codex_work")
    if policy.denied or policy.requires_approval:
        return _blocked_result(
            policy.reason,
            goal_id=goal_id,
            task_id=task_id,
            policy={"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "matched_rules": policy.matched_rules},
        )

    before_status = _git_status(root)
    start = time.monotonic()
    try:
        if backend == "native_loop":
            backend_result = _run_native_loop_backend(root, user_request, goal_id=goal_id, task_id=task_id, sandbox=sandbox)
        else:
            backend_result = _run_codex_exec_backend(root, user_request, goal_id=goal_id, task_id=task_id, sandbox=sandbox)
    except Exception as exc:
        backend_result = {"returncode": 127, "report": "", "stderr": type(exc).__name__, "backend_error": str(exc)[:500]}
    work_root = Path(str(backend_result.get("worktree") or root)).resolve()
    after_status = _git_status(work_root)
    duration_ms = int((time.monotonic() - start) * 1000)
    unsafe_files = _unsafe_changed_files(after_status)
    returncode = int(backend_result.get("returncode", 127))
    status = "codex_work_completed" if returncode == 0 and not unsafe_files else "codex_work_blocked" if unsafe_files else "codex_work_failed"
    result = {
        "status": status,
        "attempted": True,
        "executed": returncode == 0 and not unsafe_files,
        "returncode": returncode,
        "duration_ms": duration_ms,
        "report": _redact(str(backend_result.get("report") or "")),
        "stderr": _redact(str(backend_result.get("stderr") or ""), 1200),
        "changed_files": after_status,
        "changed_files_before": before_status,
        "unsafe_changed_files": unsafe_files,
        "backend": backend,
        "model": backend_result.get("model"),
        "reasoning_effort": backend_result.get("reasoning_effort"),
        "timeout_seconds": backend_result.get("timeout_seconds"),
        "profile": current_profile(),
        "goal_id": goal_id,
        "task_id": task_id,
        "sandbox": sandbox,
        "policy": {"risk_level": policy.risk_level, "requires_approval": policy.requires_approval, "denied": policy.denied, "matched_rules": policy.matched_rules},
    }
    for key in ("worktree", "worktree_branch", "worktree_status", "mode", "backend_error", "iterations_used", "max_iterations", "verification_commands", "evidence_ledger", "integration_status"):
        if backend_result.get(key) is not None:
            result[key] = backend_result[key]
    evidence = backend_result.get("evidence_ledger") if isinstance(backend_result.get("evidence_ledger"), list) else []
    result["release_gate"] = evaluate_release_candidate(
        changed_files=after_status,
        worktree_isolated=backend_result.get("worktree_status") == "created",
        tests_passed=_verification_passed(evidence),
        audit_passed=False,
        eval_passed=False,
        review_passed=False,
        approval_status="pending",
        risk_level=policy.risk_level,
        rollback_plan=f"remove worktree branch {backend_result.get('worktree_branch')}" if backend_result.get("worktree_branch") else None,
        secrets_touched=bool(unsafe_files),
    )
    result["code_review"] = review_codex_work_result(result, self_improvement=self_improvement)
    if self_improvement:
        approval_id = _maybe_create_self_improvement_approval(result)
        if approval_id is not None:
            result["approval_id"] = approval_id
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
