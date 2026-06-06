from __future__ import annotations

import os
import signal
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .trace import redact_text


class SelfPatchError(RuntimeError):
    pass


class CommandRunner(Protocol):
    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout_seconds: int,
        idle_timeout_seconds: int | None = None,
        watchdog_root: Path | None = None,
        watchdog_poll_seconds: float = 2.0,
        watchdog_file_scan_seconds: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class SelfPatchConfig:
    project_root: Path = Path(".")
    run_root: Path = Path("artifacts/self_patch")
    codex_bin: str = "codex"
    codex_timeout_seconds: int = 420
    test_timeout_seconds: int = 300
    codex_idle_timeout_seconds: int = 120
    test_idle_timeout_seconds: int = 90
    watchdog_poll_seconds: float = 2.0
    watchdog_file_scan_seconds: float = 10.0
    test_command: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    codex_model: str | None = None
    sandbox: str = "workspace-write"
    isolation_mode: str = "auto"
    ignore_names: tuple[str, ...] = field(
        default_factory=lambda: (
            ".git",
            "venv",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "data",
            "artifacts",
            "baselines",
            "logs",
            "language_workspace",
            "model_transrate",
        )
    )


class CodexSelfPatchWorker:
    def __init__(self, config: SelfPatchConfig, *, runner: CommandRunner | None = None):
        self.config = config
        self.runner = runner or _run_command

    def run(self, *, job_id: str, work: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if not job_id.strip():
            raise SelfPatchError("job_id is required")
        project_root = self.config.project_root.resolve()
        if not project_root.exists():
            raise SelfPatchError(f"project_root does not exist: {project_root}")

        run_dir = (project_root / self.config.run_root / _safe_name(job_id)).resolve()
        workspace = run_dir / "workspace"
        if run_dir.exists():
            _remove_existing_run_dir(self.runner, project_root, workspace, run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        commands: list[dict[str, Any]] = []
        isolation = _prepare_workspace(
            self.runner,
            project_root=project_root,
            workspace=workspace,
            commands=commands,
            mode=self.config.isolation_mode,
            ignore_names=set(self.config.ignore_names),
        )

        prompt = _self_patch_prompt(work=work, payload=payload)
        codex_cmd = _codex_command(self.config, workspace)
        codex_result = self.runner(
            codex_cmd,
            cwd=workspace,
            input_text=prompt,
            timeout_seconds=self.config.codex_timeout_seconds,
            idle_timeout_seconds=self.config.codex_idle_timeout_seconds,
            watchdog_root=workspace,
            watchdog_poll_seconds=self.config.watchdog_poll_seconds,
            watchdog_file_scan_seconds=self.config.watchdog_file_scan_seconds,
        )
        commands.append(_command_record("codex", codex_cmd, codex_result))
        _write_text(run_dir / "codex_stdout.txt", codex_result.stdout)
        _write_text(run_dir / "codex_stderr.txt", codex_result.stderr)
        if codex_result.returncode != 0:
            raise SelfPatchError(f"codex self-patch failed: {_trim(redact_text(codex_result.stderr or codex_result.stdout))}")

        test_result = self.runner(
            list(self.config.test_command),
            cwd=workspace,
            input_text=None,
            timeout_seconds=self.config.test_timeout_seconds,
            idle_timeout_seconds=self.config.test_idle_timeout_seconds,
            watchdog_root=workspace,
            watchdog_poll_seconds=self.config.watchdog_poll_seconds,
            watchdog_file_scan_seconds=self.config.watchdog_file_scan_seconds,
        )
        commands.append(_command_record("test", list(self.config.test_command), test_result))
        _write_text(run_dir / "test_stdout.txt", test_result.stdout)
        _write_text(run_dir / "test_stderr.txt", test_result.stderr)

        diff_check_result = self.runner(
            ["git", "diff", "--check"],
            cwd=workspace,
            input_text=None,
            timeout_seconds=30,
        )
        commands.append(_command_record("diff_check", ["git", "diff", "--check"], diff_check_result))
        _write_text(run_dir / "diff_check_stdout.txt", diff_check_result.stdout)
        _write_text(run_dir / "diff_check_stderr.txt", diff_check_result.stderr)

        diff_result = self.runner(
            ["git", "diff", "--no-ext-diff", "--binary"],
            cwd=workspace,
            input_text=None,
            timeout_seconds=30,
        )
        commands.append(_command_record("diff", ["git", "diff", "--no-ext-diff", "--binary"], diff_result))
        if diff_result.returncode != 0:
            raise SelfPatchError(f"git diff failed: {_trim(redact_text(diff_result.stderr or diff_result.stdout))}")

        patch_text = diff_result.stdout
        patch_path = run_dir / "proposal.patch"
        _write_text(patch_path, patch_text)
        changed_files = _changed_files(self.runner, workspace, commands)
        status = _result_status(
            has_patch=bool(patch_text.strip()),
            tests_passed=test_result.returncode == 0,
            diff_check_passed=diff_check_result.returncode == 0,
        )
        result = {
            "status": status,
            "job_id": job_id,
            "work_id": work.get("work_id"),
            "work_type": work.get("type"),
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "isolation": isolation,
            "patch_path": str(patch_path),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "changed_files": changed_files,
            "codex": _public_command_result(codex_result),
            "test": _public_command_result(test_result),
            "diff_check": _public_command_result(diff_check_result),
            "commands": commands,
            "next_required_action": _next_required_action(status),
            "created_at_epoch": time.time(),
        }
        _write_contract(run_dir, result)
        _write_evidence(run_dir, result, work=work, payload=payload)
        _write_summary_markdown(run_dir, result)
        _write_text(run_dir / "summary.json", _json_dump(result))
        return result


def build_self_patch_config_from_env(*, project_root: str | Path = ".") -> SelfPatchConfig:
    return SelfPatchConfig(
        project_root=Path(project_root),
        run_root=Path(os.environ.get("NEUROKERNEL_SELF_PATCH_RUN_ROOT", "artifacts/self_patch")),
        codex_bin=os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_BIN") or os.environ.get("NEUROKERNEL_CODEX_BIN", "codex"),
        codex_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_TIMEOUT", "420")),
        test_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_TIMEOUT", "300")),
        codex_idle_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_IDLE_TIMEOUT", "120")),
        test_idle_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_IDLE_TIMEOUT", "90")),
        watchdog_poll_seconds=float(os.environ.get("NEUROKERNEL_SELF_PATCH_WATCHDOG_POLL", "2.0")),
        watchdog_file_scan_seconds=float(os.environ.get("NEUROKERNEL_SELF_PATCH_WATCHDOG_FILE_SCAN", "10.0")),
        test_command=tuple(_split_command(os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_COMMAND", "python -m pytest -q"))),
        codex_model=os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_MODEL") or None,
        sandbox=os.environ.get("NEUROKERNEL_SELF_PATCH_SANDBOX", "workspace-write"),
        isolation_mode=os.environ.get("NEUROKERNEL_SELF_PATCH_ISOLATION", "auto"),
    )


def _prepare_workspace(
    runner: CommandRunner,
    *,
    project_root: Path,
    workspace: Path,
    commands: list[dict[str, Any]],
    mode: str,
    ignore_names: set[str],
) -> dict[str, Any]:
    normalized = mode.strip().lower()
    if normalized not in {"auto", "copy", "worktree"}:
        raise SelfPatchError(f"unknown self-patch isolation mode: {mode}")
    is_git_repo = _is_git_repo(runner, project_root, commands) if normalized in {"auto", "worktree"} else False
    if normalized in {"auto", "worktree"} and is_git_repo:
        result = runner(["git", "worktree", "add", "--detach", str(workspace), "HEAD"], cwd=project_root, input_text=None, timeout_seconds=120)
        commands.append(_command_record("git_worktree_add", ["git", "worktree", "add", "--detach", str(workspace), "HEAD"], result))
        if result.returncode == 0:
            return {"mode": "worktree", "source": str(project_root), "workspace": str(workspace)}
        raise SelfPatchError(f"git worktree add failed: {_trim(redact_text(result.stderr or result.stdout))}")
    if normalized == "worktree":
        raise SelfPatchError(f"project_root is not a git work tree: {project_root}")
    _copy_project(project_root, workspace, ignore_names=ignore_names)
    _git_init_workspace(runner, workspace, commands)
    return {"mode": "copy", "source": str(project_root), "workspace": str(workspace)}


def _is_git_repo(runner: CommandRunner, project_root: Path, commands: list[dict[str, Any]]) -> bool:
    result = runner(["git", "rev-parse", "--is-inside-work-tree"], cwd=project_root, input_text=None, timeout_seconds=30)
    commands.append(_command_record("git_repo_probe", ["git", "rev-parse", "--is-inside-work-tree"], result))
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _remove_existing_run_dir(runner: CommandRunner, project_root: Path, workspace: Path, run_dir: Path) -> None:
    if workspace.exists() and (workspace / ".git").exists():
        runner(["git", "worktree", "remove", "--force", str(workspace)], cwd=project_root, input_text=None, timeout_seconds=60)
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _copy_project(project_root: Path, workspace: Path, *, ignore_names: set[str]) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in ignore_names}
        ignored.update(name for name in names if name.endswith((".pyc", ".pyo", ".db", ".sqlite", ".pt", ".onnx", ".rknn", ".jsonl")))
        return ignored

    shutil.copytree(project_root, workspace, ignore=ignore)


def _git_init_workspace(runner: CommandRunner, workspace: Path, commands: list[dict[str, Any]]) -> None:
    for label, cmd in (
        ("git_init", ["git", "init"]),
        ("git_config_name", ["git", "config", "user.name", "NeuroKernel SelfPatch"]),
        ("git_config_email", ["git", "config", "user.email", "selfpatch@local.invalid"]),
        ("git_add", ["git", "add", "-A"]),
        ("git_commit", ["git", "commit", "-m", "baseline"]),
    ):
        result = runner(cmd, cwd=workspace, input_text=None, timeout_seconds=60)
        commands.append(_command_record(label, cmd, result))
        if result.returncode != 0:
            raise SelfPatchError(f"{label} failed: {_trim(redact_text(result.stderr or result.stdout))}")


def _codex_command(config: SelfPatchConfig, workspace: Path) -> list[str]:
    cmd = _codex_command_prefix(config.codex_bin) + ["--ask-for-approval", "never", "exec"]
    if config.codex_model:
        cmd.extend(["--model", config.codex_model])
    cmd.extend(
        [
            "--sandbox",
            config.sandbox,
            "--cd",
            str(workspace),
            "--skip-git-repo-check",
            "--ephemeral",
            "-",
        ]
    )
    return cmd


def _codex_command_prefix(codex_bin: str) -> list[str]:
    if os.name == "nt":
        return ["cmd.exe", "/d", "/c", codex_bin]
    return [codex_bin]


def _self_patch_prompt(*, work: dict[str, Any], payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the development worker for NeuroKernel AGI Seed.",
            "Implement only the approved self-patch work item in this isolated workspace.",
            "",
            "Non-negotiable constraints:",
            "- Do not read or print secrets, tokens, .env values, browser cookies, SSH keys, or private credentials.",
            "- Do not push, deploy, reboot, install system packages, or modify files outside this workspace.",
            "- Keep the change small and directly tied to the work item.",
            "- Add or update tests for the implemented behavior.",
            "- If the request cannot be implemented safely, write a clear failing note in a new docs/self_patch_blockers.md file and do not pretend success.",
            "- Do not add fallback behavior that claims capability without a real implementation.",
            "- If Queue payload JSON contains retry.previous_result, treat it as the failed prior attempt: inspect the listed changed files and failing test tails, then fix the underlying issue instead of repeating the same patch.",
            "",
            "Work item JSON:",
            _json_dump(work),
            "",
            "Queue payload JSON:",
            _json_dump(payload),
            "",
            "Expected output:",
            "- Modify the repository files in this workspace.",
            "- Leave tests runnable with python -m pytest -q.",
            "- Do not produce a final prose report; the harness will inspect git diff and test output.",
        ]
    )


def _changed_files(runner: CommandRunner, workspace: Path, commands: list[dict[str, Any]]) -> list[str]:
    cmd = ["git", "diff", "--name-only"]
    result = runner(cmd, cwd=workspace, input_text=None, timeout_seconds=30)
    commands.append(_command_record("changed_files", cmd, result))
    if result.returncode != 0:
        raise SelfPatchError(f"git diff --name-only failed: {_trim(redact_text(result.stderr or result.stdout))}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _result_status(*, has_patch: bool, tests_passed: bool, diff_check_passed: bool) -> str:
    if not has_patch:
        return "no_patch"
    if not diff_check_passed:
        return "diff_check_failed"
    if tests_passed:
        return "patch_ready"
    return "test_failed"


def _next_required_action(status: str) -> str:
    if status == "patch_ready":
        return "human_review_then_activation"
    if status == "test_failed":
        return "human_or_worker_review_failed_patch"
    if status == "diff_check_failed":
        return "worker_review_patch_format_errors"
    return "revise_work_item_or_prompt"


def _write_contract(run_dir: Path, result: dict[str, Any]) -> None:
    contract = {
        "schema_version": "neurokernel-self-patch-artifact-v2",
        "required_artifacts": [
            "proposal.patch",
            "summary.json",
            "evidence.json",
            "summary.md",
            "contract.json",
            "codex_stdout.txt",
            "codex_stderr.txt",
            "test_stdout.txt",
            "test_stderr.txt",
            "diff_check_stdout.txt",
            "diff_check_stderr.txt",
        ],
        "status": result.get("status"),
        "patch_path": result.get("patch_path"),
        "activation_ready": result.get("status") == "patch_ready",
    }
    _write_text(run_dir / "contract.json", _json_dump(contract))


def _write_evidence(run_dir: Path, result: dict[str, Any], *, work: dict[str, Any], payload: dict[str, Any]) -> None:
    evidence = {
        "schema_version": "neurokernel-self-patch-evidence-v1",
        "job_id": result.get("job_id"),
        "work_id": result.get("work_id"),
        "status": result.get("status"),
        "isolation": result.get("isolation"),
        "changed_files": result.get("changed_files"),
        "checks": {
            "codex_returncode": (result.get("codex") or {}).get("returncode"),
            "test_returncode": (result.get("test") or {}).get("returncode"),
            "diff_check_returncode": (result.get("diff_check") or {}).get("returncode"),
        },
        "work_title": work.get("title"),
        "work_goal": work.get("goal"),
        "queue_payload": payload,
        "commands": result.get("commands") or [],
    }
    _write_text(run_dir / "evidence.json", _json_dump(evidence))


def _write_summary_markdown(run_dir: Path, result: dict[str, Any]) -> None:
    changed = result.get("changed_files") or []
    lines = [
        "# Self-Patch Result",
        "",
        f"- status: `{result.get('status')}`",
        f"- job_id: `{result.get('job_id')}`",
        f"- work_id: `{result.get('work_id')}`",
        f"- isolation: `{(result.get('isolation') or {}).get('mode')}`",
        f"- patch_bytes: `{result.get('patch_bytes')}`",
        f"- next_required_action: `{result.get('next_required_action')}`",
        "",
        "## Changed Files",
        "",
    ]
    lines.extend(f"- `{item}`" for item in changed)
    if not changed:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- codex returncode: `{(result.get('codex') or {}).get('returncode')}`",
            f"- test returncode: `{(result.get('test') or {}).get('returncode')}`",
            f"- diff_check returncode: `{(result.get('diff_check') or {}).get('returncode')}`",
            "",
        ]
    )
    _write_text(run_dir / "summary.md", "\n".join(lines))


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    input_text: str | None = None,
    timeout_seconds: int,
    idle_timeout_seconds: int | None = None,
    watchdog_root: Path | None = None,
    watchdog_poll_seconds: float = 2.0,
    watchdog_file_scan_seconds: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdin": subprocess.PIPE if input_text is not None else None,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    if input_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(input_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    activity = {"last": time.monotonic()}
    stdout_thread = _pipe_reader(proc.stdout, stdout_chunks, activity) if proc.stdout is not None else None
    stderr_thread = _pipe_reader(proc.stderr, stderr_chunks, activity) if proc.stderr is not None else None

    started = time.monotonic()
    last_workspace_check = started
    last_workspace_mtime = _workspace_mtime(watchdog_root) if watchdog_root is not None else None
    stop_reason: str | None = None
    poll_seconds = max(0.1, watchdog_poll_seconds)

    while proc.poll() is None:
        now = time.monotonic()
        if timeout_seconds > 0 and now - started >= timeout_seconds:
            stop_reason = f"Command timed out after {timeout_seconds} seconds."
            break
        if (
            idle_timeout_seconds is not None
            and idle_timeout_seconds > 0
            and now - activity["last"] >= idle_timeout_seconds
        ):
            if watchdog_root is not None and now - last_workspace_check >= max(0.1, watchdog_file_scan_seconds):
                current_mtime = _workspace_mtime(watchdog_root)
                last_workspace_check = now
                if current_mtime is not None and current_mtime != last_workspace_mtime:
                    last_workspace_mtime = current_mtime
                    activity["last"] = now
                    continue
            stop_reason = f"Command stalled after {idle_timeout_seconds} seconds without output or workspace changes."
            break
        time.sleep(min(poll_seconds, 1.0))

    if stop_reason is not None:
        _terminate_process_group(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            proc.wait()
        _join_reader(stdout_thread)
        _join_reader(stderr_thread)
        code = 124 if "timed out" in stop_reason else 125
        return subprocess.CompletedProcess(cmd, code, "".join(stdout_chunks), "".join(stderr_chunks) + f"\n{stop_reason}")

    _join_reader(stdout_thread)
    _join_reader(stderr_thread)
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(stdout_chunks), "".join(stderr_chunks))


def _pipe_reader(pipe: Any, chunks: list[str], activity: dict[str, float]) -> threading.Thread:
    def read_loop() -> None:
        try:
            for chunk in iter(pipe.readline, ""):
                if not chunk:
                    break
                chunks.append(chunk)
                activity["last"] = time.monotonic()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    thread = threading.Thread(target=read_loop, daemon=True)
    thread.start()
    return thread


def _join_reader(thread: threading.Thread | None) -> None:
    if thread is not None:
        thread.join(timeout=2)


def _workspace_mtime(root: Path | None, *, max_files: int = 5_000) -> int | None:
    if root is None or not root.exists():
        return None
    ignored = {".git", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    newest = root.stat().st_mtime_ns
    seen = 0
    stack = [root]
    while stack and seen < max_files:
        current = stack.pop()
        try:
            for child in current.iterdir():
                if child.name in ignored:
                    continue
                seen += 1
                try:
                    stat = child.stat()
                except OSError:
                    continue
                newest = max(newest, stat.st_mtime_ns)
                if child.is_dir():
                    stack.append(child)
                if seen >= max_files:
                    break
        except OSError:
            continue
    return newest


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except Exception:
            proc.terminate()
            return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            proc.kill()
            return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _command_record(label: str, cmd: list[str], result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "label": label,
        "cmd": _redacted_cmd(cmd),
        "returncode": result.returncode,
        "stdout_tail": _trim(redact_text(result.stdout)),
        "stderr_tail": _trim(redact_text(result.stderr)),
    }


def _public_command_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_tail": _trim(redact_text(result.stdout)),
        "stderr_tail": _trim(redact_text(result.stderr)),
    }


def _redacted_cmd(cmd: list[str]) -> list[str]:
    redacted = []
    for item in cmd:
        lower = item.lower()
        if "token" in lower or "password" in lower or "secret" in lower:
            redacted.append("<redacted>")
        else:
            redacted.append(item)
    return redacted


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value).strip("_")
    if not safe:
        raise SelfPatchError("safe job name is empty")
    return safe[:120]


def _split_command(value: str) -> list[str]:
    parts = shlex.split(value, posix=os.name != "nt")
    if not parts:
        raise SelfPatchError("test command is empty")
    return parts


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_dump(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _trim(text: str, limit: int = 4_000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]
