from __future__ import annotations

import os
import shlex
import shutil
import subprocess
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
    ) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class SelfPatchConfig:
    project_root: Path = Path(".")
    run_root: Path = Path("artifacts/self_patch")
    codex_bin: str = "codex"
    codex_timeout_seconds: int = 900
    test_timeout_seconds: int = 300
    test_command: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    codex_model: str | None = None
    sandbox: str = "workspace-write"
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

        run_dir = (self.config.run_root / _safe_name(job_id)).resolve()
        workspace = run_dir / "workspace"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        _copy_project(project_root, workspace, ignore_names=set(self.config.ignore_names))

        commands: list[dict[str, Any]] = []
        _git_init_workspace(self.runner, workspace, commands)

        prompt = _self_patch_prompt(work=work, payload=payload)
        codex_cmd = _codex_command(self.config, workspace)
        codex_result = self.runner(
            codex_cmd,
            cwd=workspace,
            input_text=prompt,
            timeout_seconds=self.config.codex_timeout_seconds,
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
        )
        commands.append(_command_record("test", list(self.config.test_command), test_result))
        _write_text(run_dir / "test_stdout.txt", test_result.stdout)
        _write_text(run_dir / "test_stderr.txt", test_result.stderr)

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
        status = _result_status(has_patch=bool(patch_text.strip()), tests_passed=test_result.returncode == 0)
        result = {
            "status": status,
            "job_id": job_id,
            "work_id": work.get("work_id"),
            "work_type": work.get("type"),
            "run_dir": str(run_dir),
            "workspace": str(workspace),
            "patch_path": str(patch_path),
            "patch_bytes": len(patch_text.encode("utf-8")),
            "changed_files": changed_files,
            "codex": _public_command_result(codex_result),
            "test": _public_command_result(test_result),
            "commands": commands,
            "next_required_action": _next_required_action(status),
            "created_at_epoch": time.time(),
        }
        _write_text(run_dir / "summary.json", _json_dump(result))
        return result


def build_self_patch_config_from_env(*, project_root: str | Path = ".") -> SelfPatchConfig:
    return SelfPatchConfig(
        project_root=Path(project_root),
        run_root=Path(os.environ.get("NEUROKERNEL_SELF_PATCH_RUN_ROOT", "artifacts/self_patch")),
        codex_bin=os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_BIN") or os.environ.get("NEUROKERNEL_CODEX_BIN", "codex"),
        codex_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_TIMEOUT", "900")),
        test_timeout_seconds=int(os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_TIMEOUT", "300")),
        test_command=tuple(_split_command(os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_COMMAND", "python -m pytest -q"))),
        codex_model=os.environ.get("NEUROKERNEL_SELF_PATCH_CODEX_MODEL") or None,
        sandbox=os.environ.get("NEUROKERNEL_SELF_PATCH_SANDBOX", "workspace-write"),
    )


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


def _result_status(*, has_patch: bool, tests_passed: bool) -> str:
    if not has_patch:
        return "no_patch"
    if tests_passed:
        return "patch_ready"
    return "test_failed"


def _next_required_action(status: str) -> str:
    if status == "patch_ready":
        return "human_review_then_activation"
    if status == "test_failed":
        return "human_or_worker_review_failed_patch"
    return "revise_work_item_or_prompt"


def _run_command(cmd: list[str], *, cwd: Path, input_text: str | None = None, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


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
