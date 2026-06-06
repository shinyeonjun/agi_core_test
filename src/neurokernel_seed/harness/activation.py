from __future__ import annotations

import os
import shlex
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .memory import HarnessMemory
from .trace import redact_text


class ActivationError(RuntimeError):
    pass


class CommandRunner(Protocol):
    def __call__(self, cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class ActivationConfig:
    db_path: Path = Path("data/harness.db")
    project_root: Path = Path(".")
    self_patch_run_root: Path = Path("artifacts/self_patch")
    test_command: tuple[str, ...] = ("python", "-m", "pytest", "-q")
    test_timeout_seconds: int = 300
    reload_command: tuple[str, ...] = ()
    reload_timeout_seconds: int = 30
    require_clean_git: bool = True
    commit_after_apply: bool = True
    archive_root: Path = Path("artifacts/activations")


class ActivationService:
    def __init__(self, config: ActivationConfig, *, runner: CommandRunner | None = None):
        self.config = config
        self.runner = runner or _run_command

    def activate_work_item(self, work_id: str, *, actor: str = "activation") -> dict[str, Any]:
        project_root = self.config.project_root.resolve()
        _require_project_root(project_root)
        with HarnessMemory(self.config.db_path) as memory:
            work = memory.get_work_item(work_id)
            if str(work.get("status") or "") != "waiting_approval":
                raise ActivationError(f"work item is not waiting for activation approval: {work.get('status')}")
            proposal = memory.find_capability_proposal_by_work_id(work_id)
            patch_result = _latest_patch_ready_result(memory.work_events(work_id))

        patch_path = self._resolve_patch_path(patch_result)
        commands: list[dict[str, Any]] = []
        pre_sha = _git_output(self.runner, ["git", "rev-parse", "HEAD"], cwd=project_root, commands=commands)
        if self.config.require_clean_git:
            status = _git_output(self.runner, ["git", "status", "--porcelain"], cwd=project_root, commands=commands)
            if status.strip():
                raise ActivationError("live repository has uncommitted changes; activation requires a clean tree")

        try:
            _run_checked(self.runner, ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)], cwd=project_root, timeout_seconds=60, commands=commands)
            _run_checked(self.runner, ["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=project_root, timeout_seconds=60, commands=commands)
        except Exception as exc:
            with HarnessMemory(self.config.db_path) as memory:
                memory.add_work_event(work_id, "activation_failed", actor=actor, payload={"stage": "apply", "error": redact_text(str(exc))})
                _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload={"stage": "apply", "error": redact_text(str(exc))})
            raise

        test_result = self.runner(list(self.config.test_command), cwd=project_root, timeout_seconds=self.config.test_timeout_seconds)
        commands.append(_command_record("test", list(self.config.test_command), test_result))
        if test_result.returncode != 0:
            rollback = self.runner(["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], cwd=project_root, timeout_seconds=60)
            commands.append(_command_record("rollback", ["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], rollback))
            with HarnessMemory(self.config.db_path) as memory:
                payload = {"stage": "test", "test": _public_result(test_result), "rollback": _public_result(rollback), "commands": commands}
                memory.add_work_event(work_id, "activation_failed", actor=actor, payload=payload)
                _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload=payload)
            raise ActivationError("activation tests failed; patch was rolled back")

        commit_result = None
        if self.config.commit_after_apply:
            try:
                commit_result = self._commit_activation(project_root, work_id, commands)
            except Exception as exc:
                rollback = self.runner(["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], cwd=project_root, timeout_seconds=60)
                commands.append(_command_record("rollback", ["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], rollback))
                with HarnessMemory(self.config.db_path) as memory:
                    payload = {"stage": "commit", "error": redact_text(str(exc)), "rollback": _public_result(rollback), "commands": commands}
                    memory.add_work_event(work_id, "activation_failed", actor=actor, payload=payload)
                    _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload=payload)
                raise ActivationError("activation commit failed; patch was rolled back") from exc

        reload_result = None
        if self.config.reload_command:
            reload_result = self.runner(list(self.config.reload_command), cwd=project_root, timeout_seconds=self.config.reload_timeout_seconds)
            commands.append(_command_record("reload", list(self.config.reload_command), reload_result))
            if reload_result.returncode != 0:
                with HarnessMemory(self.config.db_path) as memory:
                    payload = {"stage": "reload", "reload": _public_result(reload_result), "commands": commands}
                    memory.add_work_event(work_id, "activation_reload_failed", actor=actor, payload=payload)
                    _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload=payload)
                raise ActivationError("activation reload command failed")

        post_sha = _git_output(self.runner, ["git", "rev-parse", "HEAD"], cwd=project_root, commands=commands)
        result = {
            "activated": True,
            "work_id": work_id,
            "proposal_id": proposal.get("proposal_id") if proposal else None,
            "action_id": proposal.get("action_id") if proposal else None,
            "patch_path": str(patch_path),
            "pre_activation_sha": pre_sha.strip(),
            "post_activation_sha": post_sha.strip(),
            "tests": _public_result(test_result),
            "commit": _public_result(commit_result) if commit_result else None,
            "reload": _public_result(reload_result) if reload_result else None,
            "service_reload_required": reload_result is None,
            "commands": commands,
        }
        archive_path = _write_activation_archive(project_root, self.config.archive_root, work_id, result, patch_path)
        result["archive_path"] = str(archive_path)
        with HarnessMemory(self.config.db_path) as memory:
            memory.add_work_event(work_id, "activated", actor=actor, payload=result)
            _transition_if_possible(memory, work_id, "completed", actor=actor, payload=result)
            if proposal:
                try:
                    memory.transition_capability_proposal(str(proposal["proposal_id"]), "active", actor=actor, payload=result)
                except ValueError:
                    memory.add_proposal_event(str(proposal["proposal_id"]), "activation_status_not_changed", actor=actor, payload=result)
                    memory.conn.commit()
        return result

    def _commit_activation(self, project_root: Path, work_id: str, commands: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
        _run_checked(self.runner, ["git", "add", "-A"], cwd=project_root, timeout_seconds=60, commands=commands)
        commit_message = f"Activate self-patch {work_id}"
        result = self.runner(["git", "commit", "-m", commit_message], cwd=project_root, timeout_seconds=60)
        commands.append(_command_record("git commit", ["git", "commit", "-m", commit_message], result))
        if result.returncode != 0:
            unstage = self.runner(["git", "reset", "--mixed", "HEAD"], cwd=project_root, timeout_seconds=60)
            commands.append(_command_record("git reset", ["git", "reset", "--mixed", "HEAD"], unstage))
            raise ActivationError(_trim(redact_text(result.stderr or result.stdout)))
        return result

    def _resolve_patch_path(self, patch_result: dict[str, Any]) -> Path:
        raw_patch = str(patch_result.get("patch_path") or "").strip()
        if not raw_patch:
            raise ActivationError("patch_path missing from self-patch result")
        patch_path = Path(raw_patch)
        if not patch_path.is_absolute():
            patch_path = self.config.project_root / patch_path
        patch_path = patch_path.resolve()
        run_root = _resolve_child(self.config.project_root, self.config.self_patch_run_root)
        try:
            patch_path.relative_to(run_root)
        except ValueError as exc:
            raise ActivationError(f"patch path is outside self-patch run root: {patch_path}") from exc
        if patch_path.name != "proposal.patch":
            raise ActivationError(f"unexpected patch file name: {patch_path.name}")
        if not patch_path.is_file():
            raise ActivationError(f"patch file not found: {patch_path}")
        return patch_path


def build_activation_config_from_env(*, db_path: str | Path = "data/harness.db", project_root: str | Path = ".") -> ActivationConfig:
    reload_command = tuple(_split_command(os.environ.get("NEUROKERNEL_ACTIVATION_RELOAD_COMMAND", "")))
    return ActivationConfig(
        db_path=Path(db_path),
        project_root=Path(project_root),
        self_patch_run_root=Path(os.environ.get("NEUROKERNEL_SELF_PATCH_RUN_ROOT", "artifacts/self_patch")),
        test_command=tuple(_split_command(os.environ.get("NEUROKERNEL_ACTIVATION_TEST_COMMAND", os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_COMMAND", "python -m pytest -q")))),
        test_timeout_seconds=int(os.environ.get("NEUROKERNEL_ACTIVATION_TEST_TIMEOUT", os.environ.get("NEUROKERNEL_SELF_PATCH_TEST_TIMEOUT", "300"))),
        reload_command=reload_command,
        reload_timeout_seconds=int(os.environ.get("NEUROKERNEL_ACTIVATION_RELOAD_TIMEOUT", "30")),
        require_clean_git=os.environ.get("NEUROKERNEL_ACTIVATION_REQUIRE_CLEAN_GIT", "1").lower() in {"1", "true", "yes", "y"},
        commit_after_apply=os.environ.get("NEUROKERNEL_ACTIVATION_COMMIT", "1").lower() in {"1", "true", "yes", "y"},
        archive_root=Path(os.environ.get("NEUROKERNEL_ACTIVATION_ARCHIVE_ROOT", "artifacts/activations")),
    )


def _latest_patch_ready_result(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event_type") != "job_completed":
            continue
        payload = event.get("payload_json") if isinstance(event.get("payload_json"), dict) else {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if result.get("status") == "patch_ready":
            return result
    raise ActivationError("no patch-ready self-patch result found")


def _require_project_root(project_root: Path) -> None:
    if not project_root.exists():
        raise ActivationError(f"project root does not exist: {project_root}")
    if not (project_root / ".git").exists():
        raise ActivationError("activation requires a git repository")


def _resolve_child(root: Path, child: Path) -> Path:
    if child.is_absolute():
        return child.resolve()
    return (root.resolve() / child).resolve()


def _git_output(runner: CommandRunner, cmd: list[str], *, cwd: Path, commands: list[dict[str, Any]]) -> str:
    result = runner(cmd, cwd=cwd, timeout_seconds=30)
    commands.append(_command_record(" ".join(cmd[:2]), cmd, result))
    if result.returncode != 0:
        raise ActivationError(_trim(redact_text(result.stderr or result.stdout)))
    return result.stdout


def _run_checked(runner: CommandRunner, cmd: list[str], *, cwd: Path, timeout_seconds: int, commands: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    result = runner(cmd, cwd=cwd, timeout_seconds=timeout_seconds)
    commands.append(_command_record(" ".join(cmd[:2]), cmd, result))
    if result.returncode != 0:
        raise ActivationError(_trim(redact_text(result.stderr or result.stdout)))
    return result


def _run_command(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _write_activation_archive(project_root: Path, archive_root: Path, work_id: str, result: dict[str, Any], patch_path: Path) -> Path:
    root = _resolve_child(project_root, archive_root)
    target = root / work_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "activation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (target / "proposal.patch").write_text(patch_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    return target / "activation.json"


def _transition_if_possible(memory: HarnessMemory, work_id: str, next_status: str, *, actor: str, payload: dict[str, Any]) -> None:
    try:
        memory.transition_work_item(work_id, next_status, actor=actor, payload=payload)
    except ValueError:
        memory.add_work_event(work_id, "activation_transition_skipped", actor=actor, payload={"next_status": next_status, "payload": payload})
        memory.conn.commit()


def _command_record(label: str, cmd: list[str], result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "label": label,
        "cmd": _redacted_cmd(cmd),
        "returncode": result.returncode,
        "stdout_tail": _trim(redact_text(result.stdout)),
        "stderr_tail": _trim(redact_text(result.stderr)),
    }


def _public_result(result: subprocess.CompletedProcess[str] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "returncode": result.returncode,
        "stdout_tail": _trim(redact_text(result.stdout)),
        "stderr_tail": _trim(redact_text(result.stderr)),
    }


def _redacted_cmd(cmd: list[str]) -> list[str]:
    redacted = []
    for part in cmd:
        lower = part.lower()
        redacted.append("<redacted>" if "token" in lower or "password" in lower or "secret" in lower else part)
    return redacted


def _split_command(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    return shlex.split(value, posix=os.name != "nt")


def _trim(text: str, limit: int = 2_000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[-limit:]
