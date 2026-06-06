from __future__ import annotations

import os
import shlex
import subprocess
import json
import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .action_catalog import ActionDefinition
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
    verify_activation: bool = True
    action_registry_path: Path | None = None
    smoke_timeout_seconds: int = 30


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
        preflight = _preflight_patch_result(patch_result)
        if not preflight.get("passed"):
            with HarnessMemory(self.config.db_path) as memory:
                memory.add_work_event(work_id, "activation_failed", actor=actor, payload={"stage": "preflight", "preflight": preflight})
                _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload={"stage": "preflight", "preflight": preflight})
            raise ActivationError(f"activation preflight failed: {preflight.get('reason')}")
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

        verification = self._verify_activation(project_root, proposal)
        commands.extend(verification.pop("commands", []))
        if not verification.get("passed"):
            rollback = self.runner(["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], cwd=project_root, timeout_seconds=60)
            commands.append(_command_record("rollback", ["git", "apply", "-R", "--whitespace=nowarn", str(patch_path)], rollback))
            with HarnessMemory(self.config.db_path) as memory:
                payload = {"stage": "verify", "verification": verification, "rollback": _public_result(rollback), "commands": commands}
                memory.add_work_event(work_id, "activation_failed", actor=actor, payload=payload)
                _transition_if_possible(memory, work_id, "reviewing", actor=actor, payload=payload)
            raise ActivationError("activation verification failed; patch was rolled back")

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
            "verification": verification,
            "commit": _public_result(commit_result) if commit_result else None,
            "reload": _public_result(reload_result) if reload_result else None,
            "service_reload_required": reload_result is None,
            "commands": commands,
        }
        archive_path = _write_activation_archive(project_root, self.config.archive_root, work_id, result, patch_path)
        result["archive_path"] = str(archive_path)
        with HarnessMemory(self.config.db_path) as memory:
            memory.add_work_event(work_id, "activation_verified", actor=actor, payload=verification)
            memory.add_work_event(work_id, "activated", actor=actor, payload=result)
            _transition_if_possible(memory, work_id, "completed", actor=actor, payload=result)
            if proposal:
                try:
                    memory.transition_capability_proposal(str(proposal["proposal_id"]), "active", actor=actor, payload=result)
                except ValueError:
                    memory.add_proposal_event(str(proposal["proposal_id"]), "activation_status_not_changed", actor=actor, payload=result)
                    memory.conn.commit()
        return result

    def _verify_activation(self, project_root: Path, proposal: dict[str, Any] | None) -> dict[str, Any]:
        if not self.config.verify_activation:
            return {"passed": True, "mode": "disabled", "commands": []}
        if not proposal:
            return {"passed": True, "mode": "no_proposal", "commands": []}
        action_id = str(proposal.get("action_id") or "").strip()
        if not action_id:
            return {"passed": False, "mode": "missing_action_id", "reason": "proposal has no action_id", "commands": []}
        try:
            registry_path = self.config.action_registry_path
            if registry_path is not None and not registry_path.is_absolute():
                registry_path = project_root / registry_path
            catalog = _fresh_action_catalog(registry_path=registry_path)
            action = catalog[action_id]
        except Exception as exc:
            return {"passed": False, "mode": "catalog_load_failed", "action_id": action_id, "reason": redact_text(str(exc)), "commands": []}
        spec_check = _activation_action_spec_check(action)
        if not spec_check.get("passed"):
            return {"passed": False, "mode": "action_schema_check", "action_id": action_id, "schema_check": spec_check, "commands": []}
        smoke = _smoke_action(project_root, self.config.db_path, action, timeout_seconds=self.config.smoke_timeout_seconds)
        output_check = _output_schema_smoke_check(smoke, action.outputs_schema)
        return {
            "passed": bool(smoke.get("passed")) and bool(output_check.get("passed")),
            "mode": "action_smoke",
            "action_id": action_id,
            "action_source": action.source,
            "action_version": action.version,
            "action_status": action.status,
            "executor": action.executor,
            "schema_check": spec_check,
            "smoke": smoke,
            "output_check": output_check,
            "commands": [],
        }

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
        verify_activation=os.environ.get("NEUROKERNEL_ACTIVATION_VERIFY", "1").lower() in {"1", "true", "yes", "y"},
        action_registry_path=Path(os.environ["NEUROKERNEL_ACTION_REGISTRY"]) if os.environ.get("NEUROKERNEL_ACTION_REGISTRY") else None,
        smoke_timeout_seconds=int(os.environ.get("NEUROKERNEL_ACTIVATION_SMOKE_TIMEOUT", "30")),
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


def _preflight_patch_result(patch_result: dict[str, Any]) -> dict[str, Any]:
    changed_files = patch_result.get("changed_files")
    if changed_files is None:
        return {"passed": True, "mode": "no_changed_files_declared"}
    if not isinstance(changed_files, list) or not all(isinstance(item, str) for item in changed_files):
        return {"passed": False, "reason": "changed_files must be a string list"}
    invalid = []
    blocked = []
    for path in changed_files:
        normalized = path.replace("\\", "/").strip()
        parts = [part for part in normalized.split("/") if part]
        if not normalized or normalized.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
            invalid.append(path)
            continue
        if _is_sensitive_patch_path(normalized):
            blocked.append(path)
    if invalid:
        return {"passed": False, "reason": "changed_files contains invalid paths", "invalid_paths": invalid}
    if blocked:
        return {"passed": False, "reason": "patch touches blocked paths", "blocked_paths": blocked}
    return {"passed": True, "changed_files": changed_files}


def _is_sensitive_patch_path(path: str) -> bool:
    if path == ".env" or (path.startswith(".env.") and path != ".env.example"):
        return True
    return path.startswith(("secrets/", "data/private/"))


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


def _fresh_action_catalog(*, registry_path: Path | None) -> dict[str, ActionDefinition]:
    import neurokernel_seed.harness.action_catalog as action_catalog_module

    importlib.invalidate_caches()
    action_catalog_module = importlib.reload(action_catalog_module)
    return action_catalog_module.build_action_catalog(registry_path=registry_path)


def _smoke_action(project_root: Path, db_path: Path, action: ActionDefinition, *, timeout_seconds: int) -> dict[str, Any]:
    import neurokernel_seed.harness.executors.readonly_command as readonly_command_module
    import neurokernel_seed.harness.executors.readonly_system as readonly_system_module

    importlib.invalidate_caches()
    readonly_command_module = importlib.reload(readonly_command_module)
    readonly_system_module = importlib.reload(readonly_system_module)

    params = _default_params(action)
    if params is None:
        return {"passed": False, "reason": "action has required params without defaults", "action_id": action.action_id}
    if action.executor == "readonly_command":
        configured_timeout = int(action.executor_config.get("timeout_seconds", timeout_seconds))
        smoke_action = replace(action, executor_config={**action.executor_config, "timeout_seconds": min(configured_timeout, timeout_seconds)})
        result = readonly_command_module.ReadOnlyCommandExecutor(project_root=project_root).execute(smoke_action, params, {"source": "activation_smoke"}).as_dict()
        return {"passed": bool(result.get("success")), "execution_result": result}
    if action.executor == "readonly_system":
        result = readonly_system_module.ReadOnlyExecutor(project_root=project_root, memory_path=db_path).execute(action.action_id, params, {"source": "activation_smoke"}).as_dict()
        return {"passed": bool(result.get("success")), "execution_result": result}
    return {"passed": False, "reason": f"executor is not smoke-testable: {action.executor}", "action_id": action.action_id}


def _activation_action_spec_check(action: ActionDefinition) -> dict[str, Any]:
    if action.status != "active":
        return {"passed": False, "reason": f"action status must be active before execution: {action.status}"}
    if not action.version:
        return {"passed": False, "reason": "action version is required"}
    if not action.description:
        return {"passed": False, "reason": "action description is required"}
    if not action.test_plan:
        return {"passed": False, "reason": "action test_plan is required"}
    if action.executor not in {"readonly_system", "readonly_command", "benchmark"}:
        return {"passed": False, "reason": f"executor is not allowed: {action.executor}"}
    if action.risk_level not in {"none", "low", "medium"}:
        return {"passed": False, "reason": f"risk_level is not activation-safe: {action.risk_level}"}
    if action.risk_level == "medium" and not action.requires_approval:
        return {"passed": False, "reason": "medium risk action must require approval"}
    if action.side_effect and not action.requires_approval:
        return {"passed": False, "reason": "side-effect action must require approval"}
    return {"passed": True}


def _output_schema_smoke_check(smoke: dict[str, Any], outputs_schema: dict[str, Any]) -> dict[str, Any]:
    if not outputs_schema:
        return {"passed": True, "mode": "no_outputs_schema"}
    result = smoke.get("execution_result")
    if not isinstance(result, dict):
        return {"passed": False, "reason": "smoke execution_result is missing"}
    payload = result.get("result")
    if not isinstance(payload, dict):
        return {"passed": False, "reason": "executor result must be an object"}
    required = outputs_schema.get("required") if isinstance(outputs_schema.get("required"), list) else []
    missing = [str(key) for key in required if str(key) not in payload]
    if missing:
        return {"passed": False, "reason": "executor result missing required output fields", "missing": missing}
    properties = outputs_schema.get("properties") if isinstance(outputs_schema.get("properties"), dict) else {}
    type_errors = []
    for key, schema in properties.items():
        if key not in payload or not isinstance(schema, dict) or "type" not in schema:
            continue
        expected_type = str(schema["type"])
        if not _matches_json_schema_type(payload[key], expected_type):
            type_errors.append({"field": key, "expected": expected_type, "actual": type(payload[key]).__name__})
    if type_errors:
        return {"passed": False, "reason": "executor result output field type mismatch", "type_errors": type_errors}
    return {"passed": True, "required": required, "checked_properties": sorted(properties)}


def _matches_json_schema_type(value: Any, expected_type: str) -> bool:
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "null":
        return value is None
    return False


def _default_params(action: ActionDefinition) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    required = set()
    if isinstance(action.params_schema.get("required"), list):
        required.update(str(item) for item in action.params_schema.get("required", []))
    properties = action.params_schema.get("properties") if isinstance(action.params_schema.get("properties"), dict) else action.params_schema
    if not isinstance(properties, dict):
        return {}
    for key, schema in properties.items():
        if isinstance(schema, dict) and "default" in schema:
            params[str(key)] = schema["default"]
        elif str(key) in required:
            return None
    return params


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
