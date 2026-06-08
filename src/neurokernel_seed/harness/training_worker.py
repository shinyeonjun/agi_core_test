from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .trace import redact_text


ALLOWED_NK_ACTIONS = {
    "runtime-pipeline",
    "runtime-auto",
    "runtime-cycle",
    "runtime-compare",
    "current-bench-all",
    "deploy-all-best",
}


class TrainingWorkerError(RuntimeError):
    pass


def normalize_training_command(command_spec: dict[str, Any]) -> tuple[str, list[str]]:
    action = str(command_spec.get("action") or "").strip()
    args = command_spec.get("args") if isinstance(command_spec.get("args"), list) else []
    if action not in ALLOWED_NK_ACTIONS:
        raise TrainingWorkerError(f"nk action is not allowed for training worker: {action}")
    return action, [str(value) for value in args]


def build_training_command(command_spec: dict[str, Any], *, python_executable: str | None = None) -> list[str]:
    action, args = normalize_training_command(command_spec)
    command = [python_executable or sys.executable, "-m", "neurokernel_seed.nk_cli", action, *args]
    if "--json" not in args:
        command.append("--json")
    return command


class TrainingCommandRunner(Protocol):
    def __call__(self, cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
        ...


@dataclass(frozen=True)
class TrainingWorkerConfig:
    project_root: Path = Path(".")
    run_root: Path = Path("artifacts/training_jobs")
    enabled: bool = False
    timeout_seconds: int = 7200


class TrainingPipelineWorker:
    def __init__(self, config: TrainingWorkerConfig, *, runner: TrainingCommandRunner | None = None):
        self.config = config
        self.runner = runner or _run_training_command

    def run(self, *, job_id: str, work: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        metadata = work.get("metadata_json") if isinstance(work.get("metadata_json"), dict) else {}
        command_spec = metadata.get("nk_command") if isinstance(metadata.get("nk_command"), dict) else {}
        command = build_training_command(command_spec)

        project_root = self.config.project_root.resolve()
        run_dir = (project_root / self.config.run_root / _safe_name(job_id)).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "work.json", {"work": work, "payload": payload, "command": command})

        if not self.config.enabled:
            result = {
                "status": "waiting_for_training_node",
                "job_id": job_id,
                "work_id": work.get("work_id"),
                "work_type": work.get("type"),
                "run_dir": str(run_dir),
                "command": command,
                "reason": "training worker is disabled on this node",
                "next_required_action": "run a training worker on the laptop with NEUROKERNEL_TRAINING_WORKER_ENABLED=true and queue training_pipeline",
                "created_at_epoch": time.time(),
            }
            _write_json(run_dir / "summary.json", result)
            return result

        completed = self.runner(command, cwd=project_root, timeout_seconds=self.config.timeout_seconds)
        stdout = redact_text(completed.stdout or "", max_chars=20_000)
        stderr = redact_text(completed.stderr or "", max_chars=20_000)
        _write_text(run_dir / "stdout.txt", stdout)
        _write_text(run_dir / "stderr.txt", stderr)
        parsed = _parse_json(stdout)
        result = {
            "status": "training_completed" if completed.returncode == 0 else "training_failed",
            "job_id": job_id,
            "work_id": work.get("work_id"),
            "work_type": work.get("type"),
            "run_dir": str(run_dir),
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": stdout[-4_000:],
            "stderr_tail": stderr[-4_000:],
            "nk_result": parsed,
            "next_required_action": "inspect_training_report" if completed.returncode != 0 else "review_benchmark_and_deployment_report",
            "created_at_epoch": time.time(),
        }
        _write_json(run_dir / "summary.json", result)
        return result


def build_training_worker_config_from_env(*, project_root: str | Path = ".") -> TrainingWorkerConfig:
    return TrainingWorkerConfig(
        project_root=Path(project_root),
        run_root=Path(os.environ.get("NEUROKERNEL_TRAINING_WORK_RUN_ROOT", "artifacts/training_jobs")),
        enabled=os.environ.get("NEUROKERNEL_TRAINING_WORKER_ENABLED", "").lower() in {"1", "true", "yes", "y"},
        timeout_seconds=int(os.environ.get("NEUROKERNEL_TRAINING_WORKER_TIMEOUT", "7200")),
    )


def _run_training_command(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )


def _parse_json(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value).strip("._")
    return safe or "training_job"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
