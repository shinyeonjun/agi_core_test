from __future__ import annotations

import json
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.model.pipeline import TrainingPipelineConfig, run_training_pipeline


MODEL_RELEASE_SCHEMA_VERSION = "neurokernel-model-release-v1"
TRAIN_DEPLOY_SCHEMA_VERSION = "neurokernel-train-deploy-model-v1"
SAFE_RELEASE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class ModelReleaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainDeployModelConfig:
    features: str | Path | None = None
    out_dir: str | Path | None = None
    run_name: str | None = None
    remote_host: str | None = None
    remote_project: str | None = None
    activate: bool = False
    dry_run: bool = False
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    hidden_layers: int = 2
    device: str = "auto"
    patience: int = 10
    split: str = "test"
    verify_onnx: bool = True
    dynamic_batch: bool = True
    run_gate_ablation: bool = True
    gate_ablation_episodes: int = 50
    trace_episodes: int = 10
    max_failures_per_env: int = 10
    strict: bool = True
    allow_gate_failure: bool = False
    allow_benchmark_failure: bool = False
    overwrite_local_run: bool = False
    ssh_connect_timeout: int = 10
    ssh_server_alive_interval: int = 5
    ssh_server_alive_count_max: int = 2


@dataclass(frozen=True)
class ResolvedTrainDeployConfig:
    features: Path
    out_dir: Path
    run_name: str
    remote_host: str
    remote_project: str


def train_deploy_model(config: TrainDeployModelConfig) -> dict[str, Any]:
    resolved = _resolve_train_deploy_config(config)
    preflight = _preflight_model_release(resolved, config)
    if config.dry_run:
        return {
            "schema_version": TRAIN_DEPLOY_SCHEMA_VERSION,
            "status": "dry_run",
            "created_at": _utc_now(),
            "run_name": resolved.run_name,
            "preflight": preflight,
        }
    training = run_training_pipeline(
        TrainingPipelineConfig(
            features=resolved.features,
            out_dir=resolved.out_dir,
            run_name=resolved.run_name,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            weight_decay=config.weight_decay,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            device=config.device,
            patience=config.patience,
            split=config.split,
            verify_onnx=config.verify_onnx,
            dynamic_batch=config.dynamic_batch,
            run_gate_ablation=config.run_gate_ablation,
            gate_ablation_episodes=config.gate_ablation_episodes,
            trace_episodes=config.trace_episodes,
            max_failures_per_env=config.max_failures_per_env,
            strict=config.strict,
            allow_gate_failure=config.allow_gate_failure,
            allow_benchmark_failure=config.allow_benchmark_failure,
            overwrite=config.overwrite_local_run,
        )
    )
    release = deploy_training_run(
        Path(training["run_dir"]),
        release_name=resolved.run_name,
        remote_host=resolved.remote_host,
        remote_project=resolved.remote_project,
        activate=config.activate,
        ssh_connect_timeout=config.ssh_connect_timeout,
        ssh_server_alive_interval=config.ssh_server_alive_interval,
        ssh_server_alive_count_max=config.ssh_server_alive_count_max,
    )
    return {
        "schema_version": TRAIN_DEPLOY_SCHEMA_VERSION,
        "status": "completed",
        "created_at": _utc_now(),
        "run_name": resolved.run_name,
        "preflight": preflight,
        "training": training,
        "release": release,
    }


def preflight_model_release(config: TrainDeployModelConfig) -> dict[str, Any]:
    return _preflight_model_release(_resolve_train_deploy_config(config), config)


def deploy_training_run(
    run_dir: str | Path,
    *,
    release_name: str,
    remote_host: str,
    remote_project: str,
    activate: bool = False,
    ssh_connect_timeout: int = 10,
    ssh_server_alive_interval: int = 5,
    ssh_server_alive_count_max: int = 2,
) -> dict[str, Any]:
    _assert_safe_release_name(release_name)
    source_dir = Path(run_dir)
    _require_release_artifacts(source_dir)
    remote_project = _normalize_remote_project(remote_project)
    remote_releases_dir = f"{remote_project.rstrip('/')}/artifacts/model_releases"
    remote_release_dir = f"{remote_releases_dir}/{release_name}"
    remote_incoming_dir = f"{remote_releases_dir}/.incoming"
    remote_stage_dir = f"{remote_incoming_dir}/{release_name}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    release_manifest = _build_release_manifest(
        source_dir,
        release_name=release_name,
        remote_host=remote_host,
        remote_project=remote_project,
        remote_release_dir=remote_release_dir,
        remote_stage_dir=remote_stage_dir,
        activate=activate,
    )
    release_manifest_path = source_dir / "model_release_manifest.json"
    release_manifest_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    ssh_options = _ssh_options(
        connect_timeout=ssh_connect_timeout,
        server_alive_interval=ssh_server_alive_interval,
        server_alive_count_max=ssh_server_alive_count_max,
    )
    _remote_assert_release_absent(remote_host, remote_release_dir, ssh_options)
    stage_created = False
    _run_native(
        [
            "ssh",
            *ssh_options,
            remote_host,
            (
                "set -eu; "
                f"mkdir -p {_sh_quote(remote_incoming_dir)}; "
                f"if [ -e {_sh_quote(remote_stage_dir)} ]; then "
                f"echo {_sh_quote('model release stage already exists: ' + remote_stage_dir)} >&2; exit 18; "
                "fi; "
                f"mkdir {_sh_quote(remote_stage_dir)}"
            ),
        ]
    )
    stage_created = True
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / f"{release_name}.tar.gz"
            _package_run_dir(source_dir, package_path)
            remote_package = f"{remote_host}:{remote_stage_dir}/payload.tar.gz"
            _run_native(["scp", *ssh_options, str(package_path), remote_package])
        _run_native(
            [
                "ssh",
                *ssh_options,
                remote_host,
                (
                    "set -eu; "
                    f"tar -xzf {_sh_quote(remote_stage_dir + '/payload.tar.gz')} -C {_sh_quote(remote_stage_dir)}; "
                    f"rm {_sh_quote(remote_stage_dir + '/payload.tar.gz')}; "
                    f"test -f {_sh_quote(remote_stage_dir + '/world_model.onnx')}; "
                    f"test -f {_sh_quote(remote_stage_dir + '/world_model.manifest.json')}; "
                    f"test -f {_sh_quote(remote_stage_dir + '/model_release_manifest.json')}; "
                    f"if [ -e {_sh_quote(remote_release_dir)} ]; then "
                    f"echo {_sh_quote('model release already exists: ' + remote_release_dir)} >&2; exit 17; "
                    "fi; "
                    f"mv {_sh_quote(remote_stage_dir)} {_sh_quote(remote_release_dir)}"
                ),
            ]
        )
        stage_created = False
    except Exception:
        if stage_created:
            _cleanup_remote_stage(remote_host, remote_stage_dir, ssh_options)
        raise
    release_manifest["status"] = "deployed"
    release_manifest["deployed_at"] = _utc_now()
    release_manifest_path.write_text(json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _run_native(["scp", *ssh_options, str(release_manifest_path), f"{remote_host}:{remote_release_dir}/model_release_manifest.json"])
    if activate:
        _activate_remote_release(remote_host, remote_project, remote_releases_dir, remote_release_dir, ssh_options)
    return release_manifest


def _require_release_artifacts(run_dir: Path) -> None:
    required = [
        "training_pipeline_manifest.json",
        "world_model.pt",
        "world_model.onnx",
        "world_model.manifest.json",
        "train_result.json",
        "onnx_export.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise ModelReleaseError(f"training run is not releasable; missing: {', '.join(missing)}")


def _build_release_manifest(
    run_dir: Path,
    *,
    release_name: str,
    remote_host: str,
    remote_project: str,
    remote_release_dir: str,
    remote_stage_dir: str,
    activate: bool,
) -> dict[str, Any]:
    files = sorted(str(path.relative_to(run_dir)).replace("\\", "/") for path in run_dir.rglob("*") if path.is_file())
    return {
        "schema_version": MODEL_RELEASE_SCHEMA_VERSION,
        "status": "prepared",
        "created_at": _utc_now(),
        "release_name": release_name,
        "source_run_dir": str(run_dir),
        "remote_host": remote_host,
        "remote_project": remote_project,
        "remote_release_dir": remote_release_dir,
        "remote_stage_dir": remote_stage_dir,
        "activate_requested": activate,
        "files": files,
        "primary_model": "world_model.onnx",
        "checkpoint": "world_model.pt",
        "training_manifest": "training_pipeline_manifest.json",
    }


def _package_run_dir(run_dir: Path, package_path: Path) -> None:
    with tarfile.open(package_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(run_dir))


def _run_native(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"command failed: {command[0]} exited {exc.returncode}"
        if detail:
            message = f"{message}: {detail}"
        raise ModelReleaseError(message) from exc


def _resolve_train_deploy_config(config: TrainDeployModelConfig) -> ResolvedTrainDeployConfig:
    features = _resolve_required_path(config.features, "NEUROKERNEL_TRAIN_FEATURES", "features")
    out_dir = Path(_resolve_value(config.out_dir, "NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    remote_host = _resolve_value(config.remote_host, "NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = _normalize_remote_project(
        _resolve_value(config.remote_project, "NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")
    )
    run_name = config.run_name or _default_release_name(features)
    _assert_safe_release_name(run_name)
    return ResolvedTrainDeployConfig(
        features=features,
        out_dir=out_dir,
        run_name=run_name,
        remote_host=str(remote_host),
        remote_project=remote_project,
    )


def _preflight_model_release(resolved: ResolvedTrainDeployConfig, config: TrainDeployModelConfig) -> dict[str, Any]:
    feature_manifest = resolved.features.with_suffix(resolved.features.suffix + ".manifest.json")
    if not resolved.features.exists():
        raise ModelReleaseError(f"feature file does not exist: {resolved.features}")
    if not feature_manifest.exists():
        raise ModelReleaseError(f"feature manifest does not exist: {feature_manifest}")
    local_run_dir = resolved.out_dir / resolved.run_name
    if local_run_dir.exists() and any(local_run_dir.iterdir()) and not config.overwrite_local_run:
        raise ModelReleaseError(f"local training run already exists: {local_run_dir}")
    remote_release_dir = f"{resolved.remote_project.rstrip('/')}/artifacts/model_releases/{resolved.run_name}"
    ssh_options = _ssh_options(
        connect_timeout=config.ssh_connect_timeout,
        server_alive_interval=config.ssh_server_alive_interval,
        server_alive_count_max=config.ssh_server_alive_count_max,
    )
    _remote_assert_release_absent(resolved.remote_host, remote_release_dir, ssh_options)
    return {
        "passed": True,
        "features": str(resolved.features),
        "feature_manifest": str(feature_manifest),
        "local_run_dir": str(local_run_dir),
        "remote_host": resolved.remote_host,
        "remote_release_dir": remote_release_dir,
        "checked_at": _utc_now(),
    }


def _remote_assert_release_absent(remote_host: str, remote_release_dir: str, ssh_options: list[str]) -> None:
    _run_native(
        [
            "ssh",
            *ssh_options,
            remote_host,
            (
                "set -eu; "
                f"if [ -e {_sh_quote(remote_release_dir)} ]; then "
                f"echo {_sh_quote('model release already exists: ' + remote_release_dir)} >&2; exit 17; "
                "fi"
            ),
        ]
    )


def _cleanup_remote_stage(remote_host: str, remote_stage_dir: str, ssh_options: list[str]) -> None:
    try:
        _run_native(["ssh", *ssh_options, remote_host, f"rm -rf {_sh_quote(remote_stage_dir)}"])
    except ModelReleaseError:
        pass


def _activate_remote_release(
    remote_host: str,
    remote_project: str,
    remote_releases_dir: str,
    remote_release_dir: str,
    ssh_options: list[str],
) -> None:
    _run_native(
        [
            "ssh",
            *ssh_options,
            remote_host,
            (
                "set -eu; "
                f"cp {_sh_quote(remote_release_dir + '/model_release_manifest.json')} {_sh_quote(remote_releases_dir + '/current.json')}; "
                f"ln -sfn {_sh_quote(remote_release_dir + '/world_model.onnx')} {_sh_quote(remote_project.rstrip('/') + '/artifacts/current_world_model.onnx')}; "
                f"ln -sfn {_sh_quote(remote_release_dir + '/world_model.manifest.json')} {_sh_quote(remote_project.rstrip('/') + '/artifacts/current_world_model.manifest.json')}"
            ),
        ]
    )


def _ssh_options(*, connect_timeout: int, server_alive_interval: int, server_alive_count_max: int) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        f"ServerAliveInterval={server_alive_interval}",
        "-o",
        f"ServerAliveCountMax={server_alive_count_max}",
    ]


def _resolve_required_path(value: str | Path | None, env_name: str, label: str) -> Path:
    resolved = _resolve_value(value, env_name, None)
    if not resolved:
        raise ModelReleaseError(f"missing {label}; provide --{label.replace('_', '-')} or set {env_name}")
    return Path(resolved)


def _resolve_value(value: str | Path | None, env_name: str, default: str | None) -> str | None:
    if value is not None:
        return str(value)
    env_value = os.getenv(env_name)
    if env_value:
        return env_value
    return default


def _normalize_remote_project(path: str) -> str:
    if path == "~":
        return "/home/ubuntu"
    if path.startswith("~/"):
        return "/home/ubuntu/" + path[2:]
    return path


def _assert_safe_release_name(name: str) -> None:
    if not SAFE_RELEASE_NAME.fullmatch(name):
        raise ModelReleaseError("release name may contain only letters, numbers, dot, underscore, and hyphen")


def _default_release_name(features: Path) -> str:
    return f"{features.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
