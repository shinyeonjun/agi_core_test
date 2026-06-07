from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.eval.gate_ablation import GateAblationConfig, run_gate_ablation
from neurokernel_seed.model.evaluate import eval_world_model
from neurokernel_seed.model.export_onnx import export_onnx
from neurokernel_seed.model.ranking import eval_action_ranking
from neurokernel_seed.model.train import train_world_model
from neurokernel_seed.replay.dataset import validate_flat_features
from neurokernel_seed.replay.slot_dataset import check_slot_dataset_gates, validate_slot_features


PIPELINE_SCHEMA_VERSION = "neurokernel-training-pipeline-v1"


class TrainingPipelineError(RuntimeError):
    def __init__(self, message: str, *, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


@dataclass(frozen=True)
class TrainingPipelineConfig:
    features: str | Path
    out_dir: str | Path
    run_name: str | None = None
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
    overwrite: bool = False


def run_training_pipeline(config: TrainingPipelineConfig) -> dict[str, Any]:
    features = Path(config.features)
    run_name = config.run_name or _default_run_name(features)
    run_dir = Path(config.out_dir) / run_name
    _prepare_run_dir(run_dir, overwrite=config.overwrite)
    status_path = run_dir / "training_pipeline_manifest.json"
    status: dict[str, Any] = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "features": str(features),
        "config": _config_payload(config),
        "steps": [],
        "artifacts": {},
    }
    _write_json(status_path, status)
    try:
        feature_manifest = _load_feature_manifest(features)
        status["feature_manifest"] = _manifest_summary(feature_manifest)
        validation = _run_step(status, status_path, "validate_features", lambda: _validate_features(features, feature_manifest))
        _write_json(run_dir / "feature_validation.json", validation)
        if _is_slot_schema(feature_manifest):
            gates = _run_step(status, status_path, "check_dataset_gates", lambda: check_slot_dataset_gates(features))
            _write_json(run_dir / "dataset_gates.json", gates)
            if not gates.get("passed") and not config.allow_gate_failure:
                raise TrainingPipelineError("dataset gates failed", payload=gates)
        checkpoint = run_dir / "world_model.pt"
        train_result = _run_step(
            status,
            status_path,
            "train_world_model",
            lambda: train_world_model(
                features,
                checkpoint,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                hidden_dim=config.hidden_dim,
                hidden_layers=config.hidden_layers,
                device=config.device,
                patience=config.patience,
            ),
        )
        _write_json(run_dir / "train_result.json", train_result)
        status["artifacts"]["checkpoint"] = str(checkpoint)
        eval_result = _run_step(
            status,
            status_path,
            f"eval_world_model_{config.split}",
            lambda: eval_world_model(checkpoint, features, split=config.split, device=config.device),
        )
        eval_path = run_dir / f"world_model_eval_{config.split}.json"
        _write_json(eval_path, eval_result)
        status["artifacts"]["world_model_eval"] = str(eval_path)
        ranking_result = _run_step(
            status,
            status_path,
            f"eval_action_ranking_{config.split}",
            lambda: eval_action_ranking(checkpoint, features, split=config.split, device=config.device),
        )
        ranking_path = run_dir / f"action_ranking_{config.split}.json"
        _write_json(ranking_path, ranking_result)
        status["artifacts"]["action_ranking"] = str(ranking_path)
        onnx_path = run_dir / "world_model.onnx"
        onnx_result = _run_step(
            status,
            status_path,
            "export_world_model_onnx",
            lambda: export_onnx(checkpoint, onnx_path, verify=config.verify_onnx, dynamic_batch=config.dynamic_batch),
        )
        _write_json(run_dir / "onnx_export.json", onnx_result)
        status["artifacts"]["onnx"] = str(onnx_path)
        status["artifacts"]["onnx_manifest"] = str(onnx_path.with_suffix(".manifest.json"))
        if config.run_gate_ablation:
            ablation = _run_step(
                status,
                status_path,
                "run_gate_ablation",
                lambda: run_gate_ablation(
                    onnx_path,
                    out_dir=run_dir / "gate_ablation",
                    config=GateAblationConfig(
                        episodes=config.gate_ablation_episodes,
                        trace_episodes=config.trace_episodes,
                        max_failures_per_env=config.max_failures_per_env,
                        strict=config.strict,
                    ),
                ),
            )
            _write_json(run_dir / "gate_ablation_result.json", ablation)
            status["artifacts"]["gate_ablation"] = str(run_dir / "gate_ablation" / "gate_ablation_v4_4.json")
            if not _gate_ablation_promoted(ablation) and not config.allow_benchmark_failure:
                raise TrainingPipelineError("gate ablation did not promote the model", payload=ablation)
        status["status"] = "completed"
        status["completed_at"] = _utc_now()
        status["updated_at"] = status["completed_at"]
        _write_json(status_path, status)
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["failed_at"] = _utc_now()
        status["updated_at"] = status["failed_at"]
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        if isinstance(exc, TrainingPipelineError) and exc.payload:
            status["error_payload"] = exc.payload
        _write_json(status_path, status)
        raise


def _run_step(status: dict[str, Any], status_path: Path, name: str, fn) -> dict[str, Any]:
    step = {"name": name, "status": "running", "started_at": _utc_now()}
    status["steps"].append(step)
    status["updated_at"] = step["started_at"]
    _write_json(status_path, status)
    try:
        result = fn()
    except Exception as exc:
        step["status"] = "failed"
        step["failed_at"] = _utc_now()
        step["error_type"] = type(exc).__name__
        step["error_message"] = str(exc)
        status["updated_at"] = step["failed_at"]
        _write_json(status_path, status)
        raise
    step["status"] = "completed"
    step["completed_at"] = _utc_now()
    step["summary"] = _step_summary(result)
    status["updated_at"] = step["completed_at"]
    _write_json(status_path, status)
    return result


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise TrainingPipelineError(f"training run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)


def _load_feature_manifest(features: Path) -> dict[str, Any]:
    manifest_path = features.with_suffix(features.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise TrainingPipelineError(f"missing feature manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_features(features: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if _is_slot_schema(manifest):
        return validate_slot_features(features)
    return validate_flat_features(features)


def _is_slot_schema(manifest: dict[str, Any]) -> bool:
    return str(manifest.get("schema_version", "")).startswith("neurokernel-slot-transition")


def _gate_ablation_promoted(result: dict[str, Any]) -> bool:
    mode_passed = result.get("mode_passed", {})
    return bool(mode_passed.get("hybrid") or mode_passed.get("hybrid_veto"))


def _default_run_name(features: Path) -> str:
    return f"{features.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _config_payload(config: TrainingPipelineConfig) -> dict[str, Any]:
    payload = dict(config.__dict__)
    payload["features"] = str(config.features)
    payload["out_dir"] = str(config.out_dir)
    return payload


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = ("schema_version", "dataset_profile", "rows", "input_dim", "target_dim", "splits")
    return {key: manifest.get(key) for key in keys if key in manifest}


def _step_summary(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "passed",
        "rows",
        "schema_version",
        "dataset_profile",
        "checkpoint",
        "onnx",
        "split",
        "device",
        "candidate_groups",
        "grouped_top1_action_accuracy",
        "pairwise_ranking_accuracy",
        "counterfactual_regret",
        "out",
    )
    return {key: result.get(key) for key in keys if key in result}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
