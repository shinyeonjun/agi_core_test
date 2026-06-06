from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_NAME = "v3.1_first_full_success"
BASELINE_FILES = {
    "world_model_onnx": "world_model.onnx",
    "world_model_manifest": "world_model.manifest.json",
    "world_model_pt": "world_model.pt",
    "world_model_metrics": "world_model.metrics.json",
}


def freeze_baseline(
    *,
    baseline_name: str = DEFAULT_BASELINE_NAME,
    artifacts_dir: str | Path = "artifacts",
    data_dir: str | Path = "data",
    out_dir: str | Path = "baselines",
    evaluation_path: str | Path | None = None,
    artifact_prefix: str = "world_model",
    edge_runtime: str = "orangepi5_onnxruntime",
    training_device: str = "cuda",
    notes: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    artifacts = Path(artifacts_dir)
    data = Path(data_dir)
    baseline_dir = Path(out_dir) / baseline_name
    if baseline_dir.exists() and not overwrite:
        raise FileExistsError(f"baseline already exists: {baseline_dir}")
    baseline_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict[str, Any]] = {}
    baseline_files = _baseline_files(artifact_prefix)
    for key, filename in baseline_files.items():
        copied[key] = _copy_required(artifacts / filename, baseline_dir / filename)

    optional_artifact_files = list(_optional_artifact_files(artifacts))
    for source in optional_artifact_files:
        if source.exists():
            copied[_safe_key(source.name)] = _copy_required(source, baseline_dir / source.name)

    optional_data_files = list(_optional_data_files(data))
    for source in optional_data_files:
        if source.exists():
            copied[_safe_key(source.name)] = _copy_required(source, baseline_dir / source.name)

    evaluation: dict[str, Any] | None = None
    if evaluation_path:
        evaluation_file = Path(evaluation_path)
        if evaluation_file.exists():
            copied["evaluation_result"] = _copy_required(evaluation_file, baseline_dir / evaluation_file.name)
            evaluation = json.loads(evaluation_file.read_text(encoding="utf-8-sig"))

    manifest = {
        "baseline_name": baseline_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_device": training_device,
        "edge_runtime": edge_runtime,
        "git_commit": _git_commit(Path.cwd()),
        "source_hash": _source_hash(Path.cwd() / "src"),
        "artifact_prefix": artifact_prefix,
        "learned_gate_success_rate": _extract_success_rates(evaluation),
        "notes": notes or "Frozen learned gate baseline.",
        "files": copied,
    }
    manifest_path = baseline_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"baseline_name": baseline_name, "out": str(baseline_dir), "manifest": str(manifest_path), "files": copied}


def _baseline_files(artifact_prefix: str) -> dict[str, str]:
    if artifact_prefix == "world_model":
        return dict(BASELINE_FILES)
    return {
        "world_model_onnx": f"{artifact_prefix}.onnx",
        "world_model_manifest": f"{artifact_prefix}.manifest.json",
        "world_model_pt": f"{artifact_prefix}.pt",
        "world_model_metrics": f"{artifact_prefix}.metrics.json",
    }


def _optional_artifact_files(artifacts: Path) -> list[Path]:
    names = {
        "learned_gate_eval.json",
        "hard_heldout_eval.json",
        "hard_failure_traces.json",
        "hard_heldout_eval_v4.json",
        "hard_heldout_eval_v4_orangepi.json",
        "hard_failure_traces_v4.json",
    }
    files = [artifacts / name for name in sorted(names)]
    files.extend(sorted(path for path in artifacts.glob("*.json") if path.name.endswith(("_eval.json", "_traces.json"))))
    return list(dict.fromkeys(files))


def _optional_data_files(data: Path) -> list[Path]:
    files = [
        data / "features.jsonl",
        data / "features.jsonl.manifest.json",
        data / "counterfactual.jsonl",
        data / "counterfactual.jsonl.meta.json",
    ]
    model_ready = data / "model_ready"
    if model_ready.exists():
        files.extend(sorted(model_ready.glob("features*.jsonl")))
        files.extend(sorted(model_ready.glob("features*.jsonl.manifest.json")))
        files.extend(sorted(model_ready.glob("counterfactual*.jsonl")))
        files.extend(sorted(model_ready.glob("counterfactual*.jsonl.meta.json")))
    return list(dict.fromkeys(files))


def _copy_required(source: Path, destination: Path) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {"source": str(source), "path": str(destination), "sha256": _file_hash(destination), "bytes": destination.stat().st_size}


def _extract_success_rates(evaluation: dict[str, Any] | None) -> dict[str, float] | None:
    if not evaluation:
        return None
    rates: dict[str, float] = {}
    groups = evaluation.get("groups")
    if isinstance(groups, dict):
        for group_name, group in groups.items():
            if not isinstance(group, dict):
                continue
            aggregate = group.get("aggregate")
            if isinstance(aggregate, dict) and "learned_gate_success_rate" in aggregate:
                rates[f"group:{group_name}"] = float(aggregate["learned_gate_success_rate"])
            envs = group.get("envs")
            if isinstance(envs, dict):
                for env_name, result in envs.items():
                    learned = result.get("learned_gate") if isinstance(result, dict) else None
                    if isinstance(learned, dict) and "success_rate" in learned:
                        rates[f"{group_name}:{env_name}"] = float(learned["success_rate"])
    for env_name, result in evaluation.items():
        if isinstance(result, dict) and "success_rate" in result:
            rates[str(env_name)] = float(result["success_rate"])
    return rates or None


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*.py")):
        digest.update(str(file_path.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, capture_output=True, check=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def _safe_key(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")
