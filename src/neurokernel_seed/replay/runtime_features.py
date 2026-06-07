from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from neurokernel_seed.replay.runtime_dataset import RUNTIME_REPLAY_SCHEMA_VERSION, validate_runtime_replay

RUNTIME_FEATURE_SCHEMA_VERSION = "neurokernel-runtime-action-feature-v1"
TARGET_NAMES = ["success", "reward", "duration_seconds_log1p", "failure_present"]
NUMERIC_FEATURE_NAMES = [
    "requires_approval",
    "candidate_count",
    "allowed_action_count",
    "chosen_in_candidates",
    "step",
    "goal_chars",
    "goal_words",
    "goal_digits",
    "goal_non_ascii_ratio",
    "has_model_score",
    "has_gate_trace",
]


@dataclass(frozen=True)
class RuntimeFeatureManifest:
    schema_version: str
    replay_schema_version: str
    rows: int
    splits: list[str]
    action_vocab: list[str]
    source_vocab: list[str]
    target_vocab: list[str]
    status_vocab: list[str]
    risk_vocab: list[str]
    safety_vocab: list[str]
    numeric_feature_names: list[str]
    target_names: list[str]
    input_dim: int
    target_dim: int
    input_layout: dict[str, Any]
    target_layout: dict[str, Any]
    split_policy: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "replay_schema_version": self.replay_schema_version,
            "rows": self.rows,
            "splits": self.splits,
            "action_vocab": self.action_vocab,
            "source_vocab": self.source_vocab,
            "target_vocab": self.target_vocab,
            "status_vocab": self.status_vocab,
            "risk_vocab": self.risk_vocab,
            "safety_vocab": self.safety_vocab,
            "numeric_feature_names": self.numeric_feature_names,
            "target_names": self.target_names,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "input_layout": self.input_layout,
            "target_layout": self.target_layout,
            "split_policy": self.split_policy,
        }


def export_runtime_features(
    replay_path: str | Path,
    out_path: str | Path,
    *,
    test_ratio: float = 0.2,
) -> dict[str, Any]:
    rows = _load_runtime_rows(replay_path)
    items = _feature_items(rows)
    manifest = build_runtime_feature_manifest(rows, test_ratio=test_ratio)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        for row_index, item in enumerate(items):
            sample = encode_runtime_feature_row(item["row"], manifest, row_index, candidate=item.get("candidate"))
            fp.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_runtime_features(out)
    return {
        "out": str(out),
        "manifest": str(manifest_path),
        "rows": validation["rows"],
        "schema_version": validation["schema_version"],
        "input_dim": validation["input_dim"],
        "target_dim": validation["target_dim"],
        "splits": validation["splits"],
    }


def validate_runtime_features(path: str | Path) -> dict[str, Any]:
    feature_path = Path(path)
    manifest_path = feature_path.with_suffix(feature_path.suffix + ".manifest.json")
    if not feature_path.exists():
        raise ValueError(f"missing runtime feature file: {feature_path}")
    if not manifest_path.exists():
        raise ValueError(f"missing runtime feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[dict[str, Any]] = []
    rows = 0
    splits: set[str] = set()
    action_counts: dict[str, int] = {}
    for line_no, line in enumerate(feature_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        sample = json.loads(line)
        errors.extend(_validate_sample(sample, manifest, line_no))
        rows += 1
        splits.add(str(sample.get("split")))
        action_key = str(sample.get("action_key") or "")
        action_counts[action_key] = action_counts.get(action_key, 0) + 1
    if rows != manifest.get("rows"):
        errors.append({"line": None, "error": f"row count mismatch: data={rows}, manifest={manifest.get('rows')}"})
    return {
        "accepted": not errors,
        "path": str(feature_path),
        "manifest": str(manifest_path),
        "rows": rows,
        "schema_version": manifest.get("schema_version"),
        "input_dim": manifest.get("input_dim"),
        "target_dim": manifest.get("target_dim"),
        "splits": sorted(splits),
        "action_counts": dict(sorted(action_counts.items())),
        "errors": errors[:50],
    }


def check_runtime_feature_gates(path: str | Path, *, min_rows: int = 10, min_actions: int = 1) -> dict[str, Any]:
    validation = validate_runtime_features(path)
    rows = _load_feature_rows(path) if validation["accepted"] else []
    success_values = [float(row["target_vector"][0]) for row in rows]
    known_outcome_rows = sum(1 for row in rows if any(float(value) > 0.0 for value in row.get("target_mask", [])))
    action_count = len(validation["action_counts"])
    splits = set(validation.get("splits") or [])
    gates = {
        "schema_valid": {"passed": bool(validation["accepted"])},
        "min_rows": {"passed": validation["rows"] >= min_rows, "actual": validation["rows"], "threshold": min_rows},
        "min_actions": {"passed": action_count >= min_actions, "actual": action_count, "threshold": min_actions},
        "train_and_test_splits": {"passed": {"train", "test"}.issubset(splits), "actual": sorted(splits), "threshold": ["train", "test"]},
        "targets_present": {"passed": all(len(row.get("target_vector", [])) == len(TARGET_NAMES) for row in rows) if rows else False},
        "known_outcome_rows": {"passed": known_outcome_rows > 0, "actual": known_outcome_rows, "threshold": 1},
    }
    warnings = []
    if rows and len(set(success_values)) < 2:
        warnings.append({"kind": "single_success_class", "message": "현재 runtime feature에는 성공/실패 다양성이 부족함"})
    if action_count <= 1:
        warnings.append({"kind": "single_action", "message": "현재 runtime feature에는 action 다양성이 부족함"})
    return {
        "passed": all(item["passed"] for item in gates.values()),
        "path": str(path),
        "rows": validation["rows"],
        "schema_version": validation["schema_version"],
        "input_dim": validation["input_dim"],
        "target_dim": validation["target_dim"],
        "known_outcome_rows": known_outcome_rows,
        "action_counts": validation["action_counts"],
        "validation": validation,
        "gates": gates,
        "warnings": warnings,
        "ready_for_runtime_model_training": all(item["passed"] for item in gates.values()),
    }


def build_runtime_feature_manifest(rows: list[dict[str, Any]], *, test_ratio: float = 0.2) -> RuntimeFeatureManifest:
    if not rows:
        raise ValueError("cannot build runtime feature manifest from empty replay")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0.0, 1.0)")
    items = _feature_items(rows)
    action_vocab = sorted({_feature_action_key(item["row"], item.get("candidate")) for item in items} | {_candidate_action_key(candidate) for row in rows for candidate in _candidate_actions(row)})
    source_vocab = sorted({str(row.get("task", {}).get("source") or "unknown") for row in rows})
    target_vocab = sorted({str(row.get("task", {}).get("target") or "unknown") for row in rows})
    status_vocab = sorted({str(row.get("task", {}).get("status") or "unknown") for row in rows})
    risk_vocab = sorted({str(row.get("task", {}).get("risk_level") or "unknown") for row in rows})
    safety_vocab = sorted({_safety_decision_key(row) for row in rows})
    splits = sorted({_split_for_row(row, test_ratio=test_ratio) for row in rows})
    layout: dict[str, Any] = {}
    cursor = 0
    for name, vocab in (
        ("action_onehot", action_vocab),
        ("source_onehot", source_vocab),
        ("target_onehot", target_vocab),
        ("status_onehot", status_vocab),
        ("risk_onehot", risk_vocab),
        ("safety_onehot", safety_vocab),
        ("numeric", NUMERIC_FEATURE_NAMES),
    ):
        layout[name] = [cursor, cursor + len(vocab)]
        cursor += len(vocab)
    target_layout = {name: index for index, name in enumerate(TARGET_NAMES)}
    return RuntimeFeatureManifest(
        schema_version=RUNTIME_FEATURE_SCHEMA_VERSION,
        replay_schema_version=RUNTIME_REPLAY_SCHEMA_VERSION,
        rows=len(items),
        splits=sorted({_split_for_row(item["row"], test_ratio=test_ratio) for item in items}),
        action_vocab=action_vocab,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        status_vocab=status_vocab,
        risk_vocab=risk_vocab,
        safety_vocab=safety_vocab,
        numeric_feature_names=list(NUMERIC_FEATURE_NAMES),
        target_names=list(TARGET_NAMES),
        input_dim=cursor,
        target_dim=len(TARGET_NAMES),
        input_layout=layout,
        target_layout=target_layout,
        split_policy={"type": "row_id_sha256_modulo", "test_ratio": test_ratio},
    )


def encode_runtime_feature_row(row: dict[str, Any], manifest: RuntimeFeatureManifest, row_index: int = 0, *, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    action_key = _feature_action_key(row, candidate)
    input_vector = _onehot(action_key, manifest.action_vocab)
    input_vector += _onehot(str(row.get("task", {}).get("source") or "unknown"), manifest.source_vocab)
    input_vector += _onehot(str(row.get("task", {}).get("target") or "unknown"), manifest.target_vocab)
    input_vector += _onehot(str(row.get("task", {}).get("status") or "unknown"), manifest.status_vocab)
    input_vector += _onehot(str(row.get("task", {}).get("risk_level") or "unknown"), manifest.risk_vocab)
    input_vector += _onehot(_safety_decision_key(row), manifest.safety_vocab)
    input_vector += _numeric_features(row, action_key)
    target_vector = _target_vector(row, candidate)
    target_mask = _target_mask(candidate)
    return {
        "schema_version": manifest.schema_version,
        "row_index": row_index,
        "row_id": row.get("row_id"),
        "split": _split_for_row(row, test_ratio=float(manifest.split_policy["test_ratio"])),
        "env_name": f"runtime.{row.get('task', {}).get('target') or 'unknown'}",
        "action_key": action_key,
        "candidate_set_id": _candidate_set_id(row),
        "input_vector": input_vector,
        "target_vector": target_vector,
        "target_mask": target_mask,
        "actual_action_score": float(target_vector[1]) if any(target_mask) else 0.0,
        "source": {
            "runtime_row_id": row.get("row_id"),
            "lineage": row.get("lineage"),
            "candidate": _candidate_source(candidate),
        },
    }


def _load_runtime_rows(path: str | Path) -> list[dict[str, Any]]:
    validation = validate_runtime_replay(path)
    if not validation["accepted"]:
        raise ValueError(f"runtime replay validation failed: {validation['errors'][:3]}")
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no runtime replay rows found: {path}")
    return rows


def _load_feature_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _feature_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        candidates = row.get("candidate_outcomes")
        if isinstance(candidates, list) and candidates:
            for candidate in candidates:
                if isinstance(candidate, dict):
                    items.append({"row": row, "candidate": candidate})
            continue
        items.append({"row": row, "candidate": None})
    return items


def _validate_sample(sample: Any, manifest: dict[str, Any], line_no: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(sample, dict):
        return [{"line": line_no, "error": "sample must be object"}]
    required = ["schema_version", "row_id", "input_vector", "target_vector", "target_mask", "split", "env_name", "action_key", "source"]
    for field in required:
        if field not in sample:
            errors.append({"line": line_no, "error": f"{field} is required"})
    if sample.get("schema_version") != manifest.get("schema_version"):
        errors.append({"line": line_no, "error": "schema_version mismatch"})
    if len(sample.get("input_vector", [])) != int(manifest.get("input_dim", -1)):
        errors.append({"line": line_no, "error": "input_vector dimension mismatch"})
    if len(sample.get("target_vector", [])) != int(manifest.get("target_dim", -1)):
        errors.append({"line": line_no, "error": "target_vector dimension mismatch"})
    if len(sample.get("target_mask", [])) != int(manifest.get("target_dim", -1)):
        errors.append({"line": line_no, "error": "target_mask dimension mismatch"})
    for name in ("input_vector", "target_vector", "target_mask"):
        values = sample.get(name)
        if isinstance(values, list) and not all(_is_finite_number(value) for value in values):
            errors.append({"line": line_no, "error": f"{name} must contain only finite numbers"})
    if sample.get("action_key") not in set(manifest.get("action_vocab", [])):
        errors.append({"line": line_no, "error": "action_key not in manifest action_vocab"})
    if sample.get("split") not in set(manifest.get("splits", [])):
        errors.append({"line": line_no, "error": "split not in manifest splits"})
    return errors


def _chosen_action_key(row: dict[str, Any]) -> str:
    chosen = row.get("decision", {}).get("chosen_action")
    if isinstance(chosen, dict):
        return str(chosen.get("action_id") or chosen.get("name") or "unknown")
    execution_action = row.get("execution", {}).get("action_id")
    return str(execution_action or "unknown")


def _feature_action_key(row: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
    if candidate is not None:
        return str(candidate.get("action_id") or "unknown")
    return _chosen_action_key(row)


def _candidate_actions(row: dict[str, Any]) -> list[Any]:
    candidates = row.get("decision", {}).get("candidate_actions")
    return candidates if isinstance(candidates, list) else []


def _candidate_action_key(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("action_id") or candidate.get("name") or "unknown")
    return str(candidate or "unknown")


def _safety_decision_key(row: dict[str, Any]) -> str:
    safety = row.get("decision", {}).get("safety_decision")
    if isinstance(safety, dict):
        return str(safety.get("decision") or safety.get("status") or "unknown")
    return "unknown"


def _candidate_set_id(row: dict[str, Any]) -> str:
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    decision_id = lineage.get("decision_id")
    return f"runtime_decision_{decision_id}" if decision_id is not None else str(row.get("row_id") or "runtime_unknown")


def _numeric_features(row: dict[str, Any], action_key: str) -> list[float]:
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    allowed = task.get("allowed_actions") if isinstance(task.get("allowed_actions"), list) else []
    candidates = _candidate_actions(row)
    goal = str(task.get("goal_redacted") or "")
    non_ascii = sum(1 for char in goal if ord(char) > 127)
    return [
        1.0 if task.get("requires_approval") else 0.0,
        float(len(candidates)),
        float(len(allowed)),
        1.0 if action_key in {_candidate_action_key(candidate) for candidate in candidates} else 0.0,
        float(decision.get("step") or 0),
        float(len(goal)),
        float(len(goal.split())),
        float(sum(1 for char in goal if char.isdigit())),
        float(non_ascii / max(1, len(goal))),
        1.0 if decision.get("model_score") else 0.0,
        1.0 if decision.get("gate_trace") else 0.0,
    ]


def _target_vector(row: dict[str, Any], candidate: dict[str, Any] | None = None) -> list[float]:
    if candidate is not None:
        outcome = candidate.get("outcome") if isinstance(candidate.get("outcome"), dict) else {}
        if candidate.get("execution_result_known"):
            success = 1.0 if outcome.get("success") is True else 0.0
            return [
                success,
                float(outcome.get("reward") or (1.0 if success else -1.0)),
                math.log1p(max(0.0, float(outcome.get("duration_seconds") or 0.0))),
                1.0 if outcome.get("failure_present") else 0.0,
            ]
        return [0.0, 0.0, 0.0, 0.0]
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
    duration_seconds = _duration_seconds(execution.get("started_at"), execution.get("ended_at"))
    failure_reason = outcome.get("failure_reason") or outcome.get("failure_bucket") or execution.get("error_type")
    success = 1.0 if outcome.get("success") is True else 0.0
    return [
        success,
        float(outcome.get("reward") or (1.0 if success else -1.0)),
        math.log1p(max(0.0, duration_seconds)),
        1.0 if failure_reason else 0.0,
    ]


def _target_mask(candidate: dict[str, Any] | None = None) -> list[float]:
    if candidate is None:
        return [1.0] * len(TARGET_NAMES)
    mask = candidate.get("target_mask") if isinstance(candidate.get("target_mask"), dict) else {}
    return [
        1.0 if mask.get("success") else 0.0,
        1.0 if mask.get("reward") else 0.0,
        1.0 if mask.get("duration_seconds") else 0.0,
        1.0 if mask.get("failure_present") else 0.0,
    ]


def _candidate_source(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if candidate is None:
        return {}
    return {
        "action_id": candidate.get("action_id"),
        "candidate_index": candidate.get("candidate_index"),
        "selected": bool(candidate.get("selected")),
        "executed": bool(candidate.get("executed")),
        "execution_result_known": bool(candidate.get("execution_result_known")),
    }


def _duration_seconds(started_at: Any, ended_at: Any) -> float:
    try:
        if not started_at or not ended_at:
            return 0.0
        start = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(ended_at).replace("Z", "+00:00"))
        return max(0.0, (end - start).total_seconds())
    except ValueError:
        return 0.0


def _split_for_row(row: dict[str, Any], *, test_ratio: float) -> str:
    if test_ratio <= 0.0:
        return "train"
    row_id = str(row.get("row_id") or json.dumps(row.get("lineage", {}), sort_keys=True))
    bucket = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_ratio else "train"


def _onehot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if item == value else 0.0 for item in vocab]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
