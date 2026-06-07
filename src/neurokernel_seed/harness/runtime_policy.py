from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurokernel_seed.model.dataset import require_torch
from neurokernel_seed.model.mlp import WorldModelConfig, build_model
from neurokernel_seed.model.runtime_action import RUNTIME_ACTION_MODEL_SCHEMA_VERSION
from neurokernel_seed.replay.runtime_features import NUMERIC_FEATURE_NAMES, TARGET_NAMES

from .task_spec import TaskSpec


@dataclass(frozen=True)
class RuntimePolicyConfig:
    model_path: Path | None = None


class RuntimeActionPolicy:
    def __init__(self, config: RuntimePolicyConfig | None = None):
        self.config = config or RuntimePolicyConfig()
        self._loaded: dict[str, Any] | None = None
        self._load_error: str | None = None

    @classmethod
    def from_env(cls, project_root: str | Path = ".") -> "RuntimeActionPolicy":
        root = Path(project_root)
        configured = os.environ.get("NEUROKERNEL_RUNTIME_ACTION_MODEL")
        if configured:
            model_path = Path(configured)
        else:
            model_path = root / "artifacts" / "current_runtime_action_model.pt"
        return cls(RuntimePolicyConfig(model_path=model_path))

    def rank(self, *, task: TaskSpec, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        loaded = self._load()
        if loaded is None:
            return _unavailable("model_unavailable", self._load_error or "runtime action model not loaded", decisions)

        scored: list[dict[str, Any]] = []
        per_action: dict[str, dict[str, Any]] = {}
        for decision in decisions:
            action_id = str(decision.get("action_id") or "")
            safety = decision.get("safety") if isinstance(decision.get("safety"), dict) else {}
            if safety.get("decision") not in {"allow", "dry_run_only"}:
                per_action[action_id] = {
                    "model_used": False,
                    "reason": "safety_not_rankable",
                    "safety_decision": safety.get("decision"),
                }
                continue
            if action_id not in set(loaded["manifest"].get("action_vocab") or []):
                per_action[action_id] = {
                    "model_used": False,
                    "reason": "action_not_in_runtime_vocab",
                    "action_id": action_id,
                }
                continue
            try:
                vector = _encode_live_feature(task, action_id=action_id, safety_decision=str(safety.get("decision") or "unknown"), manifest=loaded["manifest"])
                score = _score_vector(loaded, vector)
            except Exception as exc:
                per_action[action_id] = {"model_used": False, "reason": "runtime_policy_encode_failed", "error": str(exc)}
                continue
            per_action[action_id] = score
            scored.append({"action_id": action_id, **score})

        if not scored:
            return {
                "model_used": False,
                "reason": "no_rankable_runtime_candidates",
                "model_path": str(self.config.model_path) if self.config.model_path else None,
                "scores": per_action,
                "ranked_actions": [],
            }
        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        return {
            "model_used": True,
            "reason": "ranked_by_runtime_action_model",
            "model_path": str(self.config.model_path) if self.config.model_path else None,
            "schema_version": RUNTIME_ACTION_MODEL_SCHEMA_VERSION,
            "ranked_actions": [item["action_id"] for item in scored],
            "scores": per_action,
            "top_action": scored[0]["action_id"],
        }

    def _load(self) -> dict[str, Any] | None:
        if self._loaded is not None:
            return self._loaded
        model_path = self.config.model_path
        if model_path is None:
            self._load_error = "runtime action model path is not configured"
            return None
        if not model_path.exists():
            self._load_error = f"runtime action model not found: {model_path}"
            return None
        try:
            torch = require_torch()
            checkpoint = torch.load(model_path, map_location="cpu")
            if checkpoint.get("schema_version") != RUNTIME_ACTION_MODEL_SCHEMA_VERSION:
                raise ValueError("runtime action model schema mismatch")
            manifest = checkpoint.get("manifest")
            if not isinstance(manifest, dict):
                raise ValueError("runtime action model manifest missing")
            _validate_runtime_manifest(manifest)
            config = WorldModelConfig.from_dict(checkpoint["config"])
            model = build_model(config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._loaded = {"torch": torch, "model": model, "manifest": manifest}
            return self._loaded
        except Exception as exc:
            self._load_error = str(exc)
            return None


def _unavailable(reason: str, detail: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model_used": False,
        "reason": reason,
        "detail": detail,
        "ranked_actions": [],
        "scores": {
            str(decision.get("action_id") or ""): {"model_used": False, "reason": reason, "detail": detail}
            for decision in decisions
        },
    }


def _validate_runtime_manifest(manifest: dict[str, Any]) -> None:
    if int(manifest.get("target_dim") or 0) != len(TARGET_NAMES):
        raise ValueError("runtime action model target dimension mismatch")
    for key in ("input_dim", "action_vocab", "source_vocab", "target_vocab", "status_vocab", "risk_vocab", "safety_vocab", "target_layout"):
        if key not in manifest:
            raise ValueError(f"runtime action manifest missing {key}")


def _encode_live_feature(task: TaskSpec, *, action_id: str, safety_decision: str, manifest: dict[str, Any]) -> list[float]:
    source = str(task.context.get("source") or task.context.get("request_source") or "unknown")
    status = str(task.context.get("decision_status") or "deciding")
    values: list[float] = []
    values += _onehot(action_id, list(manifest.get("action_vocab") or []))
    values += _onehot(source, list(manifest.get("source_vocab") or []))
    values += _onehot(task.target, list(manifest.get("target_vocab") or []))
    values += _onehot(status, list(manifest.get("status_vocab") or []))
    values += _onehot(task.risk_level, list(manifest.get("risk_vocab") or []))
    values += _onehot(safety_decision, list(manifest.get("safety_vocab") or []))
    values += _numeric_features(task, action_id)
    expected = int(manifest.get("input_dim") or 0)
    if len(values) != expected:
        raise ValueError(f"runtime live feature input dimension mismatch: got {len(values)}, expected {expected}")
    return values


def _numeric_features(task: TaskSpec, action_id: str) -> list[float]:
    goal = str(task.goal or "")
    non_ascii = sum(1 for char in goal if ord(char) > 127)
    candidates = list(task.allowed_actions)
    allowed = [item for item in task.allowed_actions if item not in set(task.blocked_actions)]
    values = [
        1.0 if task.requires_approval else 0.0,
        float(len(candidates)),
        float(len(allowed)),
        1.0 if action_id in set(candidates) else 0.0,
        0.0,
        float(len(goal)),
        float(len(goal.split())),
        float(sum(1 for char in goal if char.isdigit())),
        float(non_ascii / max(1, len(goal))),
        0.0,
        0.0,
    ]
    if len(values) != len(NUMERIC_FEATURE_NAMES):
        raise ValueError("runtime numeric feature layout mismatch")
    return values


def _score_vector(loaded: dict[str, Any], vector: list[float]) -> dict[str, Any]:
    torch = loaded["torch"]
    model = loaded["model"]
    manifest = loaded["manifest"]
    layout = manifest["target_layout"]
    with torch.no_grad():
        inputs = torch.tensor([vector], dtype=torch.float32)
        prediction = model(inputs)[0]
        success_probability = float(torch.sigmoid(prediction[int(layout["success"])]).item())
        reward = float(prediction[int(layout["reward"])].item())
        duration_seconds_log1p = max(0.0, float(prediction[int(layout["duration_seconds_log1p"])].item()))
        failure_probability = float(torch.sigmoid(prediction[int(layout["failure_present"])]).item())
    score = success_probability + 0.5 * reward - 0.2 * duration_seconds_log1p - 0.8 * failure_probability
    return {
        "model_used": True,
        "reason": "runtime_action_model_score",
        "score": float(score),
        "success_probability": success_probability,
        "predicted_reward": reward,
        "predicted_duration_log1p": duration_seconds_log1p,
        "failure_probability": failure_probability,
    }


def _onehot(value: str, vocab: list[str]) -> list[float]:
    return [1.0 if item == value else 0.0 for item in vocab]
