import json

import pytest

from neurokernel_seed.model.runtime_action import (
    RUNTIME_ACTION_MODEL_SCHEMA_VERSION,
    eval_runtime_action_checkpoint,
    train_runtime_action_model,
)
from neurokernel_seed.replay.runtime_features import (
    NUMERIC_FEATURE_NAMES,
    RUNTIME_FEATURE_SCHEMA_VERSION,
    TARGET_NAMES,
    check_runtime_feature_gates,
)


pytest.importorskip("torch")


def test_runtime_action_model_trains_and_evaluates(tmp_path):
    features = _write_runtime_features(tmp_path, split_mode="train_test")
    checkpoint = tmp_path / "runtime_action_model.pt"

    result = train_runtime_action_model(
        features,
        checkpoint,
        epochs=8,
        batch_size=4,
        hidden_dim=8,
        hidden_layers=1,
        device="cpu",
        patience=4,
        min_rows=8,
        min_actions=2,
    )
    evaluated = eval_runtime_action_checkpoint(checkpoint, features, split="test", device="cpu")

    assert checkpoint.exists()
    assert checkpoint.with_suffix(".metrics.json").exists()
    assert result["dataset_gate"]["ready_for_runtime_model_training"] is True
    assert result["train"]["rows"] > 0
    assert result["test"]["rows"] > 0
    assert "success_accuracy" in evaluated
    assert evaluated["split"] == "test"
    assert _checkpoint_schema(checkpoint) == RUNTIME_ACTION_MODEL_SCHEMA_VERSION


def test_runtime_action_model_refuses_dataset_without_test_split(tmp_path):
    features = _write_runtime_features(tmp_path, split_mode="train_only")
    gates = check_runtime_feature_gates(features, min_rows=8, min_actions=2)

    with pytest.raises(RuntimeError, match="runtime feature gates failed"):
        train_runtime_action_model(features, tmp_path / "runtime_action_model.pt", epochs=2, device="cpu", min_rows=8, min_actions=2)

    assert gates["ready_for_runtime_model_training"] is False
    assert gates["gates"]["train_and_test_splits"]["passed"] is False


def _write_runtime_features(tmp_path, *, split_mode: str):
    features = tmp_path / "runtime_features.jsonl"
    actions = ["inspect", "repair"]
    sources = ["discord"]
    targets = ["orangepi5"]
    statuses = ["completed"]
    risks = ["low"]
    safety = ["allow"]
    input_dim = len(actions) + len(sources) + len(targets) + len(statuses) + len(risks) + len(safety) + len(NUMERIC_FEATURE_NAMES)
    manifest = {
        "schema_version": RUNTIME_FEATURE_SCHEMA_VERSION,
        "replay_schema_version": "neurokernel-runtime-action-v1",
        "rows": 12,
        "splits": ["train"] if split_mode == "train_only" else ["test", "train"],
        "action_vocab": actions,
        "source_vocab": sources,
        "target_vocab": targets,
        "status_vocab": statuses,
        "risk_vocab": risks,
        "safety_vocab": safety,
        "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "input_dim": input_dim,
        "target_dim": len(TARGET_NAMES),
        "input_layout": {},
        "target_layout": {name: index for index, name in enumerate(TARGET_NAMES)},
        "split_policy": {"type": "test_fixture", "test_ratio": 0.25},
    }
    rows = []
    for index in range(12):
        action = actions[index % len(actions)]
        success = 1.0 if action == "repair" else 0.0
        split = "train" if split_mode == "train_only" or index % 4 else "test"
        input_vector = _onehot(action, actions)
        input_vector += [1.0, 1.0, 1.0, 1.0, 1.0]
        input_vector += [
            0.0,
            2.0,
            2.0,
            1.0,
            float(index),
            12.0,
            2.0,
            0.0,
            0.5,
            0.0,
            1.0,
        ]
        rows.append(
            {
                "schema_version": RUNTIME_FEATURE_SCHEMA_VERSION,
                "row_index": index,
                "row_id": f"row_{index}",
                "split": split,
                "env_name": "runtime.orangepi5",
                "action_key": action,
                "candidate_set_id": f"candidate_{index}",
                "input_vector": input_vector,
                "target_vector": [success, 1.0 if success else -1.0, 0.01 * index, 0.0 if success else 1.0],
                "target_mask": [1.0, 1.0, 1.0, 1.0],
                "actual_action_score": 1.0 if success else -1.0,
                "source": {"runtime_row_id": f"row_{index}", "lineage": {"decision_id": index}},
            }
        )
    features.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    features.with_suffix(features.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return features


def _onehot(value, vocab):
    return [1.0 if item == value else 0.0 for item in vocab]


def _checkpoint_schema(checkpoint):
    import torch

    payload = torch.load(checkpoint, map_location="cpu")
    return payload["schema_version"]
