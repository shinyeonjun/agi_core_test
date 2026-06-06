import json
from pathlib import Path

from neurokernel_seed.core.belief import belief_from_facts
from neurokernel_seed.replay.counterfactual import export_candidate_counterfactuals
from neurokernel_seed.replay.slot_dataset import audit_slot_dataset, export_balanced_slot_dataset, export_slot_features, validate_slot_features


def test_belief_state_marks_current_unknown_slot_as_pending_information():
    belief = belief_from_facts(
        "lock.partial.unit",
        {
            "code": [4, 1, 0, 0],
            "visible_code": [4, 1, 0, 0],
            "known_mask": [1, 1, 0, 0],
            "unknown_mask": [0, 0, 1, 1],
            "current_index": 2,
            "sequence_length": 4,
            "remaining_steps": 6,
            "visibility_mode": "partial_hidden",
            "last_observation": {"type": "none", "slot": -1, "value": 0},
        },
    )

    assert belief.current_slot_known is False
    assert belief.pending_information_need is True
    assert belief.executable_now is False
    assert belief.current_required_value == "unknown"
    assert belief.known_slots == (0, 1)
    assert belief.unknown_slots == (2, 3)


def test_belief_state_uses_revealed_current_slot_for_execution():
    belief = belief_from_facts(
        "lock.partial.unit",
        {
            "code": [4, 1, 3, 0],
            "visible_code": [4, 1, 3, 0],
            "known_mask": [1, 1, 1, 0],
            "unknown_mask": [0, 0, 0, 1],
            "current_index": 2,
            "sequence_length": 4,
            "visibility_mode": "partial_hidden",
            "last_observation": {"type": "inspect", "slot": 2, "value": 3},
        },
    )

    assert belief.current_slot_known is True
    assert belief.pending_information_need is False
    assert belief.executable_now is True
    assert belief.current_required_value == "digit.3"
    assert belief.last_observation.value_key == "digit.3"


def test_slot_feature_export_writes_manifest_and_belief_rows(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual.jsonl"
    features = tmp_path / "features_slot.jsonl"
    export_candidate_counterfactuals(
        counterfactual,
        split="test",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        include_variants=True,
        lock_code_lengths=[3],
    )

    exported = export_slot_features(counterfactual, features)
    validated = validate_slot_features(features)
    first = json.loads(features.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(features.with_suffix(features.suffix + ".manifest.json").read_text(encoding="utf-8"))

    assert exported["schema_version"] == "neurokernel-slot-transition-v2"
    assert validated["schema_version"] == "neurokernel-slot-transition-v2"
    assert manifest["max_slots"] == 8
    assert "belief_global" in manifest["input_layout"]
    assert "slot_flat" in manifest["input_layout"]
    assert len(first["input_vector"]) == exported["input_dim"]
    assert len(first["target_vector"]) == exported["target_dim"]
    assert "belief" in first
    assert "current_slot_known" in first["belief"]


def test_slot_dataset_audit_reports_skew_and_special_signals(tmp_path: Path):
    features = _write_tiny_slot_dataset(tmp_path)

    audit = audit_slot_dataset(features)

    assert audit["schema_version"] == "neurokernel-slot-transition-v2"
    assert audit["rows"] == 6
    assert audit["candidate_groups"] == 3
    assert audit["rows_by_split_family"]["train"] == {"lock": 2, "maze": 2}
    assert audit["rows_by_split_visibility"]["test"] == {"partial_hidden": 2}
    assert audit["target_counts"]["information_gain_rows"] == 1
    assert audit["special_candidate_groups"]["post_reveal_execution_groups"] == 1


def test_balanced_slot_export_preserves_candidate_groups_and_schema(tmp_path: Path):
    features = _write_tiny_slot_dataset(tmp_path)
    balanced = tmp_path / "features_slot_balanced.jsonl"

    exported = export_balanced_slot_dataset(features, balanced, target_rows=4, seed=7)
    validated = validate_slot_features(balanced)
    rows = [json.loads(line) for line in balanced.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(balanced.with_suffix(balanced.suffix + ".manifest.json").read_text(encoding="utf-8"))

    assert exported["dataset_profile"] == "neurokernel-slot-balanced-v1"
    assert validated["schema_version"] == "neurokernel-slot-transition-v2"
    assert manifest["schema_version"] == "neurokernel-slot-transition-v2"
    assert manifest["dataset_profile"] == "neurokernel-slot-balanced-v1"
    assert manifest["balanced_train_rows"] == 4
    assert manifest["heldout_rows_preserved"] == 2
    assert len(rows) == 6
    assert sum(1 for row in rows if row["candidate_set_id"] == "g_lock") == 2
    assert sum(1 for row in rows if row["candidate_set_id"] == "g_maze") == 2
    assert sum(1 for row in rows if row["split"] == "test") == 2


def _write_tiny_slot_dataset(tmp_path: Path) -> Path:
    features = tmp_path / "features_slot.jsonl"
    rows = [
        _slot_row("train", "lock.train", "g_lock", "enter:1", "visible", [0, 0, 0, 0, 1, 0, 1, 1, 0]),
        _slot_row("train", "lock.train", "g_lock", "enter:2", "visible", [0, 0, 0, 0, 0, 0, 0, -1, 0]),
        _slot_row("train", "maze.train", "g_maze", "move:red", "visible", [0, 0, 0, 0, 1, 0, 1, 1, 0]),
        _slot_row("train", "maze.train", "g_maze", "move:blue", "visible", [0, 0, 0, 0, 0, 0, 0, -1, 0]),
        _slot_row("test", "tool.partial.test", "g_tool", "read:0", "partial_hidden", [0, 0, 0, 0, 0, 0, 0, 0, 1], current_known=False, observation_kind="none"),
        _slot_row("test", "tool.partial.test", "g_tool", "use:hammer", "partial_hidden", [0, 0, 0, 0, 1, 0, 1, 1, 0], current_known=True, observation_kind="read"),
    ]
    features.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "rows": len(rows),
        "schema_version": "neurokernel-slot-transition-v2",
        "envs": ["lock.train", "maze.train", "tool.partial.test"],
        "env_families": ["lock", "maze", "tool"],
        "splits": ["test", "train"],
        "action_vocab": ["enter:1", "enter:2", "move:blue", "move:red", "read:0", "use:hammer"],
        "input_dim": 4,
        "target_dim": 9,
        "target_layout": {
            "next_state_padded": [0, 4],
            "reward": 4,
            "done": 5,
            "local_success": 6,
            "progress_delta": 7,
            "information_gain": 8,
        },
    }
    features.with_suffix(features.suffix + ".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return features


def _slot_row(
    split: str,
    env_name: str,
    candidate_set_id: str,
    action_key: str,
    visibility: str,
    target_vector: list[float],
    *,
    current_known: bool = True,
    observation_kind: str = "none",
) -> dict:
    family = env_name.split(".", 1)[0]
    return {
        "schema_version": "neurokernel-slot-transition-v2",
        "row_index": 0,
        "episode_id": "tiny",
        "step_index": 0,
        "env_name": env_name,
        "split": split,
        "action_key": action_key,
        "candidate_set_id": candidate_set_id,
        "actual_action_score": 0.0,
        "belief": {
            "env_family": family,
            "visibility_mode": visibility,
            "known_slots": [0] if current_known else [],
            "unknown_slots": [] if current_known else [0],
            "revealed_slots": [0] if observation_kind != "none" else [],
            "current_index": 0,
            "sequence_length": 4,
            "current_slot_known": current_known,
            "current_required_value": "digit.1" if family == "lock" else "unknown",
            "last_observation": {"kind": observation_kind, "slot": 0 if observation_kind != "none" else -1, "value_key": "digit.1"},
            "pending_information_need": not current_known,
            "executable_now": current_known,
        },
        "input_vector": [0.0, 1.0, 0.0, 1.0],
        "target_vector": target_vector,
        "target_mask": [1.0] * 9,
        "source": {"candidate_set_id": candidate_set_id, "restore_id": "restore", "agent_state_id": "agent"},
    }
