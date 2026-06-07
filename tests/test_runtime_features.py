import json

from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.replay.runtime_dataset import RuntimeReplayEtlConfig, run_runtime_replay_etl
from neurokernel_seed.replay.runtime_features import (
    RUNTIME_FEATURE_SCHEMA_VERSION,
    check_runtime_feature_gates,
    export_runtime_features,
    validate_runtime_features,
)


def test_runtime_features_export_selected_action_outcome_dataset(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "목록 확인",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    run_runtime_replay_etl(RuntimeReplayEtlConfig(db_path=db, out_path=replay, min_rows=1))

    exported = export_runtime_features(replay, features, test_ratio=0.0)
    validation = validate_runtime_features(features)
    gates = check_runtime_feature_gates(features, min_rows=1)

    rows = [json.loads(line) for line in features.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads(features.with_suffix(features.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert exported["schema_version"] == RUNTIME_FEATURE_SCHEMA_VERSION
    assert validation["accepted"] is True
    assert gates["gates"]["schema_valid"]["passed"] is True
    assert gates["gates"]["targets_present"]["passed"] is True
    assert gates["ready_for_runtime_model_training"] is False
    assert rows[0]["schema_version"] == RUNTIME_FEATURE_SCHEMA_VERSION
    assert rows[0]["action_key"] == "list_artifacts"
    assert rows[0]["split"] == "train"
    assert len(rows[0]["input_vector"]) == manifest["input_dim"]
    assert len(rows[0]["target_vector"]) == manifest["target_dim"]
    assert rows[0]["target_vector"][0] == 1.0
    assert "goal_redacted" not in json.dumps(rows[0], ensure_ascii=False)


def test_runtime_features_export_candidate_rows_with_target_masks(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "inspect runtime choices",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts", "get_memory_usage"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    run_runtime_replay_etl(RuntimeReplayEtlConfig(db_path=db, out_path=replay, min_rows=1))

    exported = export_runtime_features(replay, features, test_ratio=0.0)
    rows = [json.loads(line) for line in features.read_text(encoding="utf-8").splitlines()]
    selected = next(row for row in rows if row["source"]["candidate"]["selected"])
    unselected = next(row for row in rows if not row["source"]["candidate"]["selected"])

    assert exported["rows"] == 2
    assert {row["action_key"] for row in rows} == {"list_artifacts", "get_memory_usage"}
    assert selected["source"]["candidate"]["executed"] is True
    assert selected["source"]["candidate"]["execution_result_known"] is True
    assert selected["target_mask"] == [1.0, 1.0, 1.0, 1.0]
    assert selected["target_vector"][0] == 1.0
    assert unselected["source"]["candidate"]["executed"] is False
    assert unselected["source"]["candidate"]["execution_result_known"] is False
    assert unselected["target_mask"] == [0.0, 0.0, 0.0, 0.0]
    assert unselected["target_vector"] == [0.0, 0.0, 0.0, 0.0]


def test_runtime_feature_gate_keeps_short_dataset_out_of_training_ready(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "메모리 확인",
            "target": "orangepi5",
            "allowed_actions": ["get_memory_usage"],
            "context": {},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    run_runtime_replay_etl(RuntimeReplayEtlConfig(db_path=db, out_path=replay, min_rows=1))
    export_runtime_features(replay, features, test_ratio=0.0)

    gates = check_runtime_feature_gates(features, min_rows=99)

    assert gates["passed"] is False
    assert gates["ready_for_runtime_model_training"] is False
    assert gates["gates"]["min_rows"]["passed"] is False
