import json
import os
import subprocess
import sys
from pathlib import Path

from neurokernel_seed.agents.gated import GatedAgent
from neurokernel_seed.envs.registry import make_env
from neurokernel_seed.eval.evaluator import Evaluator
from neurokernel_seed.predictors.symbolic import SymbolicPredictor
from neurokernel_seed.replay.counterfactual import export_candidate_counterfactuals, validate_counterfactual
from neurokernel_seed.replay.dataset import export_flat_features, inspect_replay_features, validate_flat_features
from neurokernel_seed.replay.exporter import ReplayExporter
from neurokernel_seed.replay.slot_dataset import audit_maze_grounding, audit_slot_dataset, check_slot_dataset_gates, export_slot_features, merge_slot_feature_files, validate_slot_features
from neurokernel_seed.replay.validator import validate_replay
from neurokernel_seed.storage.sqlite_logger import SQLiteEpisodeLogger


def test_sqlite_logger_and_replay_export(tmp_path: Path):
    db = tmp_path / "episodes.db"
    replay = tmp_path / "replay" / "events.jsonl"
    with SQLiteEpisodeLogger(db) as logger:
        result = Evaluator(logger).run(make_env("lock.test"), GatedAgent(SymbolicPredictor()), episodes=1)
        assert result.success_rate == 1.0
        assert logger.episode_count() == 1
        assert logger.event_count() > 0
    meta = ReplayExporter(db).export_jsonl(replay)
    assert meta["rows"] > 0
    assert meta["schema_version"] == "neurokernel-replay-v2"
    checked = validate_replay(replay)
    assert checked["rows"] == meta["rows"]
    first = json.loads(replay.read_text(encoding="utf-8").splitlines()[0])
    assert first["schema_version"] == "neurokernel-replay-v2"
    assert first["observation_vector"] == first["actual_next_state_vector"]
    assert "next_state_facts" in first
    assert first["env_metadata"]["name"] == "lock.test"
    assert first["split"] == "test"


def test_flat_feature_export_is_model_ready(tmp_path: Path):
    db = tmp_path / "episodes.db"
    replay = tmp_path / "replay" / "events.jsonl"
    features = tmp_path / "replay" / "features.jsonl"
    with SQLiteEpisodeLogger(db) as logger:
        Evaluator(logger).run(make_env("lock.test"), GatedAgent(SymbolicPredictor()), episodes=1)
        Evaluator(logger).run(make_env("maze.test"), GatedAgent(SymbolicPredictor()), episodes=1)
    ReplayExporter(db).export_jsonl(replay)
    inspected = inspect_replay_features(replay)
    assert inspected["schema_version"] == "neurokernel-flat-transition-v2"
    assert inspected["max_state_dim"] == 4
    assert "move(door=\"green\")" in inspected["action_vocab"]
    assert inspected["input_layout"]["env_family_onehot"] == [0, 2]
    assert "maze.target.blue" in inspected["task_feature_names"]
    exported = export_flat_features(replay, features)
    checked = validate_flat_features(features)
    assert checked["rows"] == exported["rows"]
    first = json.loads(features.read_text(encoding="utf-8").splitlines()[0])
    assert len(first["input_vector"]) == exported["input_dim"]
    assert len(first["target_vector"]) == exported["target_dim"]
    assert len(first["target_mask"]) == exported["target_dim"]
    assert exported["target_layout"]["success"] == exported["target_dim"] - 1


def test_collect_dataset_cli_exports_replay_and_features(tmp_path: Path):
    db = tmp_path / "model_ready.db"
    replay = tmp_path / "model_ready" / "replay.jsonl"
    features = tmp_path / "model_ready" / "features.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "neurokernel_seed.cli",
            "collect-dataset",
            "--split",
            "test",
            "--episodes",
            "1",
            "--agents",
            "random",
            "heuristic",
            "gated",
            "--db",
            str(db),
            "--replay-out",
            str(replay),
            "--features-out",
            str(features),
        ],
        check=True,
        text=True,
        capture_output=True,
        env=os.environ | {"PYTHONPATH": "src"},
    )
    payload = json.loads(result.stdout)
    assert payload["replay_validation"]["rows"] > 0
    assert payload["feature_validation"]["rows"] == payload["replay_validation"]["rows"]
    assert payload["features"]["input_dim"] > payload["features"]["target_dim"]
    assert replay.exists()
    assert features.exists()


def test_counterfactual_export_and_feature_v3(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual.jsonl"
    features = tmp_path / "features.jsonl"
    meta = export_candidate_counterfactuals(counterfactual, split="test", episodes=1, agents=["heuristic"], include_variants=False)
    checked = validate_counterfactual(counterfactual)
    assert checked["rows"] == meta["rows"]
    assert checked["unique_states"] > 0
    exported = export_flat_features(counterfactual, features)
    validated = validate_flat_features(features)
    assert validated["schema_version"] == "neurokernel-flat-transition-v4"
    assert exported["target_layout"]["local_success"] == exported["max_state_dim"] + 2
    assert exported["target_layout"]["progress_delta"] == exported["max_state_dim"] + 3
    assert exported["target_layout"]["information_gain"] == exported["max_state_dim"] + 4
    first = json.loads(features.read_text(encoding="utf-8").splitlines()[0])
    assert first["candidate_set_id"]
    assert first["actual_action_score"] is not None
    manifest = json.loads(features.with_suffix(features.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert "known_mask.0" in manifest["task_feature_names"]
    assert "action.meta.terminal_only" in manifest["task_feature_names"]
    assert "current_required_action.unknown" in manifest["task_feature_names"]


def test_counterfactual_variants_expand_lock_coverage(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual_v31.jsonl"
    features = tmp_path / "features_v31.jsonl"
    meta = export_candidate_counterfactuals(counterfactual, split="all", episodes=0, agents=["heuristic"], include_rollouts=False, lock_code_lengths=[3, 4])
    checked = validate_counterfactual(counterfactual)
    exported = export_flat_features(counterfactual, features)
    validated = validate_flat_features(features)
    assert checked["rows"] == meta["rows"]
    assert checked["unique_states"] >= 6_000
    assert meta["rows"] >= 30_000
    assert "lock.train.canon.3.0001" in exported["envs"]
    assert exported["schema_version"] == "neurokernel-flat-transition-v4"
    assert any(env.startswith("lock.train.partial_hidden") for env in exported["envs"])
    assert any(env.startswith("tool.train.partial_hidden") for env in exported["envs"])
    assert validated["rows"] == meta["rows"]


def test_slot_v2_curriculum_generates_balanced_visibility_coverage(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual_slot_v2.jsonl"
    meta = export_candidate_counterfactuals(
        counterfactual,
        split="all",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        curriculum_profile="slot_v2",
        curriculum_target_groups_per_family=90,
    )
    checked = validate_counterfactual(counterfactual)
    rows = [json.loads(line) for line in counterfactual.read_text(encoding="utf-8").splitlines()]
    train_families = {row["env_family"] for row in rows if row["split"] == "train"}
    train_visibility = {row["state_facts"]["visibility_mode"] for row in rows if row["split"] == "train"}
    information_rows = [row for row in rows if row["information_gain"] > 0]
    post_reveal_rows = [
        row
        for row in rows
        if row["state_facts"].get("current_slot_known")
        and row["state_facts"].get("last_observation", {}).get("type") != "none"
        and row["candidate_action"]["name"] not in {"inspect", "search", "read"}
    ]

    assert checked["rows"] == meta["rows"]
    assert meta["curriculum_profile"] == "slot_v2"
    assert train_families == {"lock", "maze", "tool"}
    assert {"partial_hidden", "hidden"}.issubset(train_visibility)
    assert information_rows
    assert post_reveal_rows


def test_model_needed_curriculum_and_slot_merge(tmp_path: Path):
    slot_v2_counterfactual = tmp_path / "counterfactual_slot_v2.jsonl"
    model_needed_counterfactual = tmp_path / "counterfactual_model_needed.jsonl"
    slot_v2_features = tmp_path / "features_slot_v2.jsonl"
    model_needed_features = tmp_path / "features_model_needed.jsonl"
    merged_features = tmp_path / "features_merged.jsonl"

    export_candidate_counterfactuals(
        slot_v2_counterfactual,
        split="all",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        curriculum_profile="slot_v2",
        curriculum_target_groups_per_family=12,
    )
    model_needed_meta = export_candidate_counterfactuals(
        model_needed_counterfactual,
        split="all",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        curriculum_profile="slot_model_needed_v1",
        curriculum_target_groups_per_family=12,
    )
    rows = [json.loads(line) for line in model_needed_counterfactual.read_text(encoding="utf-8").splitlines()]
    kinds = {row["state_facts"].get("model_needed_kind") for row in rows}
    bad_compatible_rows = [row for row in rows if row["state_facts"].get("compatible_but_bad_action")]

    assert model_needed_meta["curriculum_profile"] == "slot_model_needed_v1"
    assert {"lock_trap", "maze_hazard", "tool_precondition"} <= kinds
    assert bad_compatible_rows

    export_slot_features(slot_v2_counterfactual, slot_v2_features)
    export_slot_features(model_needed_counterfactual, model_needed_features)
    merged = merge_slot_feature_files([slot_v2_features, model_needed_features], merged_features)
    checked = validate_slot_features(merged_features)
    manifest = json.loads(merged_features.with_suffix(merged_features.suffix + ".manifest.json").read_text(encoding="utf-8"))

    assert checked["rows"] == merged["rows"]
    assert manifest["dataset_profile"] == "neurokernel-slot-merged-v1"
    assert any(".model_needed." in env_name for env_name in manifest["envs"])


def test_model_needed_v2_curriculum_has_post_setup_execution_audit(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual_model_needed_v2.jsonl"
    features = tmp_path / "features_model_needed_v2.jsonl"

    meta = export_candidate_counterfactuals(
        counterfactual,
        split="all",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        curriculum_profile="slot_model_needed_v2",
        curriculum_target_groups_per_family=30,
    )
    rows = [json.loads(line) for line in counterfactual.read_text(encoding="utf-8").splitlines()]
    post_setup_rows = [row for row in rows if row.get("post_setup_execution")]
    positive_post_setup_rows = [row for row in post_setup_rows if row["progress_delta"] > 0 or row["local_success"] > 0]

    assert meta["curriculum_profile"] == "slot_model_needed_v2"
    assert post_setup_rows
    assert positive_post_setup_rows
    assert any(row["state_facts"].get("setup_state") == "post_setup_done" for row in rows)

    export_slot_features(counterfactual, features)
    audit = audit_slot_dataset(features)

    assert audit["post_setup_counts"]["post_setup_execution_rows"] > 0
    assert audit["post_setup_counts"]["post_setup_correct_action_positive_rows"] > 0
    assert audit["special_candidate_groups"]["post_setup_execution_groups"] > 0
    gates = check_slot_dataset_gates(
        features,
        post_setup_execution_group_min_ratio=0.01,
        post_setup_positive_group_min_ratio=0.01,
        post_setup_correct_action_group_min_ratio=0.01,
    )

    assert gates["gates"]["post_setup_execution_group_min_ratio"]["passed"]
    assert gates["gates"]["post_setup_positive_group_min_ratio"]["passed"]
    assert gates["gates"]["post_setup_correct_action_group_min_ratio"]["passed"]


def test_model_needed_v3_curriculum_passes_maze_grounding_audit(tmp_path: Path):
    counterfactual = tmp_path / "counterfactual_model_needed_v3.jsonl"
    features = tmp_path / "features_model_needed_v3.jsonl"

    meta = export_candidate_counterfactuals(
        counterfactual,
        split="all",
        episodes=0,
        agents=["heuristic"],
        include_rollouts=False,
        curriculum_profile="slot_model_needed_v3",
        curriculum_target_groups_per_family=45,
    )
    export_slot_features(counterfactual, features)
    maze_audit = audit_maze_grounding(features)

    assert meta["curriculum_profile"] == "slot_model_needed_v3"
    assert maze_audit["passed"]
    assert maze_audit["counts"]["post_setup_target_move_positive_rows"] > 0
    for color in ("red", "green", "blue"):
        assert maze_audit["by_target_color"][color]["post_setup_target_move_positive_rows"] > 0
