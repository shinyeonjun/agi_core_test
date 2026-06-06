import json
from pathlib import Path

from neurokernel_seed.baseline import freeze_baseline
from neurokernel_seed.core.schema import Action, Prediction, WorldState
from neurokernel_seed.envs.lock_world import LockWorld
from neurokernel_seed.eval.hard_heldout import EnvCase, VisibilityPredictor, _diagnose_model_needed_trace, build_hard_eval_groups, classify_failure_trace, trace_failures_for_cases


class EchoPredictor:
    name = "echo"

    def __init__(self):
        self.last_facts = None

    def predict(self, state: WorldState, action: Action) -> Prediction:
        self.last_facts = state.facts
        return Prediction(state.vector, 0.0, False, 1.0, "echo")


class PressOnePredictor:
    name = "press-one"

    def predict(self, state: WorldState, action: Action) -> Prediction:
        reward = 1.0 if action.name == "press" and action.params.get("digit") == 1 else 0.0
        return Prediction(state.vector, reward, False, 1.0, "press one")


def test_freeze_baseline_copies_artifacts_and_manifest(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data"
    artifacts.mkdir()
    data.mkdir()
    for filename in ["world_model.onnx", "world_model.manifest.json", "world_model.pt", "world_model.metrics.json"]:
        (artifacts / filename).write_text(filename, encoding="utf-8")
    (data / "features.jsonl").write_text("{}", encoding="utf-8")
    (data / "features.jsonl.manifest.json").write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(json.dumps({"lock.test": {"success_rate": 1.0}}), encoding="utf-8")

    result = freeze_baseline(
        baseline_name="unit_baseline",
        artifacts_dir=artifacts,
        data_dir=data,
        out_dir=tmp_path / "baselines",
        evaluation_path=evaluation,
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["baseline_name"] == "unit_baseline"
    assert manifest["learned_gate_success_rate"]["lock.test"] == 1.0
    assert Path(result["out"], "world_model.onnx").exists()


def test_freeze_baseline_supports_artifact_prefix_and_hard_eval_manifest(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    data = tmp_path / "data" / "model_ready"
    artifacts.mkdir(parents=True)
    data.mkdir(parents=True)
    for filename in ["world_model_v4.onnx", "world_model_v4.manifest.json", "world_model_v4.pt", "world_model_v4.metrics.json"]:
        (artifacts / filename).write_text(filename, encoding="utf-8")
    (data / "features_v4.jsonl").write_text("{}", encoding="utf-8")
    (data / "features_v4.jsonl.manifest.json").write_text("{}", encoding="utf-8")
    evaluation = tmp_path / "hard_eval.json"
    evaluation.write_text(
        json.dumps({"groups": {"hard_length_heldout": {"aggregate": {"learned_gate_success_rate": 1.0}, "envs": {}}}}),
        encoding="utf-8",
    )

    result = freeze_baseline(
        baseline_name="v4_unit",
        artifacts_dir=artifacts,
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "baselines",
        evaluation_path=evaluation,
        artifact_prefix="world_model_v4",
        notes="unit v4 baseline",
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["artifact_prefix"] == "world_model_v4"
    assert manifest["notes"] == "unit v4 baseline"
    assert manifest["learned_gate_success_rate"]["group:hard_length_heldout"] == 1.0
    assert Path(result["out"], "world_model_v4.onnx").exists()
    assert Path(result["out"], "features_v4.jsonl").exists()


def test_hard_eval_groups_include_expected_sections():
    groups = build_hard_eval_groups()
    assert {
        "easy_test",
        "hard_value_heldout",
        "hard_length_heldout",
        "partial_hidden_smoke",
        "hard_length5_probe",
        "partial_hidden_hard",
        "full_hidden_probe",
        "model_needed_lock_trap",
        "model_needed_maze_hazard",
        "model_needed_tool_precondition",
    } <= set(groups)
    assert any(case.env.name.startswith("lock.heldout.length4") for case in groups["hard_length_heldout"])
    assert any(case.env.name.startswith("tool.heldout.length4") for case in groups["hard_length_heldout"])
    assert any(case.env.name.startswith("lock.probe.hidden") for case in groups["full_hidden_probe"])
    assert any(case.env.name.startswith("lock.probe.trap") for case in groups["model_needed_lock_trap"])
    assert any(case.env.name.startswith("maze.probe.hazard") for case in groups["model_needed_maze_hazard"])
    assert any(case.env.name.startswith("tool.probe.precondition") for case in groups["model_needed_tool_precondition"])


def test_visibility_predictor_hides_task_config_facts():
    predictor = EchoPredictor()
    wrapper = VisibilityPredictor(predictor, "hidden")
    state = WorldState("lock.test", "s0", (0.0, 0.0, 0.0), {"code": (2, 4, 1), "split": "test"}, False)

    wrapper.predict(state, Action("press", {"digit": 2}))

    assert predictor.last_facts["code"] == [0, 0, 0]
    assert predictor.last_facts["visibility_mode"] == "hidden"


def test_failure_trace_records_candidates_and_bucket():
    env = LockWorld("lock.heldout.length4.unit", "heldout", (2, 4, 1, 3), 5)

    traces = trace_failures_for_cases([EnvCase("hard_length_heldout", env)], PressOnePredictor(), episodes=1, visibility_mode="visible", max_failures_per_env=1, gate_mode="model_only")

    assert len(traces) == 1
    trace = traces[0]
    assert trace["success"] is False
    assert trace["failure"]["primary_bucket"] in {"wrong_current_index", "repeated_action", "max_steps_issue", "length_overrun", "padding/schema_issue"}
    first_step = trace["episode_trace"][0]
    assert first_step["expected_action"] == "press(digit=2)"
    assert first_step["chosen_action_key"] == "press(digit=1)"
    assert first_step["candidate_actions"]
    assert "predicted" in first_step["candidate_actions"][0]
    assert "counterfactual_actual" in first_step["candidate_actions"][0]
    assert "score_breakdown" in first_step["candidate_actions"][0]
    assert "model_score" in first_step["candidate_actions"][0]["score_breakdown"]["model_only"]


def test_model_needed_diagnosis_splits_veto_and_transition_buckets():
    trace = {
        "env_name": "maze.probe.hazard.red",
        "group": "model_needed_maze_hazard",
        "success": False,
        "steps": 1,
        "max_steps": 7,
        "episode_trace": [
            {
                "step": 0,
                "gate_mode": "hybrid_veto",
                "chosen_action_key": "move(door=\"red\")",
                "expected_action": "inspect",
                "visible_state_facts": {
                    "model_needed_kind": "maze_hazard",
                    "hazard_active": True,
                    "compatible_but_bad_action": True,
                },
                "candidate_actions": [
                    {
                        "action_key": "move(door=\"red\")",
                        "model_veto": 0.0,
                        "compatibility": {"reveals_information": False},
                        "score_breakdown": {"hybrid_veto": {"final_score": 2.0, "model_veto": 0.0}},
                        "predicted": {"reward": 0.5, "progress_delta": 0.5, "local_success": 0.8},
                        "counterfactual_actual": {"reward": -0.35, "progress_delta": 0.0, "success": False},
                        "candidate_diagnosis": {
                            "actual_bad_outcome": True,
                            "predicted_bad_outcome": False,
                            "transition_undertrained_signal": True,
                            "veto_not_triggered_signal": True,
                        },
                    },
                    {
                        "action_key": "inspect",
                        "model_veto": 0.0,
                        "compatibility": {"reveals_information": True},
                        "score_breakdown": {"hybrid_veto": {"final_score": 1.0, "model_veto": 0.0}},
                        "predicted": {"reward": 0.1, "progress_delta": 0.0, "local_success": 0.2},
                        "counterfactual_actual": {"reward": 0.12, "progress_delta": 0.0, "success": False},
                        "candidate_diagnosis": {
                            "actual_bad_outcome": False,
                            "predicted_bad_outcome": False,
                            "transition_undertrained_signal": False,
                            "veto_not_triggered_signal": False,
                        },
                    },
                ],
            }
        ],
    }

    diagnosis = _diagnose_model_needed_trace(trace)
    buckets = {diagnosis["primary_bucket"], *diagnosis["secondary_buckets"]}

    assert "transition_undertrained" in buckets
    assert "veto_not_triggered" in buckets
    assert "veto_harm" in buckets
    assert "veto_too_weak" in buckets


def test_partial_hidden_classifier_separates_reveal_from_bad_execution():
    trace = {
        "group": "partial_hidden_hard",
        "env_name": "lock.probe.partial.length4.unit",
        "success": False,
        "steps": 5,
        "max_steps": 5,
        "episode_trace": [
            {
                "step": 0,
                "chosen_action_key": "inspect",
                "expected_action": "inspect",
                "actual_result": {"information_gain": 1.0},
                "candidate_actions": [{"action_key": "inspect"}],
                "visible_task_config": {"code": [2, 0, 0], "current_index": 1},
                "visible_state_facts": {"current_slot_known": False, "known_mask": [1, 0, 0]},
            },
            {
                "step": 1,
                "chosen_action_key": "press(digit=1)",
                "expected_action": "press(digit=4)",
                "actual_result": {"information_gain": 0.0},
                "candidate_actions": [{"action_key": "inspect"}, {"action_key": "press(digit=1)"}],
                "visible_task_config": {"code": [2, 4, 0], "current_index": 1},
                "visible_state_facts": {"current_slot_known": True, "known_mask": [1, 1, 0]},
            },
        ],
    }

    failure = classify_failure_trace(trace)

    assert failure["primary_bucket"] == "information_collected_but_execution_failed"
    assert "acted_before_information" not in [failure["primary_bucket"], *failure["secondary_buckets"]]


def test_length_classifier_marks_lock_length_schema_limit_first():
    trace = {
        "group": "hard_length5_probe",
        "env_name": "lock.probe.length5.unit",
        "success": False,
        "steps": 10,
        "max_steps": 10,
        "final_state": {"sequence_length": 5, "facts": {"code": [1, 2, 3, 4, 1]}},
        "episode_trace": [
            {
                "step": 0,
                "chosen_action_key": "press(digit=1)",
                "expected_action": "press(digit=1)",
                "state_vector_decoded": {"current_index": 0, "sequence_length": 5},
            },
            {
                "step": 1,
                "chosen_action_key": "reset",
                "expected_action": "press(digit=2)",
                "state_vector_decoded": {"current_index": 1, "sequence_length": 5},
            },
        ],
    }

    failure = classify_failure_trace(trace)

    assert failure["primary_bucket"] == "lock_length_schema_limit"
    assert "wrong_current_index" in failure["secondary_buckets"]
