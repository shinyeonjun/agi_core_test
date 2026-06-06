from neurokernel_seed.eval.benchmark import BenchmarkConfig, summarize_benchmark
from neurokernel_seed.eval.gate_ablation import GateAblationConfig, summarize_gate_ablation


def test_benchmark_passes_normal_mode_when_baseline_passes_and_probe_failures_are_known():
    eval_payload = {
        "groups": {
            "easy_test": _group(1.0),
            "hard_value_heldout": _group(1.0),
            "hard_length_heldout": _group(1.0),
            "partial_hidden_smoke": _group(1.0),
            "hard_length5_probe": _group(0.5),
            "partial_hidden_hard": _group(0.6),
            "full_hidden_probe": _group(0.8),
        }
    }
    trace_payload = {"summary": {"failure_count": 15, "by_primary_bucket": {"lock_length_schema_limit": 6, "information_collected_but_execution_failed": 9}}}

    result = summarize_benchmark(eval_payload, trace_payload, model_path="model.onnx", config=BenchmarkConfig(strict=False))

    assert result["passed"] is True
    assert result["baseline_passed"] is True
    assert result["probe_passed"] is False
    assert result["diagnostic_passed"] is True
    assert result["gates"]["probe"]["hard_length5_probe"]["passed"] is False
    assert result["next_actions"]


def test_benchmark_fails_strict_mode_when_probe_targets_are_missed():
    eval_payload = {
        "groups": {
            "easy_test": _group(1.0),
            "hard_value_heldout": _group(1.0),
            "hard_length_heldout": _group(1.0),
            "partial_hidden_smoke": _group(1.0),
            "hard_length5_probe": _group(0.5),
            "partial_hidden_hard": _group(0.6),
            "full_hidden_probe": _group(0.8),
        }
    }
    trace_payload = {"summary": {"failure_count": 15, "by_primary_bucket": {"lock_length_schema_limit": 6}}}

    result = summarize_benchmark(eval_payload, trace_payload, model_path="model.onnx", config=BenchmarkConfig(strict=True))

    assert result["passed"] is False
    assert result["baseline_passed"] is True
    assert result["probe_passed"] is False


def test_benchmark_fails_when_unknown_failure_bucket_appears():
    eval_payload = {
        "groups": {
            "easy_test": _group(1.0),
            "hard_value_heldout": _group(1.0),
            "hard_length_heldout": _group(1.0),
            "partial_hidden_smoke": _group(1.0),
            "hard_length5_probe": _group(1.0),
            "partial_hidden_hard": _group(1.0),
            "full_hidden_probe": _group(1.0),
        }
    }
    trace_payload = {"summary": {"failure_count": 1, "by_primary_bucket": {"new_weird_failure": 1}}}

    result = summarize_benchmark(eval_payload, trace_payload, model_path="model.onnx", config=BenchmarkConfig(strict=False))

    assert result["passed"] is False
    assert result["diagnostic_passed"] is False
    assert result["gates"]["failure_trace"]["unknown_buckets"] == ["new_weird_failure"]


def test_gate_ablation_marks_prior_dominated_when_prior_matches_hybrid():
    summaries = {
        "model_only": _summary({"easy_test": 0.333, "hard_length5_probe": 0.0}, passed=False),
        "prior_only": _summary({"easy_test": 1.0, "hard_length5_probe": 1.0}, passed=True),
        "hybrid": _summary({"easy_test": 1.0, "hard_length5_probe": 1.0}, passed=True),
        "hybrid_veto": _summary({"easy_test": 1.0, "hard_length5_probe": 1.0}, passed=True),
    }

    result = summarize_gate_ablation("model.onnx", summaries, config=GateAblationConfig(episodes=3, trace_episodes=1, max_failures_per_env=1, strict=True))

    assert result["interpretation"]["verdict"] == "prior_dominated"
    assert result["aggregate"]["hybrid_gain_over_model"] > 0.0
    assert result["aggregate"]["hybrid_gain_over_prior"] == 0.0
    assert result["aggregate"]["groups_with_model_needed_signal"] == []
    assert result["by_group"]["easy_test"]["fixed_by_prior_or_hybrid"] is True


def test_gate_ablation_splits_model_needed_signal_by_family():
    summaries = {
        "model_only": _summary({"model_needed_lock_trap": 0.8, "model_needed_maze_hazard": 0.8, "model_needed_tool_precondition": 0.8}, passed=False),
        "prior_only": _summary({"model_needed_lock_trap": 0.3, "model_needed_maze_hazard": 0.4, "model_needed_tool_precondition": 0.5}, passed=False),
        "hybrid": _summary({"model_needed_lock_trap": 0.3, "model_needed_maze_hazard": 0.4, "model_needed_tool_precondition": 0.5}, passed=False),
        "hybrid_veto": _summary({"model_needed_lock_trap": 1.0, "model_needed_maze_hazard": 0.9, "model_needed_tool_precondition": 0.8}, passed=True),
    }

    result = summarize_gate_ablation("model.onnx", summaries, config=GateAblationConfig(episodes=3, trace_episodes=1, max_failures_per_env=1, strict=True))

    assert result["interpretation"]["verdict"] == "hybrid_adds_value"
    assert result["aggregate"]["model_needed_signal_group_count"] == 3
    assert result["aggregate"]["model_needed_signal_by_group"] == {
        "model_needed_lock_trap": True,
        "model_needed_maze_hazard": True,
        "model_needed_tool_precondition": True,
    }
    assert result["aggregate"]["hybrid_veto_gain_over_prior_by_group"]["model_needed_lock_trap"] == 0.7
    assert result["aggregate"]["prior_only_failure_rate_by_group"]["model_needed_tool_precondition"] == 0.5


def _group(success_rate: float):
    return {
        "aggregate": {
            "learned_gate_success_rate": success_rate,
            "heuristic_baseline_success_rate": 1.0,
            "random_baseline_success_rate": 0.0,
            "first_step_accuracy": success_rate,
            "wrong_action_rate": 1.0 - success_rate,
            "oracle_step_regret": 0.0,
            "info_acquisition_success_rate": success_rate,
            "hidden_execution_before_reveal_rate": 0.0,
            "summarize_early_wrong_rate": 0.0,
        },
        "envs": {},
        "visibility_mode": "visible",
    }


def _summary(success_by_group: dict[str, float], *, passed: bool):
    return {
        "passed": passed,
        "baseline_passed": passed,
        "probe_passed": passed,
        "group_results": {
            group: {
                "learned_gate_success_rate": success,
                "oracle_step_regret": 1.0 - success,
                "wrong_action_rate": 1.0 - success,
                "compatibility_override_rate": 0.0,
                "compatibility_fix_rate": 0.0,
                "compatibility_harm_rate": 0.0,
                "required_action_match_rate": success,
                "unknown_reveal_choice_rate": success,
                "model_veto_activation_rate": 0.0,
                "model_veto_fix_rate": 0.0,
                "model_veto_harm_rate": 0.0,
                "compatible_but_bad_action_avoidance_rate": success,
            }
            for group, success in success_by_group.items()
        },
        "source_files": {},
        "out": "summary.json",
    }
