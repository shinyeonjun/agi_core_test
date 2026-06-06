from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurokernel_seed.core.gate import GateMode
from neurokernel_seed.eval.hard_heldout import eval_hard_heldout, trace_hard_failures


BASELINE_GROUP_THRESHOLDS = {
    "easy_test": 1.0,
    "hard_value_heldout": 1.0,
    "hard_length_heldout": 1.0,
    "partial_hidden_smoke": 1.0,
}

PROBE_GROUP_THRESHOLDS = {
    "hard_length5_probe": 0.85,
    "partial_hidden_hard": 0.65,
    "full_hidden_probe": 0.85,
    "model_needed_lock_trap": 0.85,
    "model_needed_maze_hazard": 0.85,
    "model_needed_tool_precondition": 0.85,
}

KNOWN_DIAGNOSTIC_BUCKETS = {
    "lock_length_schema_limit",
    "information_collected_but_execution_failed",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 5
    trace_episodes: int = 3
    max_failures_per_env: int = 3
    strict: bool = False
    gate_mode: GateMode = "hybrid"


def run_benchmark(
    model_path: str | Path,
    *,
    out_dir: str | Path,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BenchmarkConfig()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_path = output_dir / "hard_heldout_eval.json"
    trace_path = output_dir / "hard_failure_traces.json"
    summary_path = output_dir / "benchmark_summary.json"

    eval_payload = eval_hard_heldout(model_path, episodes=cfg.episodes, gate_mode=cfg.gate_mode, out=eval_path)
    trace_payload = trace_hard_failures(
        model_path,
        groups=list(PROBE_GROUP_THRESHOLDS),
        episodes=cfg.trace_episodes,
        max_failures_per_env=cfg.max_failures_per_env,
        gate_mode=cfg.gate_mode,
        out=trace_path,
    )
    summary = summarize_benchmark(
        eval_payload,
        trace_payload,
        model_path=str(model_path),
        config=cfg,
        eval_path=str(eval_path),
        trace_path=str(trace_path),
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary["out"] = str(summary_path)
    return summary


def summarize_benchmark(
    eval_payload: dict[str, Any],
    trace_payload: dict[str, Any],
    *,
    model_path: str,
    config: BenchmarkConfig,
    eval_path: str | None = None,
    trace_path: str | None = None,
) -> dict[str, Any]:
    group_results = _group_results(eval_payload)
    baseline_gates = _threshold_gates(group_results, BASELINE_GROUP_THRESHOLDS)
    probe_gates = _threshold_gates(group_results, PROBE_GROUP_THRESHOLDS)
    trace_gate = _trace_gate(trace_payload)
    baseline_passed = all(gate["passed"] for gate in baseline_gates.values())
    probe_passed = all(gate["passed"] for gate in probe_gates.values())
    diagnostic_passed = bool(trace_gate["passed"])
    passed = baseline_passed and diagnostic_passed and (probe_passed if config.strict else True)

    return {
        "benchmark_version": "neurokernel-benchmark-v1",
        "model": model_path,
        "config": {
            "episodes": config.episodes,
            "trace_episodes": config.trace_episodes,
            "max_failures_per_env": config.max_failures_per_env,
            "strict": config.strict,
            "gate_mode": config.gate_mode,
        },
        "source_files": {"hard_eval": eval_path, "failure_traces": trace_path},
        "passed": passed,
        "baseline_passed": baseline_passed,
        "probe_passed": probe_passed,
        "diagnostic_passed": diagnostic_passed,
        "gates": {
            "baseline": baseline_gates,
            "probe": probe_gates,
            "failure_trace": trace_gate,
        },
        "group_results": group_results,
        "next_actions": _next_actions(group_results, trace_payload),
    }


def _group_results(eval_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for group_name, payload in eval_payload.get("groups", {}).items():
        aggregate = payload.get("aggregate", {})
        envs = payload.get("envs", {})
        failing_envs = {
            env_name: env_payload.get("learned_gate", {}).get("success_rate")
            for env_name, env_payload in envs.items()
            if float(env_payload.get("learned_gate", {}).get("success_rate", 0.0)) < 1.0
        }
        results[group_name] = {
            "visibility_mode": payload.get("visibility_mode"),
            "learned_gate_success_rate": float(aggregate.get("learned_gate_success_rate", 0.0)),
            "heuristic_baseline_success_rate": float(aggregate.get("heuristic_baseline_success_rate", 0.0)),
            "random_baseline_success_rate": float(aggregate.get("random_baseline_success_rate", 0.0)),
            "first_step_accuracy": float(aggregate.get("first_step_accuracy", 0.0)),
            "wrong_action_rate": float(aggregate.get("wrong_action_rate", 0.0)),
            "oracle_step_regret": float(aggregate.get("oracle_step_regret", 0.0)),
            "info_acquisition_success_rate": float(aggregate.get("info_acquisition_success_rate", 0.0)),
            "hidden_execution_before_reveal_rate": float(aggregate.get("hidden_execution_before_reveal_rate", 0.0)),
            "summarize_early_wrong_rate": float(aggregate.get("summarize_early_wrong_rate", 0.0)),
            "compatibility_override_rate": float(aggregate.get("compatibility_override_rate", 0.0)),
            "compatibility_fix_rate": float(aggregate.get("compatibility_fix_rate", 0.0)),
            "compatibility_harm_rate": float(aggregate.get("compatibility_harm_rate", 0.0)),
            "required_action_match_rate": float(aggregate.get("required_action_match_rate", 0.0)),
            "unknown_reveal_choice_rate": float(aggregate.get("unknown_reveal_choice_rate", 0.0)),
            "model_veto_activation_rate": float(aggregate.get("model_veto_activation_rate", 0.0)),
            "model_veto_fix_rate": float(aggregate.get("model_veto_fix_rate", 0.0)),
            "model_veto_harm_rate": float(aggregate.get("model_veto_harm_rate", 0.0)),
            "compatible_but_bad_action_avoidance_rate": float(aggregate.get("compatible_but_bad_action_avoidance_rate", 0.0)),
            "post_setup_correct_action_pred_good_rate": float(aggregate.get("post_setup_correct_action_pred_good_rate", 0.0)),
            "post_setup_correct_action_veto_rate": float(aggregate.get("post_setup_correct_action_veto_rate", 0.0)),
            "veto_false_positive_rate": float(aggregate.get("veto_false_positive_rate", 0.0)),
            "veto_true_positive_rate": float(aggregate.get("veto_true_positive_rate", 0.0)),
            "maze_post_setup_target_move_veto_rate": float(aggregate.get("maze_post_setup_target_move_veto_rate", 0.0)),
            "maze_wrong_door_pred_good_rate": float(aggregate.get("maze_wrong_door_pred_good_rate", 0.0)),
            "maze_target_color_grounding_accuracy": float(aggregate.get("maze_target_color_grounding_accuracy", 0.0)),
            "maze_hazard_clear_to_execute_success_rate": float(aggregate.get("maze_hazard_clear_to_execute_success_rate", 0.0)),
            "failing_envs": failing_envs,
        }
    return results


def _threshold_gates(group_results: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    for group_name, minimum in thresholds.items():
        actual = group_results.get(group_name, {}).get("learned_gate_success_rate")
        gates[group_name] = {
            "minimum": minimum,
            "actual": actual,
            "passed": actual is not None and float(actual) >= minimum,
        }
    return gates


def _trace_gate(trace_payload: dict[str, Any]) -> dict[str, Any]:
    summary = trace_payload.get("summary", {})
    buckets = set(summary.get("by_primary_bucket", {}))
    unknown_buckets = sorted(buckets - KNOWN_DIAGNOSTIC_BUCKETS)
    return {
        "known_buckets": sorted(KNOWN_DIAGNOSTIC_BUCKETS),
        "actual_buckets": sorted(buckets),
        "unknown_buckets": unknown_buckets,
        "failure_count": int(summary.get("failure_count", 0)),
        "passed": not unknown_buckets,
    }


def _next_actions(group_results: dict[str, dict[str, Any]], trace_payload: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if group_results.get("hard_length5_probe", {}).get("learned_gate_success_rate", 1.0) < PROBE_GROUP_THRESHOLDS["hard_length5_probe"]:
        actions.append("Implement LockWorld length >4 schema support or shared slot encoding.")
    if group_results.get("partial_hidden_hard", {}).get("learned_gate_success_rate", 1.0) < PROBE_GROUP_THRESHOLDS["partial_hidden_hard"]:
        actions.append("Add curriculum data for partial-hidden LockWorld execution after reveal.")
    if group_results.get("full_hidden_probe", {}).get("learned_gate_success_rate", 1.0) < PROBE_GROUP_THRESHOLDS["full_hidden_probe"]:
        actions.append("Add full-hidden LockWorld reveal/execution traces and last_observation-focused features.")
    for group_name, label in (
        ("model_needed_lock_trap", "LockTrap"),
        ("model_needed_maze_hazard", "MazeHazard"),
        ("model_needed_tool_precondition", "ToolPrecondition"),
    ):
        if group_results.get(group_name, {}).get("learned_gate_success_rate", 1.0) < PROBE_GROUP_THRESHOLDS[group_name]:
            actions.append(f"Increase {label} model-needed probe data and inspect model-veto false negatives.")
    buckets = trace_payload.get("summary", {}).get("by_primary_bucket", {})
    if buckets.get("information_collected_but_execution_failed", 0) > 0:
        actions.append("Evaluate whether last_observation/current_slot_known features are visible to the model and used by the gate.")
    if buckets.get("lock_length_schema_limit", 0) > 0:
        actions.append("Decide fixed flat slot extension versus object/slot representation before retraining.")
    return actions
