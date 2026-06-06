from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurokernel_seed.core.gate import GateMode
from neurokernel_seed.eval.benchmark import BenchmarkConfig, run_benchmark

GATE_MODES: tuple[GateMode, ...] = ("model_only", "prior_only", "hybrid", "hybrid_veto")


@dataclass(frozen=True)
class GateAblationConfig:
    episodes: int = 50
    trace_episodes: int = 10
    max_failures_per_env: int = 10
    strict: bool = True


def run_gate_ablation(
    model_path: str | Path,
    *,
    out_dir: str | Path,
    config: GateAblationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GateAblationConfig()
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in GATE_MODES:
        summary = run_benchmark(
            model_path,
            out_dir=output_dir / mode,
            config=BenchmarkConfig(
                episodes=cfg.episodes,
                trace_episodes=cfg.trace_episodes,
                max_failures_per_env=cfg.max_failures_per_env,
                strict=cfg.strict,
                gate_mode=mode,
            ),
        )
        mode_summaries[mode] = summary
    report = summarize_gate_ablation(str(model_path), mode_summaries, config=cfg)
    report_path = output_dir / "gate_ablation_v4_4.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report["out"] = str(report_path)
    return report


def summarize_gate_ablation(
    model_path: str,
    mode_summaries: dict[str, dict[str, Any]],
    *,
    config: GateAblationConfig,
) -> dict[str, Any]:
    groups = sorted(
        {
            group_name
            for summary in mode_summaries.values()
            for group_name in summary.get("group_results", {})
        }
    )
    by_group = {group: _group_comparison(group, mode_summaries) for group in groups}
    aggregate = _aggregate_comparison(by_group)
    return {
        "ablation_version": "neurokernel-gate-ablation-v1",
        "model": model_path,
        "config": {
            "episodes": config.episodes,
            "trace_episodes": config.trace_episodes,
            "max_failures_per_env": config.max_failures_per_env,
            "strict": config.strict,
            "gate_modes": list(GATE_MODES),
        },
        "mode_passed": {mode: bool(summary.get("passed")) for mode, summary in mode_summaries.items()},
        "mode_probe_passed": {mode: bool(summary.get("probe_passed")) for mode, summary in mode_summaries.items()},
        "mode_baseline_passed": {mode: bool(summary.get("baseline_passed")) for mode, summary in mode_summaries.items()},
        "aggregate": aggregate,
        "by_group": by_group,
        "interpretation": _interpretation(aggregate, mode_summaries),
        "source_files": {
            mode: summary.get("source_files", {}) | {"summary": summary.get("out")}
            for mode, summary in mode_summaries.items()
        },
    }


def _group_comparison(group: str, mode_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = {
        mode: _metrics_for_group(summary, group)
        for mode, summary in mode_summaries.items()
    }
    success = {mode: values[mode]["learned_gate_success_rate"] for mode in GATE_MODES}
    regret = {mode: values[mode]["oracle_step_regret"] for mode in GATE_MODES}
    wrong = {mode: values[mode]["wrong_action_rate"] for mode in GATE_MODES}
    return {
        "success_rate": success,
        "oracle_step_regret": regret,
        "wrong_action_rate": wrong,
        "compatibility_override_rate": {mode: values[mode]["compatibility_override_rate"] for mode in GATE_MODES},
        "compatibility_fix_rate": {mode: values[mode]["compatibility_fix_rate"] for mode in GATE_MODES},
        "compatibility_harm_rate": {mode: values[mode]["compatibility_harm_rate"] for mode in GATE_MODES},
        "required_action_match_rate": {mode: values[mode]["required_action_match_rate"] for mode in GATE_MODES},
        "unknown_reveal_choice_rate": {mode: values[mode]["unknown_reveal_choice_rate"] for mode in GATE_MODES},
        "model_veto_activation_rate": {mode: values[mode]["model_veto_activation_rate"] for mode in GATE_MODES},
        "model_veto_fix_rate": {mode: values[mode]["model_veto_fix_rate"] for mode in GATE_MODES},
        "model_veto_harm_rate": {mode: values[mode]["model_veto_harm_rate"] for mode in GATE_MODES},
        "compatible_but_bad_action_avoidance_rate": {mode: values[mode]["compatible_but_bad_action_avoidance_rate"] for mode in GATE_MODES},
        "post_setup_correct_action_pred_good_rate": {mode: values[mode]["post_setup_correct_action_pred_good_rate"] for mode in GATE_MODES},
        "post_setup_correct_action_veto_rate": {mode: values[mode]["post_setup_correct_action_veto_rate"] for mode in GATE_MODES},
        "veto_false_positive_rate": {mode: values[mode]["veto_false_positive_rate"] for mode in GATE_MODES},
        "veto_true_positive_rate": {mode: values[mode]["veto_true_positive_rate"] for mode in GATE_MODES},
        "maze_post_setup_target_move_veto_rate": {mode: values[mode]["maze_post_setup_target_move_veto_rate"] for mode in GATE_MODES},
        "maze_wrong_door_pred_good_rate": {mode: values[mode]["maze_wrong_door_pred_good_rate"] for mode in GATE_MODES},
        "maze_target_color_grounding_accuracy": {mode: values[mode]["maze_target_color_grounding_accuracy"] for mode in GATE_MODES},
        "maze_hazard_clear_to_execute_success_rate": {mode: values[mode]["maze_hazard_clear_to_execute_success_rate"] for mode in GATE_MODES},
        "hybrid_gain_over_model": success["hybrid"] - success["model_only"],
        "hybrid_gain_over_prior": success["hybrid"] - success["prior_only"],
        "hybrid_veto_gain_over_model": success["hybrid_veto"] - success["model_only"],
        "hybrid_veto_gain_over_prior": success["hybrid_veto"] - success["prior_only"],
        "hybrid_veto_gain_over_hybrid": success["hybrid_veto"] - success["hybrid"],
        "prior_gain_over_model": success["prior_only"] - success["model_only"],
        "regret_reduction_vs_model": regret["model_only"] - regret["hybrid"],
        "regret_reduction_vs_prior": regret["prior_only"] - regret["hybrid"],
        "fixed_by_prior_or_hybrid": success["model_only"] < 1.0 and success["hybrid"] >= 1.0,
        "model_needed_signal": max(success["hybrid"], success["hybrid_veto"]) > success["prior_only"],
        "prior_dominated_signal": success["prior_only"] >= max(success["hybrid"], success["hybrid_veto"]),
    }


def _metrics_for_group(summary: dict[str, Any], group: str) -> dict[str, float]:
    item = summary.get("group_results", {}).get(group, {})
    keys = (
        "learned_gate_success_rate",
        "oracle_step_regret",
        "wrong_action_rate",
        "compatibility_override_rate",
        "compatibility_fix_rate",
        "compatibility_harm_rate",
        "required_action_match_rate",
        "unknown_reveal_choice_rate",
        "model_veto_activation_rate",
        "model_veto_fix_rate",
        "model_veto_harm_rate",
        "compatible_but_bad_action_avoidance_rate",
        "post_setup_correct_action_pred_good_rate",
        "post_setup_correct_action_veto_rate",
        "veto_false_positive_rate",
        "veto_true_positive_rate",
        "maze_post_setup_target_move_veto_rate",
        "maze_wrong_door_pred_good_rate",
        "maze_target_color_grounding_accuracy",
        "maze_hazard_clear_to_execute_success_rate",
    )
    return {key: float(item.get(key, 0.0)) for key in keys}


def _aggregate_comparison(by_group: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not by_group:
        return {}
    groups = list(by_group)
    mode_success = {
        mode: _mean([by_group[group]["success_rate"][mode] for group in groups])
        for mode in GATE_MODES
    }
    model_needed_groups = [group for group in groups if group.startswith("model_needed_")]
    model_needed_signal_by_group = {
        group: bool(by_group[group]["model_needed_signal"])
        for group in model_needed_groups
    }
    return {
        "macro_success_rate": mode_success,
        "hybrid_gain_over_model": mode_success["hybrid"] - mode_success["model_only"],
        "hybrid_gain_over_prior": mode_success["hybrid"] - mode_success["prior_only"],
        "hybrid_veto_gain_over_model": mode_success["hybrid_veto"] - mode_success["model_only"],
        "hybrid_veto_gain_over_prior": mode_success["hybrid_veto"] - mode_success["prior_only"],
        "hybrid_veto_gain_over_hybrid": mode_success["hybrid_veto"] - mode_success["hybrid"],
        "prior_gain_over_model": mode_success["prior_only"] - mode_success["model_only"],
        "groups_fixed_by_hybrid_vs_model": [
            group for group, item in by_group.items() if item["fixed_by_prior_or_hybrid"]
        ],
        "groups_with_model_needed_signal": [
            group for group, item in by_group.items() if item["model_needed_signal"]
        ],
        "groups_prior_dominated": [
            group for group, item in by_group.items() if item["prior_dominated_signal"]
        ],
        "model_needed_signal_by_group": model_needed_signal_by_group,
        "model_needed_signal_group_count": sum(1 for value in model_needed_signal_by_group.values() if value),
        "hybrid_veto_gain_over_prior_by_group": {
            group: by_group[group]["hybrid_veto_gain_over_prior"]
            for group in model_needed_groups
        },
        "hybrid_veto_success_rate_by_group": {
            group: by_group[group]["success_rate"]["hybrid_veto"]
            for group in model_needed_groups
        },
        "prior_only_failure_rate_by_group": {
            group: 1.0 - by_group[group]["success_rate"]["prior_only"]
            for group in model_needed_groups
        },
        "model_veto_fix_rate_by_group": {
            group: by_group[group]["model_veto_fix_rate"]["hybrid_veto"]
            for group in model_needed_groups
        },
        "model_veto_harm_rate_by_group": {
            group: by_group[group]["model_veto_harm_rate"]["hybrid_veto"]
            for group in model_needed_groups
        },
        "compatible_but_bad_action_avoidance_rate_by_group": {
            group: by_group[group]["compatible_but_bad_action_avoidance_rate"]["hybrid_veto"]
            for group in model_needed_groups
        },
        "post_setup_correct_action_pred_good_rate_by_group": {
            group: by_group[group]["post_setup_correct_action_pred_good_rate"]["hybrid_veto"]
            for group in model_needed_groups
        },
    }


def _interpretation(aggregate: dict[str, Any], mode_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    model_passed = bool(mode_summaries.get("model_only", {}).get("passed"))
    prior_passed = bool(mode_summaries.get("prior_only", {}).get("passed"))
    hybrid_passed = bool(mode_summaries.get("hybrid", {}).get("passed"))
    hybrid_veto_passed = bool(mode_summaries.get("hybrid_veto", {}).get("passed"))
    model_needed_groups = aggregate.get("groups_with_model_needed_signal", [])
    if (hybrid_passed or hybrid_veto_passed) and prior_passed and not model_needed_groups:
        verdict = "prior_dominated"
        summary = "The current benchmark is solved by compatibility prior alone; it does not prove that the world model adds decision value."
    elif (hybrid_passed or hybrid_veto_passed) and model_needed_groups:
        verdict = "hybrid_adds_value"
        summary = "Hybrid or hybrid_veto improves over prior-only on at least one group, so the model is adding decision value in this benchmark."
    elif hybrid_passed and not model_passed:
        verdict = "compatibility_fix"
        summary = "Hybrid fixes model-only failures, but the report should be checked for prior-only parity."
    else:
        verdict = "unresolved_or_regressed"
        summary = "The ablation did not produce a clean promoted result."
    return {
        "verdict": verdict,
        "summary": summary,
        "next_action": _next_action_for_verdict(verdict),
    }


def _next_action_for_verdict(verdict: str) -> str:
    if verdict == "prior_dominated":
        return "Add model-needed probes where structurally compatible actions have different outcomes, costs, or delayed consequences."
    if verdict == "hybrid_adds_value":
        return "Split model-needed probes by family, increase probe diversity, and verify the hybrid_veto gain survives before changing the model."
    if verdict == "compatibility_fix":
        return "Inspect prior-only parity and add harder probes before changing the model."
    return "Inspect failed gate summaries and failure traces before making model changes."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
