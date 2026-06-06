from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from neurokernel_seed.agents.baselines import HeuristicAgent, RandomAgent
from neurokernel_seed.agents.gated import GatedAgent
from neurokernel_seed.core.gate import ActionGate, GateMode
from neurokernel_seed.core.schema import Action, Prediction, WorldState
from neurokernel_seed.envs.base import MicroWorld
from neurokernel_seed.envs.lock_world import LockWorld
from neurokernel_seed.envs.memory_maze import MemoryMaze
from neurokernel_seed.envs.model_needed import LockTrapWorld, MazeHazardWorld, ToolPreconditionWorld
from neurokernel_seed.envs.registry import make_env
from neurokernel_seed.envs.tool_world import ToolWorld
from neurokernel_seed.eval.evaluator import Evaluator
from neurokernel_seed.predictors.learned_onnx import OnnxWorldModelPredictor
from neurokernel_seed.replay.dataset import canonical_action_key

VISIBILITY_MODES = ("visible", "partial_hidden", "hidden")


@dataclass(frozen=True)
class EnvCase:
    group: str
    env: MicroWorld


class VisibilityPredictor:
    def __init__(self, predictor, mode: str):
        if mode not in VISIBILITY_MODES:
            raise ValueError(f"unknown visibility mode: {mode}")
        self.predictor = predictor
        self.mode = mode
        self.name = f"{predictor.name}:{mode}"

    def predict(self, state: WorldState, action: Action) -> Prediction:
        return self.predictor.predict(self.visible_state(state), action)

    def visible_state(self, state: WorldState) -> WorldState:
        if self.mode == "visible":
            return state
        return WorldState(state.env_name, state.state_id, state.vector, _hide_facts(state.facts, self.mode), state.terminal)


def eval_hard_heldout(
    model_path: str | Path,
    *,
    episodes: int = 5,
    gate_mode: GateMode = "hybrid",
    out: str | Path | None = None,
) -> dict[str, Any]:
    predictor = OnnxWorldModelPredictor(model_path)
    payload = {
        "model": str(model_path),
        "episodes": episodes,
        "gate_mode": gate_mode,
        "visibility_modes": list(VISIBILITY_MODES),
        "groups": {},
    }
    groups = build_hard_eval_groups()
    for group_name, cases in groups.items():
        mode = _visibility_mode_for_group(group_name)
        payload["groups"][group_name] = _eval_group(cases, predictor, episodes, mode, gate_mode)
    if out:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        payload["out"] = str(output)
    return payload


def trace_hard_failures(
    model_path: str | Path,
    *,
    groups: list[str] | None = None,
    episodes: int = 3,
    max_failures_per_env: int = 2,
    gate_mode: GateMode = "hybrid",
    out: str | Path | None = None,
) -> dict[str, Any]:
    predictor = OnnxWorldModelPredictor(model_path)
    group_map = build_hard_eval_groups()
    requested_groups = groups or ["hard_length_heldout", "partial_hidden_smoke"]
    traces: list[dict[str, Any]] = []
    for group_name in requested_groups:
        if group_name not in group_map:
            raise ValueError(f"unknown hard eval group: {group_name}")
        visibility_mode = _visibility_mode_for_group(group_name)
        traces.extend(trace_failures_for_cases(group_map[group_name], predictor, episodes=episodes, visibility_mode=visibility_mode, max_failures_per_env=max_failures_per_env, gate_mode=gate_mode))
    summary = _trace_summary(traces)
    payload = {
        "model": str(model_path),
        "groups": requested_groups,
        "episodes_per_env": episodes,
        "max_failures_per_env": max_failures_per_env,
        "gate_mode": gate_mode,
        "summary": summary,
        "traces": traces,
    }
    if out:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        payload["out"] = str(output)
    return payload


def debug_model_needed_failures(
    model_path: str | Path,
    *,
    env_names: list[str] | None = None,
    episodes: int = 1,
    gate_mode: GateMode = "hybrid_veto",
    out: str | Path | None = None,
) -> dict[str, Any]:
    predictor = OnnxWorldModelPredictor(model_path)
    requested_envs = env_names or ["lock.probe.trap.1", "maze.probe.hazard.active.green", "maze.probe.hazard.active.red"]
    cases = {
        case.env.name: case
        for group_name, group_cases in build_hard_eval_groups().items()
        if group_name.startswith("model_needed_")
        for case in group_cases
    }
    missing = [name for name in requested_envs if name not in cases]
    if missing:
        raise ValueError(f"unknown model-needed envs: {missing}")
    traces: list[dict[str, Any]] = []
    for env_name in requested_envs:
        case = cases[env_name]
        wrapped_predictor = VisibilityPredictor(predictor, _visibility_mode_for_group(case.group))
        for seed in range(episodes):
            trace = _trace_episode(_clone_env(case.env), wrapped_predictor, case.group, "visible", seed, gate_mode)
            if not trace["success"]:
                trace["failure"] = classify_failure_trace(trace)
            trace["model_needed_diagnosis"] = _diagnose_model_needed_trace(trace)
            traces.append(trace)
    payload = {
        "debug_version": "neurokernel-model-needed-debug-v1",
        "model": str(model_path),
        "gate_mode": gate_mode,
        "envs": requested_envs,
        "episodes_per_env": episodes,
        "summary": _model_needed_debug_summary(traces),
        "traces": traces,
    }
    if out:
        output = Path(out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        payload["out"] = str(output)
    return payload


def trace_failures_for_cases(
    cases: Iterable[EnvCase],
    predictor,
    *,
    episodes: int = 3,
    visibility_mode: str = "visible",
    max_failures_per_env: int = 2,
    gate_mode: GateMode = "hybrid",
) -> list[dict[str, Any]]:
    wrapped_predictor = VisibilityPredictor(predictor, visibility_mode)
    traces: list[dict[str, Any]] = []
    for case in cases:
        failures_for_env = 0
        for seed in range(episodes):
            trace = _trace_episode(_clone_env(case.env), wrapped_predictor, case.group, visibility_mode, seed, gate_mode)
            if trace["success"]:
                continue
            trace["failure"] = classify_failure_trace(trace)
            traces.append(trace)
            failures_for_env += 1
            if failures_for_env >= max_failures_per_env:
                break
    return traces


def build_hard_eval_groups() -> dict[str, list[EnvCase]]:
    return {
        "easy_test": [
            EnvCase("easy_test", make_env("lock.test")),
            EnvCase("easy_test", make_env("maze.test")),
            EnvCase("easy_test", make_env("tool.test")),
        ],
        "hard_value_heldout": [
            EnvCase("hard_value_heldout", LockWorld("lock.heldout.value.000", "heldout", (1, 1, 1), 7)),
            EnvCase("hard_value_heldout", LockWorld("lock.heldout.value.015", "heldout", (1, 4, 4), 7)),
            EnvCase("hard_value_heldout", LockWorld("lock.heldout.value.040", "heldout", (3, 3, 1), 7)),
            EnvCase("hard_value_heldout", MemoryMaze("maze.heldout.doors.green", "heldout", "green", 9, ("green", "red", "blue"))),
            EnvCase("hard_value_heldout", MemoryMaze("maze.heldout.doors.red", "heldout", "red", 9, ("blue", "green", "red"))),
            EnvCase("hard_value_heldout", ToolWorld("tool.heldout.order.0", "heldout", 6, ("read", "search", "summarize"))),
            EnvCase("hard_value_heldout", ToolWorld("tool.heldout.order.1", "heldout", 6, ("search", "search", "summarize"))),
        ],
        "hard_length_heldout": [
            EnvCase("hard_length_heldout", LockWorld("lock.heldout.length4.000", "heldout", (1, 1, 1, 1), 8)),
            EnvCase("hard_length_heldout", LockWorld("lock.heldout.length4.005", "heldout", (1, 2, 2, 2), 8)),
            EnvCase("hard_length_heldout", LockWorld("lock.heldout.length4.250", "heldout", (4, 4, 3, 3), 8)),
            EnvCase("hard_length_heldout", ToolWorld("tool.heldout.length4.0", "heldout", 8, ("search", "read", "search", "summarize"))),
            EnvCase("hard_length_heldout", ToolWorld("tool.heldout.length4.1", "heldout", 8, ("read", "search", "read", "summarize"))),
        ],
        "partial_hidden_smoke": [
            EnvCase("partial_hidden_smoke", LockWorld("lock.test", "test", (2, 4, 1), 7, visibility_mode="partial_hidden")),
            EnvCase("partial_hidden_smoke", make_env("maze.test")),
            EnvCase("partial_hidden_smoke", ToolWorld("tool.test", "test", 6, ("search", "read", "summarize"), visibility_mode="partial_hidden")),
        ],
        "hard_length5_probe": [
            EnvCase("hard_length5_probe", LockWorld("lock.probe.length5.0", "probe", (1, 2, 3, 4, 1), 10)),
            EnvCase("hard_length5_probe", LockWorld("lock.probe.length5.1", "probe", (4, 3, 2, 1, 4), 10)),
            EnvCase("hard_length5_probe", ToolWorld("tool.probe.length5.0", "probe", 10, ("search", "read", "search", "read", "summarize"))),
            EnvCase("hard_length5_probe", ToolWorld("tool.probe.length5.1", "probe", 10, ("read", "search", "read", "search", "summarize"))),
        ],
        "partial_hidden_hard": [
            EnvCase("partial_hidden_hard", LockWorld("lock.probe.partial.length4.0", "probe", (4, 1, 3, 2), 10, visibility_mode="partial_hidden")),
            EnvCase("partial_hidden_hard", LockWorld("lock.probe.partial.length4.1", "probe", (3, 4, 2, 1), 10, visibility_mode="partial_hidden")),
            EnvCase("partial_hidden_hard", ToolWorld("tool.probe.partial.length5.0", "probe", 11, ("search", "read", "search", "read", "summarize"), visibility_mode="partial_hidden")),
            EnvCase("partial_hidden_hard", ToolWorld("tool.probe.partial.length5.1", "probe", 11, ("read", "search", "read", "search", "summarize"), visibility_mode="partial_hidden")),
            EnvCase("partial_hidden_hard", MemoryMaze("maze.probe.partial.red", "probe", "red", 10, ("green", "blue", "red"), visibility_mode="partial_hidden")),
        ],
        "full_hidden_probe": [
            EnvCase("full_hidden_probe", LockWorld("lock.probe.hidden.0", "probe", (2, 4, 1), 9, visibility_mode="hidden")),
            EnvCase("full_hidden_probe", LockWorld("lock.probe.hidden.1", "probe", (4, 2, 3, 1), 11, visibility_mode="hidden")),
            EnvCase("full_hidden_probe", ToolWorld("tool.probe.hidden.0", "probe", 8, ("search", "read", "summarize"), visibility_mode="hidden")),
            EnvCase("full_hidden_probe", ToolWorld("tool.probe.hidden.1", "probe", 11, ("read", "search", "read", "summarize"), visibility_mode="hidden")),
            EnvCase("full_hidden_probe", MemoryMaze("maze.probe.hidden.green", "probe", "green", 10, ("red", "blue", "green"), visibility_mode="hidden")),
        ],
        "model_needed_lock_trap": [
            EnvCase("model_needed_lock_trap", LockTrapWorld("lock.probe.trap.0", "probe", (2, 4, 1), 8, True)),
            EnvCase("model_needed_lock_trap", LockTrapWorld("lock.probe.trap.1", "probe", (4, 1, 3), 8, True)),
            EnvCase("model_needed_lock_trap", LockTrapWorld("lock.probe.trap.2", "probe", (1, 3, 2, 4), 10, True)),
        ],
        "model_needed_maze_hazard": [
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.active.green", "probe", "green", 7, ("red", "blue", "green"), True, True, True)),
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.active.red", "probe", "red", 7, ("green", "blue", "red"), True, True, True)),
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.active.blue", "probe", "blue", 7, ("red", "green", "blue"), True, True, True)),
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.unknown.red", "probe", "red", 8, ("green", "blue", "red"), True, True, False)),
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.cleared.green", "probe", "green", 7, ("red", "blue", "green"), False, True, True)),
            EnvCase("model_needed_maze_hazard", MazeHazardWorld("maze.probe.hazard.safe.blue", "probe", "blue", 7, ("green", "red", "blue"), False, False, True)),
        ],
        "model_needed_tool_precondition": [
            EnvCase("model_needed_tool_precondition", ToolPreconditionWorld("tool.probe.precondition.0", "probe", 6, ("read", "summarize"), False)),
            EnvCase("model_needed_tool_precondition", ToolPreconditionWorld("tool.probe.precondition.1", "probe", 7, ("read", "read", "summarize"), False)),
            EnvCase("model_needed_tool_precondition", ToolPreconditionWorld("tool.probe.precondition.2", "probe", 9, ("read", "read", "read", "summarize"), False)),
        ],
    }


def classify_failure_trace(trace: dict[str, Any]) -> dict[str, Any]:
    group = str(trace["group"])
    if "partial_hidden" in group or "hidden" in group:
        return _classify_partial_hidden_failure(trace)
    if "model_needed" in group:
        return _classify_model_needed_failure(trace)
    if "length" in group:
        return _classify_hard_length_failure(trace)
    return {"primary_bucket": "unknown_group", "secondary_buckets": [], "evidence": ["No classifier registered for this group."]}


def _visibility_mode_for_group(group_name: str) -> str:
    if "full_hidden" in group_name or group_name.endswith("_hidden_probe"):
        return "hidden"
    if "partial_hidden" in group_name:
        return "partial_hidden"
    return "visible"


def _eval_group(cases: Iterable[EnvCase], predictor, episodes: int, visibility_mode: str, gate_mode: GateMode) -> dict[str, Any]:
    env_results: dict[str, dict[str, Any]] = {}
    learned_success_values: list[float] = []
    random_success_values: list[float] = []
    heuristic_success_values: list[float] = []
    oracle_regrets: list[float] = []
    first_step_values: list[float] = []
    wrong_action_rates: list[float] = []
    early_finish_rates: list[float] = []
    info_success_rates: list[float] = []
    hidden_execution_rates: list[float] = []
    summarize_early_rates: list[float] = []
    missing_known_mask_counts: list[float] = []
    compatibility_fix_rates: list[float] = []
    compatibility_harm_rates: list[float] = []
    required_match_rates: list[float] = []
    unknown_reveal_rates: list[float] = []
    compatibility_override_rates: list[float] = []
    model_veto_activation_rates: list[float] = []
    model_veto_fix_rates: list[float] = []
    model_veto_harm_rates: list[float] = []
    compatible_bad_avoidance_rates: list[float] = []
    post_setup_correct_pred_good_rates: list[float] = []
    post_setup_correct_veto_rates: list[float] = []
    veto_false_positive_rates: list[float] = []
    veto_true_positive_rates: list[float] = []
    maze_post_setup_target_move_veto_rates: list[float] = []
    maze_wrong_door_pred_good_rates: list[float] = []
    maze_target_color_grounding_accuracies: list[float] = []
    maze_hazard_clear_to_execute_success_rates: list[float] = []
    for case in cases:
        metrics = _eval_case(case.env, predictor, episodes, visibility_mode, gate_mode)
        env_results[case.env.name] = metrics
        learned_success_values.append(float(metrics["learned_gate"]["success_rate"]))
        random_success_values.append(float(metrics["random_baseline"]["success_rate"]))
        heuristic_success_values.append(float(metrics["heuristic_baseline"]["success_rate"]))
        oracle_regrets.append(float(metrics["oracle_step_regret"]))
        first_step_values.append(float(metrics["first_step_accuracy"]))
        wrong_action_rates.append(float(metrics["wrong_action_rate"]))
        early_finish_rates.append(float(metrics["early_finish_rate"]))
        info_success_rates.append(float(metrics["info_acquisition_success_rate"]))
        hidden_execution_rates.append(float(metrics["hidden_execution_before_reveal_rate"]))
        summarize_early_rates.append(float(metrics["summarize_early_wrong_rate"]))
        missing_known_mask_counts.append(float(metrics["missing_known_mask_count"]))
        compatibility_fix_rates.append(float(metrics["compatibility_fix_rate"]))
        compatibility_harm_rates.append(float(metrics["compatibility_harm_rate"]))
        required_match_rates.append(float(metrics["required_action_match_rate"]))
        unknown_reveal_rates.append(float(metrics["unknown_reveal_choice_rate"]))
        compatibility_override_rates.append(float(metrics["compatibility_override_rate"]))
        model_veto_activation_rates.append(float(metrics["model_veto_activation_rate"]))
        model_veto_fix_rates.append(float(metrics["model_veto_fix_rate"]))
        model_veto_harm_rates.append(float(metrics["model_veto_harm_rate"]))
        compatible_bad_avoidance_rates.append(float(metrics["compatible_but_bad_action_avoidance_rate"]))
        post_setup_correct_pred_good_rates.append(float(metrics["post_setup_correct_action_pred_good_rate"]))
        post_setup_correct_veto_rates.append(float(metrics["post_setup_correct_action_veto_rate"]))
        veto_false_positive_rates.append(float(metrics["veto_false_positive_rate"]))
        veto_true_positive_rates.append(float(metrics["veto_true_positive_rate"]))
        maze_post_setup_target_move_veto_rates.append(float(metrics["maze_post_setup_target_move_veto_rate"]))
        maze_wrong_door_pred_good_rates.append(float(metrics["maze_wrong_door_pred_good_rate"]))
        maze_target_color_grounding_accuracies.append(float(metrics["maze_target_color_grounding_accuracy"]))
        maze_hazard_clear_to_execute_success_rates.append(float(metrics["maze_hazard_clear_to_execute_success_rate"]))
    aggregate = {
        "env_count": len(env_results),
        "learned_gate_success_rate": _mean(learned_success_values),
        "random_baseline_success_rate": _mean(random_success_values),
        "heuristic_baseline_success_rate": _mean(heuristic_success_values),
        "random_baseline_gap": _mean(learned_success_values) - _mean(random_success_values),
        "heuristic_baseline_gap": _mean(learned_success_values) - _mean(heuristic_success_values),
        "oracle_step_regret": _mean(oracle_regrets),
        "first_step_accuracy": _mean(first_step_values),
        "wrong_action_rate": _mean(wrong_action_rates),
        "early_finish_rate": _mean(early_finish_rates),
        "info_acquisition_success_rate": _mean(info_success_rates),
        "hidden_execution_before_reveal_rate": _mean(hidden_execution_rates),
        "summarize_early_wrong_rate": _mean(summarize_early_rates),
        "missing_known_mask_count": sum(missing_known_mask_counts),
        "compatibility_override_rate": _mean(compatibility_override_rates),
        "compatibility_fix_rate": _mean(compatibility_fix_rates),
        "compatibility_harm_rate": _mean(compatibility_harm_rates),
        "required_action_match_rate": _mean(required_match_rates),
        "unknown_reveal_choice_rate": _mean(unknown_reveal_rates),
        "model_veto_activation_rate": _mean(model_veto_activation_rates),
        "model_veto_fix_rate": _mean(model_veto_fix_rates),
        "model_veto_harm_rate": _mean(model_veto_harm_rates),
        "compatible_but_bad_action_avoidance_rate": _mean(compatible_bad_avoidance_rates),
        "post_setup_correct_action_pred_good_rate": _mean(post_setup_correct_pred_good_rates),
        "post_setup_correct_action_veto_rate": _mean(post_setup_correct_veto_rates),
        "veto_false_positive_rate": _mean(veto_false_positive_rates),
        "veto_true_positive_rate": _mean(veto_true_positive_rates),
        "maze_post_setup_target_move_veto_rate": _mean(maze_post_setup_target_move_veto_rates),
        "maze_wrong_door_pred_good_rate": _mean(maze_wrong_door_pred_good_rates),
        "maze_target_color_grounding_accuracy": _mean(maze_target_color_grounding_accuracies),
        "maze_hazard_clear_to_execute_success_rate": _mean(maze_hazard_clear_to_execute_success_rates),
    }
    return {"visibility_mode": visibility_mode, "gate_mode": gate_mode, "aggregate": aggregate, "envs": env_results}


def _eval_case(env: MicroWorld, predictor, episodes: int, visibility_mode: str, gate_mode: GateMode) -> dict[str, Any]:
    learned_predictor = VisibilityPredictor(predictor, visibility_mode)
    learned = Evaluator().run(_clone_env(env), GatedAgent(learned_predictor, gate_mode=gate_mode), episodes=episodes).as_dict()
    random = Evaluator().run(_clone_env(env), RandomAgent(0), episodes=episodes).as_dict()
    heuristic = Evaluator().run(_clone_env(env), HeuristicAgent(), episodes=episodes).as_dict()
    diagnostics = _diagnostic_rates(env, learned_predictor, episodes, visibility_mode, gate_mode)
    return {
        "learned_gate": learned,
        "random_baseline": random,
        "heuristic_baseline": heuristic,
        "random_baseline_gap": float(learned["success_rate"]) - float(random["success_rate"]),
        "heuristic_baseline_gap": float(learned["success_rate"]) - float(heuristic["success_rate"]),
        "oracle_step_regret": max(0.0, float(learned["avg_steps"]) - float(heuristic["avg_steps"])),
        "first_step_accuracy": _first_step_accuracy(_clone_env(env), learned_predictor, gate_mode),
        **diagnostics,
    }


def _diagnostic_rates(env: MicroWorld, predictor, episodes: int, visibility_mode: str, gate_mode: GateMode) -> dict[str, Any]:
    wrong_steps = 0
    total_steps = 0
    early_finish = 0
    hidden_execution_before_reveal = 0
    info_opportunities = 0
    info_successes = 0
    summarize_early_wrong = 0
    missing_known_mask_count = 0
    compatibility_overrides = 0
    compatibility_comparisons = 0
    compatibility_fixes = 0
    compatibility_harms = 0
    required_known_steps = 0
    required_matches = 0
    unknown_info_opportunities = 0
    unknown_reveal_choices = 0
    model_veto_activations = 0
    model_veto_fixes = 0
    model_veto_harms = 0
    model_veto_comparisons = 0
    compatible_bad_opportunities = 0
    compatible_bad_avoidances = 0
    post_setup_correct_opportunities = 0
    post_setup_correct_pred_good = 0
    post_setup_correct_vetoed = 0
    veto_false_positive_opportunities = 0
    veto_false_positives = 0
    veto_true_positive_opportunities = 0
    veto_true_positives = 0
    maze_post_setup_target_opportunities = 0
    maze_post_setup_target_vetoes = 0
    maze_wrong_door_candidates = 0
    maze_wrong_door_pred_good = 0
    maze_grounding_opportunities = 0
    maze_grounding_correct = 0
    maze_post_setup_episodes = 0
    maze_post_setup_successes = 0
    for seed in range(episodes):
        trace = _trace_episode(_clone_env(env), predictor, "diagnostic", visibility_mode, seed, gate_mode)
        trace_had_maze_post_setup = False
        if trace["episode_trace"] and _missing_known_mask(trace):
            missing_known_mask_count += 1
        for step in trace["episode_trace"]:
            total_steps += 1
            chosen_key = str(step["chosen_action_key"])
            expected_key = str(step["expected_action"])
            if chosen_key != step["expected_action"]:
                wrong_steps += 1
            choices = step.get("gate_mode_choices", {})
            model_choice = choices.get("model_only")
            hybrid_choice = choices.get("hybrid")
            hybrid_veto_choice = choices.get("hybrid_veto")
            if model_choice is not None and hybrid_choice is not None:
                compatibility_comparisons += 1
                if model_choice != hybrid_choice:
                    compatibility_overrides += 1
                if model_choice != expected_key and hybrid_choice == expected_key:
                    compatibility_fixes += 1
                if model_choice == expected_key and hybrid_choice != expected_key:
                    compatibility_harms += 1
            if hybrid_choice is not None and hybrid_veto_choice is not None:
                model_veto_comparisons += 1
                if hybrid_choice != hybrid_veto_choice:
                    model_veto_activations += 1
                if hybrid_choice != expected_key and hybrid_veto_choice == expected_key:
                    model_veto_fixes += 1
                if hybrid_choice == expected_key and hybrid_veto_choice != expected_key:
                    model_veto_harms += 1
            if bool(step.get("visible_state_facts", {}).get("compatible_but_bad_action", False)):
                compatible_bad_opportunities += 1
                if chosen_key == expected_key:
                    compatible_bad_avoidances += 1
            if bool(step.get("visible_state_facts", {}).get("post_setup_state", False)):
                expected_candidate = _candidate_by_key(list(step.get("candidate_actions", [])), expected_key)
                if expected_candidate is not None:
                    post_setup_correct_opportunities += 1
                    expected_diag = expected_candidate.get("candidate_diagnosis", {})
                    if not bool(expected_diag.get("predicted_bad_outcome", False)):
                        post_setup_correct_pred_good += 1
                    if _candidate_vetoed(expected_candidate):
                        post_setup_correct_vetoed += 1
            visible = step.get("visible_state_facts", {})
            if _is_maze_post_setup_step(visible):
                trace_had_maze_post_setup = True
                target_key = str(step.get("compatibility_context", {}).get("required_action_key") or expected_key)
                target_candidate = _candidate_by_key(list(step.get("candidate_actions", [])), target_key)
                if target_candidate is not None:
                    maze_post_setup_target_opportunities += 1
                    if _candidate_vetoed(target_candidate):
                        maze_post_setup_target_vetoes += 1
                move_candidates = [candidate for candidate in step.get("candidate_actions", []) if str(candidate.get("action_key", "")).startswith("move(")]
                wrong_move_candidates = [candidate for candidate in move_candidates if str(candidate.get("action_key")) != target_key]
                for candidate in wrong_move_candidates:
                    maze_wrong_door_candidates += 1
                    if not bool(candidate.get("candidate_diagnosis", {}).get("predicted_bad_outcome", False)):
                        maze_wrong_door_pred_good += 1
                if target_candidate is not None and wrong_move_candidates:
                    maze_grounding_opportunities += 1
                    target_score = _candidate_model_score(target_candidate)
                    wrong_best = max(_candidate_model_score(candidate) for candidate in wrong_move_candidates)
                    if target_score > wrong_best:
                        maze_grounding_correct += 1
            for candidate in step.get("candidate_actions", []):
                if _candidate_actual_good(candidate):
                    veto_false_positive_opportunities += 1
                    if _candidate_vetoed(candidate):
                        veto_false_positives += 1
                if _candidate_actual_bad(candidate):
                    veto_true_positive_opportunities += 1
                    if _candidate_vetoed(candidate):
                        veto_true_positives += 1
            context = step.get("compatibility_context", {})
            required_key = context.get("required_action_key")
            if required_key:
                required_known_steps += 1
                if chosen_key == required_key:
                    required_matches += 1
            elif _has_information_action(step):
                unknown_info_opportunities += 1
                if _chosen_information_action(step):
                    unknown_reveal_choices += 1
            chosen = _action_from_dict(step["chosen_action"])
            spec = _find_action_spec(env.action_specs, chosen.name)
            visible_facts = step.get("visible_state_facts", {})
            current_unknown = not bool(visible_facts.get("current_slot_known", True))
            is_last = bool(visible_facts.get("is_last_step_index", False))
            if spec and spec.terminal_only and not is_last:
                early_finish += 1
            if current_unknown:
                has_info_action = _has_information_action(step)
                if has_info_action:
                    info_opportunities += 1
                if spec and spec.requires_known_slot:
                    hidden_execution_before_reveal += 1
                actual_info = float(step["actual_result"].get("information_gain", 0.0))
                if actual_info > 0.0:
                    info_successes += 1
            if chosen.name == "summarize" and not is_last:
                summarize_early_wrong += 1
        if trace_had_maze_post_setup:
            maze_post_setup_episodes += 1
            if bool(trace.get("success", False)):
                maze_post_setup_successes += 1
    return {
        "wrong_action_rate": wrong_steps / total_steps if total_steps else 0.0,
        "early_finish_rate": early_finish / total_steps if total_steps else 0.0,
        "info_acquisition_success_rate": info_successes / info_opportunities if info_opportunities else 0.0,
        "hidden_execution_before_reveal_rate": hidden_execution_before_reveal / total_steps if total_steps else 0.0,
        "summarize_early_wrong_rate": summarize_early_wrong / total_steps if total_steps else 0.0,
        "missing_known_mask_count": missing_known_mask_count,
        "compatibility_override_rate": compatibility_overrides / compatibility_comparisons if compatibility_comparisons else 0.0,
        "compatibility_fix_rate": compatibility_fixes / compatibility_comparisons if compatibility_comparisons else 0.0,
        "compatibility_harm_rate": compatibility_harms / compatibility_comparisons if compatibility_comparisons else 0.0,
        "required_action_match_rate": required_matches / required_known_steps if required_known_steps else 0.0,
        "unknown_reveal_choice_rate": unknown_reveal_choices / unknown_info_opportunities if unknown_info_opportunities else 0.0,
        "model_veto_activation_rate": model_veto_activations / model_veto_comparisons if model_veto_comparisons else 0.0,
        "model_veto_fix_rate": model_veto_fixes / model_veto_comparisons if model_veto_comparisons else 0.0,
        "model_veto_harm_rate": model_veto_harms / model_veto_comparisons if model_veto_comparisons else 0.0,
        "compatible_but_bad_action_avoidance_rate": compatible_bad_avoidances / compatible_bad_opportunities if compatible_bad_opportunities else 0.0,
        "post_setup_correct_action_pred_good_rate": post_setup_correct_pred_good / post_setup_correct_opportunities if post_setup_correct_opportunities else 0.0,
        "post_setup_correct_action_veto_rate": post_setup_correct_vetoed / post_setup_correct_opportunities if post_setup_correct_opportunities else 0.0,
        "veto_false_positive_rate": veto_false_positives / veto_false_positive_opportunities if veto_false_positive_opportunities else 0.0,
        "veto_true_positive_rate": veto_true_positives / veto_true_positive_opportunities if veto_true_positive_opportunities else 0.0,
        "maze_post_setup_target_move_veto_rate": maze_post_setup_target_vetoes / maze_post_setup_target_opportunities if maze_post_setup_target_opportunities else 0.0,
        "maze_wrong_door_pred_good_rate": maze_wrong_door_pred_good / maze_wrong_door_candidates if maze_wrong_door_candidates else 0.0,
        "maze_target_color_grounding_accuracy": maze_grounding_correct / maze_grounding_opportunities if maze_grounding_opportunities else 0.0,
        "maze_hazard_clear_to_execute_success_rate": maze_post_setup_successes / maze_post_setup_episodes if maze_post_setup_episodes else 0.0,
    }


def _first_step_accuracy(env: MicroWorld, predictor, gate_mode: GateMode) -> float:
    state = env.reset(0)
    expected = canonical_action_key(env.expert_action(state))
    decision = ActionGate(env.action_specs, predictor, mode=gate_mode).choose(state, env.candidate_actions(state))
    return 1.0 if canonical_action_key(decision.action) == expected else 0.0


def _trace_episode(env: MicroWorld, predictor, group: str, visibility_mode: str, seed: int, gate_mode: GateMode) -> dict[str, Any]:
    state = env.reset(seed)
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    step_index = 0
    while not state.terminal and step_index < env.max_steps:
        snapshot = env.snapshot()
        step_trace = _trace_step(env, state, snapshot, predictor, visibility_mode, step_index, gate_mode)
        chosen_action = _action_from_dict(step_trace["chosen_action"])
        result = env.step(chosen_action)
        step_trace["actual_result"] = _transition_result_payload(env.name, state, result.next_state, result.reward, result.done, result.info, chosen_action)
        steps.append(step_trace)
        total_reward += float(result.reward)
        state = result.next_state
        step_index += 1
    success = env.success(state)
    return {
        "world_id": f"{env.name}.seed_{seed:04d}",
        "env_name": env.name,
        "group": group,
        "visibility_mode": visibility_mode,
        "gate_mode": gate_mode,
        "seed": seed,
        "success": success,
        "done": state.terminal,
        "steps": step_index,
        "max_steps": env.max_steps,
        "total_reward": total_reward,
        "final_state": _decode_state(state),
        "failure_reason": _episode_failure_reason(env, state, step_index),
        "episode_trace": steps,
    }


def _trace_step(env: MicroWorld, state: WorldState, snapshot: dict[str, Any], predictor, visibility_mode: str, step_index: int, gate_mode: GateMode) -> dict[str, Any]:
    candidates = env.candidate_actions(state)
    expected_action = env.expert_action(state)
    visible_facts = _hide_facts(state.facts, visibility_mode) if visibility_mode != "visible" else dict(state.facts)
    visible_state = WorldState(state.env_name, state.state_id, state.vector, visible_facts, state.terminal)
    gates = {mode: ActionGate(env.action_specs, predictor, mode=mode) for mode in ("model_only", "prior_only", "hybrid", "hybrid_veto")}
    candidate_payloads: list[dict[str, Any]] = []
    best_by_mode: dict[str, dict[str, Any]] = {}
    for action in candidates:
        prediction = predictor.predict(state, action)
        breakdowns = {mode: gate.score_breakdown(visible_state, action, prediction) for mode, gate in gates.items()}
        scores = {mode: float(breakdown["final_score"]) for mode, breakdown in breakdowns.items()}
        compatibility = gates["hybrid"].compatibility_info(visible_state, action)
        probe = _clone_env(env)
        probe.restore(snapshot)
        actual = probe.step(action)
        payload = {
            "action": _action_to_dict(action),
            "action_key": canonical_action_key(action),
            "predicted": _prediction_payload(prediction, scores[gate_mode]),
            "gate_scores": {mode: float(score) for mode, score in scores.items()},
            "score_breakdown": {mode: _jsonable(breakdown) for mode, breakdown in breakdowns.items()},
            "model_veto": float(gates["hybrid_veto"].model_veto(visible_state, action, prediction)),
            "compatibility": _compatibility_payload(compatibility),
            "counterfactual_actual": _transition_result_payload(env.name, state, actual.next_state, actual.reward, actual.done, actual.info, action),
            "candidate_diagnosis": _candidate_diagnosis(visible_state, action, prediction, actual.reward, actual.next_state, actual.done, actual.info, scores[gate_mode], breakdowns[gate_mode]),
        }
        candidate_payloads.append(payload)
        for mode, score in scores.items():
            if mode not in best_by_mode or score > float(best_by_mode[mode]["gate_scores"][mode]):
                best_by_mode[mode] = payload
    best_candidate = best_by_mode[gate_mode]
    gate_mode_choices = {mode: payload["action_key"] for mode, payload in best_by_mode.items()}
    prior_choice = best_by_mode["prior_only"]
    return {
        "step": step_index,
        "state_vector_decoded": _decode_state(state),
        "visible_task_config": _task_config(visible_facts),
        "visible_state_facts": _jsonable(visible_facts),
        "hidden_task_config_hash": _hash_json(_task_config(state.facts)),
        "expected_action": canonical_action_key(expected_action),
        "gate_mode": gate_mode,
        "gate_mode_choices": gate_mode_choices,
        "prior_only_choice_audit": prior_choice["score_breakdown"]["prior_only"].get("prior_audit", {}),
        "compatibility_context": {
            "required_action_key": gates["hybrid"].compatibility_info(visible_state, expected_action).required_action_key,
        },
        "candidate_actions": candidate_payloads,
        "chosen_action": best_candidate["action"],
        "chosen_action_key": best_candidate["action_key"],
        "chosen_gate_score": best_candidate["predicted"]["gate_score"],
    }


def _prediction_payload(prediction: Prediction, score: float) -> dict[str, Any]:
    return {
        "reward": float(prediction.reward),
        "progress_delta": prediction.progress_delta,
        "information_gain": prediction.information_gain,
        "local_success": prediction.success_probability,
        "terminal": prediction.terminal,
        "confidence": float(prediction.confidence),
        "gate_score": float(score),
        "next_state_vector": list(prediction.next_state_vector),
        "reason": prediction.reason,
    }


def _candidate_diagnosis(
    state: WorldState,
    action: Action,
    prediction: Prediction,
    actual_reward: float,
    actual_next_state: WorldState,
    actual_done: bool,
    actual_info: dict[str, Any],
    final_score: float,
    breakdown: dict[str, Any],
) -> dict[str, Any]:
    actual_progress_delta = _progress(state.env_name, actual_next_state) - _progress(state.env_name, state)
    actual_success = bool(actual_info.get("success"))
    actual_bad = _actual_bad_outcome(actual_reward, actual_progress_delta, actual_success)
    predicted_bad = _predicted_bad_outcome(prediction)
    return {
        "actual_bad_outcome": actual_bad,
        "predicted_bad_outcome": predicted_bad,
        "transition_undertrained_signal": actual_bad and not predicted_bad,
        "veto_not_triggered_signal": actual_bad and float(breakdown.get("model_veto", 0.0)) >= 0.0,
        "veto_triggered": float(breakdown.get("model_veto", 0.0)) < 0.0,
        "final_score": float(final_score),
        "actual_reward": float(actual_reward),
        "actual_progress_delta": float(actual_progress_delta),
        "actual_success": actual_success,
        "actual_done": bool(actual_done),
        "pred_reward": float(prediction.reward),
        "pred_progress_delta": None if prediction.progress_delta is None else float(prediction.progress_delta),
        "pred_local_success": None if prediction.success_probability is None else float(prediction.success_probability),
        "pred_information_gain": float(prediction.information_gain or 0.0),
    }


def _compatibility_payload(info) -> dict[str, Any]:
    return {
        "required_action_key": info.required_action_key,
        "action_key": info.action_key,
        "action_matches_required": info.action_matches_required,
        "executes_current_slot": info.executes_current_slot,
        "reveals_information": info.reveals_information,
        "requires_known_slot": info.requires_known_slot,
        "terminal_only": info.terminal_only,
        "reset_like": info.reset_like,
        "current_required_action_known": info.current_required_action_known,
        "current_slot_known": info.current_slot_known,
        "progress_fraction": info.progress_fraction,
    }


def _transition_result_payload(env_name: str, before: WorldState, after: WorldState, reward: float, done: bool, info: dict[str, Any], action: Action) -> dict[str, Any]:
    return {
        "action": canonical_action_key(action),
        "reward": float(reward),
        "done": bool(done),
        "success": bool(info.get("success")),
        "progress_delta": _progress(env_name, after) - _progress(env_name, before),
        "information_gain": float(info.get("information_gain", _visible_information_gain(before, after))),
        "next_state_vector_decoded": _decode_state(after),
        "next_state_facts": _jsonable(after.facts),
    }


def _diagnose_model_needed_trace(trace: dict[str, Any]) -> dict[str, Any]:
    buckets: list[str] = []
    evidence: list[str] = []
    per_step: list[dict[str, Any]] = []
    for step in trace.get("episode_trace", []):
        step_diag = _diagnose_model_needed_step(trace, step)
        per_step.append(step_diag)
        buckets.extend(step_diag["buckets"])
        evidence.extend(step_diag["evidence"])
    buckets = _dedupe(buckets) or ["no_model_needed_issue_detected"]
    return {
        "primary_bucket": buckets[0],
        "secondary_buckets": buckets[1:],
        "bucket_counts": _count_values(buckets),
        "evidence": _dedupe(evidence)[:20],
        "per_step": per_step,
    }


def _diagnose_model_needed_step(trace: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    buckets: list[str] = []
    evidence: list[str] = []
    facts = step.get("visible_state_facts", {})
    kind = str(facts.get("model_needed_kind", ""))
    chosen_key = str(step.get("chosen_action_key"))
    expected_key = str(step.get("expected_action"))
    candidates = list(step.get("candidate_actions", []))
    chosen = _candidate_by_key(candidates, chosen_key)
    expected = _candidate_by_key(candidates, expected_key)
    setup_candidates = [candidate for candidate in candidates if bool(candidate.get("compatibility", {}).get("reveals_information"))]
    if not kind:
        buckets.append("state_feature_missing")
        evidence.append(f"step {step['step']}: model_needed_kind missing from visible facts.")
    required_feature = {
        "lock_trap": "trap_armed",
        "maze_hazard": "hazard_active",
        "tool_precondition": "precondition_ready",
    }.get(kind)
    if required_feature and required_feature not in facts:
        buckets.append("state_feature_missing")
        evidence.append(f"step {step['step']}: {required_feature} missing from visible facts.")
    if bool(facts.get("compatible_but_bad_action", False)) and not setup_candidates:
        buckets.append("candidate_missing")
        evidence.append(f"step {step['step']}: compatible_but_bad_action is true but no reveal/setup candidate exists.")
    if expected and _candidate_actual_good(expected):
        expected_meta = expected.get("compatibility", {})
        if expected_key in {"inspect", "search"} and not bool(expected_meta.get("reveals_information")):
            buckets.append("metadata_error")
            evidence.append(f"step {step['step']}: expected setup action {expected_key} is not marked as reveals_information.")
    if chosen is not None:
        chosen_diag = chosen.get("candidate_diagnosis", {})
        if chosen_diag.get("transition_undertrained_signal"):
            buckets.append("transition_undertrained")
            evidence.append(f"step {step['step']}: model did not predict bad outcome for chosen {chosen_key}.")
        if chosen_diag.get("veto_not_triggered_signal"):
            buckets.append("veto_not_triggered")
            evidence.append(f"step {step['step']}: chosen {chosen_key} was actually bad but model_veto did not trigger.")
        if chosen_key != expected_key and _candidate_actual_bad(chosen):
            buckets.append("veto_harm")
            evidence.append(f"step {step['step']}: chose bad action {chosen_key} instead of {expected_key}.")
        if chosen_key != expected_key and _candidate_vetoed(chosen):
            buckets.append("veto_harm")
            evidence.append(f"step {step['step']}: final choice {chosen_key} had an active model veto but still won.")
    if expected is not None and chosen is not None and chosen_key != expected_key:
        expected_diag = expected.get("candidate_diagnosis", {})
        if _candidate_actual_good(expected) and bool(expected_diag.get("predicted_bad_outcome", False)):
            buckets.append("transition_undertrained")
            evidence.append(f"step {step['step']}: model predicted bad outcome for actually good expected action {expected_key}.")
        if _candidate_actual_good(expected) and _candidate_vetoed(expected):
            buckets.append("veto_harm")
            buckets.append("veto_false_positive")
            evidence.append(f"step {step['step']}: model_veto penalized actually good expected action {expected_key}.")
        chosen_score = _candidate_final_score(chosen, str(step.get("gate_mode", "hybrid_veto")))
        expected_score = _candidate_final_score(expected, str(step.get("gate_mode", "hybrid_veto")))
        if _candidate_actual_good(expected) and expected_score <= chosen_score:
            buckets.append("veto_too_weak")
            evidence.append(f"step {step['step']}: expected {expected_key} was good but scored {expected_score:.3f} <= chosen {chosen_key} {chosen_score:.3f}.")
    if not any(_candidate_actual_good(candidate) for candidate in candidates):
        buckets.append("probe_design_error")
        evidence.append(f"step {step['step']}: no candidate has a positive immediate counterfactual outcome.")
    return {
        "step": step.get("step"),
        "chosen_action": chosen_key,
        "expected_action": expected_key,
        "buckets": _dedupe(buckets),
        "evidence": _dedupe(evidence),
        "chosen_candidate": _compact_candidate_for_debug(chosen, str(step.get("gate_mode", "hybrid_veto"))) if chosen else None,
        "expected_candidate": _compact_candidate_for_debug(expected, str(step.get("gate_mode", "hybrid_veto"))) if expected else None,
    }


def _model_needed_debug_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_env: dict[str, dict[str, Any]] = {}
    buckets: list[str] = []
    for trace in traces:
        env_name = str(trace["env_name"])
        diagnosis = trace.get("model_needed_diagnosis", {})
        trace_buckets = [str(diagnosis.get("primary_bucket", "unclassified"))] + [str(item) for item in diagnosis.get("secondary_buckets", [])]
        buckets.extend(trace_buckets)
        item = by_env.setdefault(env_name, {"episodes": 0, "successes": 0, "buckets": {}})
        item["episodes"] += 1
        item["successes"] += 1 if trace.get("success") else 0
        for bucket in trace_buckets:
            item["buckets"][bucket] = item["buckets"].get(bucket, 0) + 1
    for item in by_env.values():
        item["success_rate"] = item["successes"] / item["episodes"] if item["episodes"] else 0.0
        item["buckets"] = dict(sorted(item["buckets"].items()))
    return {
        "trace_count": len(traces),
        "success_rate": sum(1 for trace in traces if trace.get("success")) / len(traces) if traces else 0.0,
        "bucket_counts": _count_values(buckets),
        "by_env": dict(sorted(by_env.items())),
    }


def _actual_bad_outcome(reward: float, progress_delta: float, success: bool) -> bool:
    return not success and (float(reward) < -0.05 or float(progress_delta) < -0.05)


def _predicted_bad_outcome(prediction: Prediction) -> bool:
    progress_delta = prediction.progress_delta
    bad_progress = progress_delta is not None and float(progress_delta) < -0.05
    return float(prediction.reward) < -0.05 or bad_progress


def _candidate_by_key(candidates: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for candidate in candidates:
        if str(candidate.get("action_key")) == key:
            return candidate
    return None


def _candidate_actual_bad(candidate: dict[str, Any]) -> bool:
    diag = candidate.get("candidate_diagnosis", {})
    if "actual_bad_outcome" in diag:
        return bool(diag.get("actual_bad_outcome", False))
    actual = candidate.get("counterfactual_actual", {})
    return _actual_bad_outcome(float(actual.get("reward", 0.0)), float(actual.get("progress_delta", 0.0)), bool(actual.get("success", False)))


def _candidate_actual_good(candidate: dict[str, Any]) -> bool:
    diag = candidate.get("candidate_diagnosis", {})
    if bool(diag.get("actual_success", False)):
        return True
    actual = candidate.get("counterfactual_actual", {})
    actual_reward = diag.get("actual_reward", actual.get("reward", 0.0))
    actual_progress = diag.get("actual_progress_delta", actual.get("progress_delta", 0.0))
    return float(actual_reward) > 0.0 or float(actual_progress) > 0.0 or bool(actual.get("success", False))


def _candidate_vetoed(candidate: dict[str, Any]) -> bool:
    return float(candidate.get("model_veto", 0.0)) < 0.0


def _candidate_model_score(candidate: dict[str, Any]) -> float:
    breakdown = candidate.get("score_breakdown", {}).get("model_only", {})
    if "model_score" in breakdown:
        return float(breakdown["model_score"])
    if "model_only" in candidate.get("gate_scores", {}):
        return float(candidate["gate_scores"]["model_only"])
    return float(candidate.get("candidate_diagnosis", {}).get("final_score", 0.0))


def _is_maze_post_setup_step(facts: dict[str, Any]) -> bool:
    return (
        str(facts.get("model_needed_kind", "")) == "maze_hazard"
        and bool(facts.get("post_setup_state", False))
        and not bool(facts.get("hazard_active", False))
    )


def _candidate_final_score(candidate: dict[str, Any], mode: str) -> float:
    breakdown = candidate.get("score_breakdown", {}).get(mode, {})
    if "final_score" in breakdown:
        return float(breakdown["final_score"])
    return float(candidate.get("gate_scores", {}).get(mode, 0.0))


def _compact_candidate_for_debug(candidate: dict[str, Any] | None, mode: str) -> dict[str, Any] | None:
    if candidate is None:
        return None
    breakdown = candidate.get("score_breakdown", {}).get(mode, {})
    diag = candidate.get("candidate_diagnosis", {})
    predicted = candidate.get("predicted", {})
    actual = candidate.get("counterfactual_actual", {})
    return {
        "action_key": candidate.get("action_key"),
        "model_score": breakdown.get("model_score"),
        "compatibility_prior": breakdown.get("compatibility_prior"),
        "visibility_guard": breakdown.get("visibility_guard"),
        "finish_guard": breakdown.get("finish_guard"),
        "reset_guard": breakdown.get("reset_guard"),
        "model_veto": breakdown.get("model_veto"),
        "hard_blocked": breakdown.get("hard_blocked"),
        "final_score": breakdown.get("final_score"),
        "pred_reward": predicted.get("reward"),
        "pred_progress_delta": predicted.get("progress_delta"),
        "pred_local_success": predicted.get("local_success"),
        "actual_reward": actual.get("reward"),
        "actual_progress_delta": actual.get("progress_delta"),
        "actual_success": actual.get("success"),
        "actual_bad_outcome": diag.get("actual_bad_outcome"),
        "transition_undertrained_signal": diag.get("transition_undertrained_signal"),
    }


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _classify_hard_length_failure(trace: dict[str, Any]) -> dict[str, Any]:
    buckets: list[str] = []
    evidence: list[str] = []
    family = str(trace["env_name"]).split(".", 1)[0]
    sequence_length = _trace_sequence_length(trace)
    wrong_steps = [step for step in trace["episode_trace"] if step["chosen_action_key"] != step["expected_action"]]
    final_wrong_steps = [step for step in wrong_steps if _is_final_progress_step(family, step)]
    repeated_steps = _repeated_action_steps(trace["episode_trace"])
    if family == "lock" and sequence_length > 4:
        buckets.append("lock_length_schema_limit")
        evidence.append(f"LockWorld sequence length {sequence_length} exceeds the current fixed-slot training/schema comfort zone.")
    if final_wrong_steps:
        buckets.append("final_step_failure")
        evidence.append(f"Wrong action on final progress step: step {final_wrong_steps[0]['step']}.")
    if repeated_steps:
        buckets.append("repeated_action")
        evidence.append(f"Repeated chosen action at steps {repeated_steps}.")
    if int(trace["steps"]) >= int(trace["max_steps"]) and not trace["success"]:
        buckets.append("max_steps_issue")
        buckets.append("length_overrun")
        evidence.append("Episode reached max_steps without success.")
    if wrong_steps:
        buckets.append("wrong_current_index")
        evidence.append(f"First wrong step chose {wrong_steps[0]['chosen_action_key']} instead of {wrong_steps[0]['expected_action']}.")
    if _schema_pressure_detected(trace):
        buckets.append("padding/schema_issue")
        evidence.append("Task length or repeated sequence stresses the current flat fixed-slot schema.")
    buckets = _dedupe(buckets) or ["unknown_length_failure"]
    return {"primary_bucket": buckets[0], "secondary_buckets": buckets[1:], "evidence": evidence}


def _classify_model_needed_failure(trace: dict[str, Any]) -> dict[str, Any]:
    buckets: list[str] = []
    evidence: list[str] = []
    wrong_steps = [step for step in trace["episode_trace"] if step["chosen_action_key"] != step["expected_action"]]
    if wrong_steps:
        first = wrong_steps[0]
        buckets.append("compatible_but_bad_action_chosen")
        evidence.append(f"Chose {first['chosen_action_key']} instead of {first['expected_action']} at step {first['step']}.")
    if int(trace["steps"]) >= int(trace["max_steps"]) and not trace["success"]:
        buckets.append("model_needed_max_steps")
        evidence.append("Episode reached max_steps without resolving the model-needed precondition.")
    buckets = _dedupe(buckets) or ["unknown_model_needed_failure"]
    return {"primary_bucket": buckets[0], "secondary_buckets": buckets[1:], "evidence": evidence}


def _classify_partial_hidden_failure(trace: dict[str, Any]) -> dict[str, Any]:
    buckets: list[str] = []
    evidence: list[str] = []
    first_step = trace["episode_trace"][0] if trace["episode_trace"] else None
    hidden_unknown = any(_has_unknown_task_values(step.get("visible_task_config", {})) for step in trace["episode_trace"])
    execution_before_reveal_steps = _execution_before_reveal_steps(trace)
    info_gain_steps = _information_gain_steps(trace)
    wrong_after_reveal_steps = _wrong_after_reveal_steps(trace)
    if execution_before_reveal_steps:
        buckets.append("acted_before_information")
        evidence.append(f"The gate executed task actions before the current hidden slot was known at steps {execution_before_reveal_steps}.")
    if first_step and _has_information_action(first_step) and not _chosen_information_action(first_step):
        buckets.append("ignored_inspect_or_search")
        evidence.append("An information-gathering action was available but not chosen.")
    if info_gain_steps and wrong_after_reveal_steps:
        buckets.append("information_collected_but_execution_failed")
        evidence.append(f"Information was collected at steps {info_gain_steps}, then wrong execution actions occurred at steps {wrong_after_reveal_steps}.")
    if _missing_known_mask(trace):
        buckets.append("missing_known_mask")
        evidence.append("Visible facts do not contain known_mask/unknown_mask fields.")
    if _hidden_info_not_revealed(trace):
        buckets.append("hidden_info_not_revealed")
        evidence.append("Unknown visible task slots remained unrevealed across the failed episode.")
    if _observation_not_stored(trace):
        buckets.append("observation_not_stored")
        evidence.append("Visible state lacks last_observation/revealed_slots/belief_state fields.")
    if _gate_underweights_information_gain(trace):
        buckets.append("gate_underweights_information_gain")
        evidence.append("A candidate with actual information gain was scored below the chosen action.")
    buckets = _dedupe(buckets) or ["unknown_partial_hidden_failure"]
    return {"primary_bucket": buckets[0], "secondary_buckets": buckets[1:], "evidence": evidence}


def _hide_facts(facts: dict[str, Any], mode: str) -> dict[str, Any]:
    hidden = dict(facts)
    family = str(facts.get("split", ""))
    hidden["visibility_mode"] = mode
    if "code" in hidden:
        code = list(hidden.get("code") or [])
        known_mask = _mask_from_facts(hidden, len(code), mode)
        hidden["code"] = [code[index] if known_mask[index] else 0 for index in range(len(code))]
        hidden["visible_code"] = tuple(hidden["code"])
        hidden["known_mask"] = tuple(1 if value else 0 for value in known_mask)
        hidden["unknown_mask"] = tuple(0 if value else 1 for value in known_mask)
        progress = int(hidden.get("progress", hidden.get("current_index", 0)))
        hidden["current_slot_known"] = progress >= len(known_mask) or bool(known_mask[progress])
    if "target_color" in hidden and mode == "hidden":
        hidden["target_color"] = ""
    if "target_color" in hidden and mode == "partial_hidden" and not bool(hidden.get("saw_hint")):
        hidden["target_color"] = ""
        hidden["visible_target_color"] = ""
        hidden["known_mask"] = (0,)
        hidden["unknown_mask"] = (1,)
        hidden["current_slot_known"] = False
    if "sequence" in hidden:
        sequence = list(hidden.get("sequence") or [])
        known_mask = _mask_from_facts(hidden, len(sequence), mode)
        hidden["sequence"] = tuple(sequence[index] if known_mask[index] else "" for index in range(len(sequence)))
        hidden["visible_sequence"] = tuple(hidden["sequence"])
        hidden["known_mask"] = tuple(1 if value else 0 for value in known_mask)
        hidden["unknown_mask"] = tuple(0 if value else 1 for value in known_mask)
        stage = int(hidden.get("stage", hidden.get("current_index", 0)))
        hidden["current_slot_known"] = stage >= len(known_mask) or bool(known_mask[stage])
        hidden["current_required_action"] = sequence[stage] if stage < len(sequence) and hidden["current_slot_known"] else "unknown"
    hidden["visibility_family_hint"] = family
    known = list(hidden.get("known_mask") or [])
    if known:
        hidden["revealed_slots_count"] = sum(1 for value in known if value)
        hidden["unknown_slots_count"] = sum(1 for value in known if not value)
    return hidden


def _hide_sequence(values: list[Any], mode: str, fill: Any) -> list[Any]:
    if mode == "hidden":
        return [fill for _ in values]
    if mode == "partial_hidden" and values:
        visible = list(values)
        midpoint = max(1, len(visible) // 2)
        for index in range(midpoint, len(visible)):
            visible[index] = fill
        return visible
    return values


def _mask_from_facts(facts: dict[str, Any], length: int, mode: str) -> list[bool]:
    existing = facts.get("known_mask")
    if existing is not None and len(existing) == length:
        return [bool(value) for value in existing]
    if mode == "visible":
        return [True] * length
    if mode == "hidden":
        return [False] * length
    midpoint = max(1, length // 2) if length else 0
    return [index < midpoint for index in range(length)]


def _task_config(facts: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key in ("code", "target_color", "doors", "sequence", "max_steps", "split", "visibility_mode", "trap_armed", "hazard_active", "precondition_ready", "model_needed_kind"):
        if key in facts:
            config[key] = _jsonable(facts[key])
    if "progress" in facts:
        config["current_index"] = int(facts["progress"])
    if "stage" in facts:
        config["current_index"] = int(facts["stage"])
    if "steps" in facts and "max_steps" in facts:
        config["remaining_steps"] = int(facts["max_steps"]) - int(facts["steps"])
    if "code" in facts:
        config["sequence_length"] = len(facts["code"])
    if "sequence" in facts:
        config["sequence_length"] = len(facts["sequence"])
    return config


def _decode_state(state: WorldState) -> dict[str, Any]:
    facts = state.facts
    decoded = {
        "env_name": state.env_name,
        "state_id": state.state_id,
        "vector": list(state.vector),
        "terminal": state.terminal,
        "facts": _jsonable(facts),
    }
    if "progress" in facts and "code" in facts:
        decoded["current_index"] = int(facts["progress"])
        decoded["sequence_length"] = len(facts["code"])
        decoded["progress_ratio"] = int(facts["progress"]) / max(1, len(facts["code"]))
    if "stage" in facts and "sequence" in facts:
        decoded["current_index"] = int(facts["stage"])
        decoded["sequence_length"] = len(facts["sequence"])
        decoded["progress_ratio"] = int(facts["stage"]) / max(1, len(facts["sequence"]))
    if "steps" in facts and "max_steps" in facts:
        decoded["remaining_steps"] = int(facts["max_steps"]) - int(facts["steps"])
    return decoded


def _progress(env_name: str, state: WorldState) -> float:
    family = env_name.split(".", 1)[0]
    if family == "lock":
        return float(state.facts["progress"]) / max(1.0, float(len(state.facts["code"])))
    if family == "maze":
        if bool(state.facts["escaped"]):
            return 1.0
        return 0.3 if bool(state.facts["saw_hint"]) else 0.0
    if family == "tool":
        return float(state.facts["stage"]) / max(1.0, float(len(state.facts["sequence"])))
    return float(state.vector[0])


def _visible_information_gain(before: WorldState, after: WorldState) -> float:
    before_visible = _task_config(_hide_facts(before.facts, "partial_hidden"))
    after_visible = _task_config(_hide_facts(after.facts, "partial_hidden"))
    before_known = _known_slot_count(before_visible)
    after_known = _known_slot_count(after_visible)
    return max(0.0, float(after_known - before_known))


def _known_slot_count(config: dict[str, Any]) -> int:
    total = 0
    for key in ("code", "sequence"):
        values = config.get(key)
        if isinstance(values, list):
            total += sum(1 for value in values if value not in (0, "", None))
    target = config.get("target_color")
    if target not in ("", None):
        total += 1
    return total


def _episode_failure_reason(env: MicroWorld, state: WorldState, steps: int) -> str | None:
    if env.success(state):
        return None
    if steps >= env.max_steps:
        return "max_steps_exhausted"
    if state.terminal:
        return "terminal_without_success"
    return "stopped_without_success"


def _is_final_progress_step(family: str, step: dict[str, Any]) -> bool:
    decoded = step["state_vector_decoded"]
    current = int(decoded.get("current_index", -1))
    length = int(decoded.get("sequence_length", 0))
    return family in {"lock", "tool"} and length > 0 and current == length - 1


def _repeated_action_steps(steps: list[dict[str, Any]]) -> list[int]:
    repeated: list[int] = []
    previous: str | None = None
    for step in steps:
        chosen = str(step["chosen_action_key"])
        if chosen == previous:
            repeated.append(int(step["step"]))
        previous = chosen
    return repeated


def _schema_pressure_detected(trace: dict[str, Any]) -> bool:
    env_name = str(trace["env_name"])
    length = _trace_sequence_length(trace)
    if env_name.startswith("lock.") and length > 4:
        return True
    if env_name.startswith("tool.") and length >= 4:
        sequence = trace.get("final_state", {}).get("facts", {}).get("sequence", [])
        return len(sequence) != len(set(sequence)) or length > 3
    return False


def _trace_sequence_length(trace: dict[str, Any]) -> int:
    final_state = trace.get("final_state", {})
    if "sequence_length" in final_state:
        return int(final_state.get("sequence_length") or 0)
    for step in trace.get("episode_trace", []):
        decoded = step.get("state_vector_decoded", {})
        if "sequence_length" in decoded:
            return int(decoded.get("sequence_length") or 0)
        config = step.get("visible_task_config", {})
        if "sequence_length" in config:
            return int(config.get("sequence_length") or 0)
    return 0


def _has_unknown_task_values(config: dict[str, Any]) -> bool:
    for key in ("code", "sequence"):
        values = config.get(key)
        if isinstance(values, list) and any(item in ("", 0, None) for item in values):
            return True
    return config.get("target_color") in ("", None)


def _current_slot_unknown(step: dict[str, Any]) -> bool:
    facts = step.get("visible_state_facts", {})
    if "current_slot_known" in facts:
        return not bool(facts["current_slot_known"])
    config = step.get("visible_task_config", {})
    index = config.get("current_index")
    if not isinstance(index, int):
        return False
    for key, unknown_markers in (("code", {0, None}), ("sequence", {"", None})):
        values = config.get(key)
        if isinstance(values, list) and 0 <= index < len(values):
            return values[index] in unknown_markers
    return False


def _has_information_action(step: dict[str, Any]) -> bool:
    keys = {candidate["action_key"] for candidate in step["candidate_actions"]}
    return bool(keys & {"inspect", "search", "read"})


def _chosen_information_action(step: dict[str, Any]) -> bool:
    return str(step["chosen_action_key"]) in {"inspect", "search", "read"}


def _execution_before_reveal_steps(trace: dict[str, Any]) -> list[int]:
    steps: list[int] = []
    for step in trace["episode_trace"]:
        if _current_slot_unknown(step) and not _chosen_information_action(step):
            steps.append(int(step["step"]))
    return steps


def _information_gain_steps(trace: dict[str, Any]) -> list[int]:
    steps: list[int] = []
    for step in trace["episode_trace"]:
        gain = float(step.get("actual_result", {}).get("information_gain", 0.0))
        if gain > 0.0:
            steps.append(int(step["step"]))
    return steps


def _wrong_after_reveal_steps(trace: dict[str, Any]) -> list[int]:
    wrong_steps: list[int] = []
    seen_information = False
    for step in trace["episode_trace"]:
        if float(step.get("actual_result", {}).get("information_gain", 0.0)) > 0.0:
            seen_information = True
            continue
        if seen_information and not _current_slot_unknown(step) and step["chosen_action_key"] != step["expected_action"]:
            wrong_steps.append(int(step["step"]))
    return wrong_steps


def _missing_known_mask(trace: dict[str, Any]) -> bool:
    for step in trace["episode_trace"]:
        facts = step.get("visible_state_facts", {})
        if "known_mask" in facts or "unknown_mask" in facts:
            return False
    return True


def _hidden_info_not_revealed(trace: dict[str, Any]) -> bool:
    if not trace["episode_trace"]:
        return False
    first = trace["episode_trace"][0].get("visible_task_config", {})
    last = trace["episode_trace"][-1].get("visible_task_config", {})
    return _has_unknown_task_values(first) and _has_unknown_task_values(last)


def _observation_not_stored(trace: dict[str, Any]) -> bool:
    for step in trace["episode_trace"]:
        facts = step.get("visible_state_facts", {})
        if any(key in facts for key in ("last_observation", "revealed_slots", "belief_state")):
            return False
    return True


def _gate_underweights_information_gain(trace: dict[str, Any]) -> bool:
    for step in trace["episode_trace"]:
        if "chosen_gate_score" not in step:
            continue
        chosen_score = float(step["chosen_gate_score"])
        for candidate in step["candidate_actions"]:
            actual_info = float(candidate.get("counterfactual_actual", {}).get("information_gain", 0.0))
            predicted_score = float(candidate.get("predicted", {}).get("gate_score", chosen_score))
            if actual_info > 0.0 and predicted_score < chosen_score and candidate["action_key"] != step["chosen_action_key"]:
                return True
    return False


def _trace_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, int] = {}
    by_env: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    for trace in traces:
        by_group[trace["group"]] = by_group.get(trace["group"], 0) + 1
        by_env[trace["env_name"]] = by_env.get(trace["env_name"], 0) + 1
        bucket = trace.get("failure", {}).get("primary_bucket", "unclassified")
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1
    return {"failure_count": len(traces), "by_group": dict(sorted(by_group.items())), "by_env": dict(sorted(by_env.items())), "by_primary_bucket": dict(sorted(by_bucket.items()))}


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _action_to_dict(action: Action) -> dict[str, Any]:
    return {"name": action.name, "params": _jsonable(action.params), "source": action.source, "confidence": action.confidence}


def _action_from_dict(payload: dict[str, Any]) -> Action:
    return Action(str(payload["name"]), dict(payload.get("params", {})), str(payload.get("source", "agent")), float(payload.get("confidence", 1.0)))


def _find_action_spec(specs, name: str):
    for spec in specs:
        if spec.name == name:
            return spec
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _hash_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clone_env(env: MicroWorld) -> MicroWorld:
    snapshot = env.snapshot()
    if snapshot.get("model_needed_kind") == "lock_trap":
        return LockTrapWorld(env.name, env.split, tuple(int(value) for value in snapshot["code"]), int(snapshot["max_steps"]), bool(snapshot["trap_armed"]))
    if snapshot.get("model_needed_kind") == "maze_hazard":
        return MazeHazardWorld(
            env.name,
            env.split,
            str(snapshot["target_color"]),
            int(snapshot["max_steps"]),
            tuple(snapshot.get("doors", ())),
            bool(snapshot["hazard_active"]),
            bool(snapshot.get("hazard_present", True)),
            bool(snapshot.get("initial_saw_hint", True)),
        )
    if snapshot.get("model_needed_kind") == "tool_precondition":
        return ToolPreconditionWorld(env.name, env.split, int(snapshot["max_steps"]), tuple(snapshot["sequence"]), bool(snapshot["precondition_ready"]))
    family = env.name.split(".", 1)[0]
    if family == "lock":
        return LockWorld(env.name, env.split, tuple(int(value) for value in snapshot["code"]), int(snapshot["max_steps"]), str(snapshot.get("visibility_mode", "visible")))
    if family == "maze":
        return MemoryMaze(env.name, env.split, str(snapshot["target_color"]), int(snapshot["max_steps"]), tuple(snapshot.get("doors", ())), str(snapshot.get("visibility_mode", "partial_hidden")))
    if family == "tool":
        return ToolWorld(env.name, env.split, int(snapshot["max_steps"]), tuple(snapshot["sequence"]), str(snapshot.get("visibility_mode", "visible")))
    raise KeyError(env.name)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
