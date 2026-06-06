from __future__ import annotations

import json
from pathlib import Path

from neurokernel_seed.cli_parser import build_parser
from neurokernel_seed.agents.baselines import HeuristicAgent, RandomAgent
from neurokernel_seed.agents.gated import GatedAgent
from neurokernel_seed.envs.registry import list_envs, make_env
from neurokernel_seed.eval.evaluator import Evaluator
from neurokernel_seed.predictors.noop import NoOpPredictor
from neurokernel_seed.predictors.symbolic import SymbolicPredictor
from neurokernel_seed.replay.counterfactual import export_candidate_counterfactuals, validate_counterfactual
from neurokernel_seed.replay.dataset import export_flat_features, inspect_replay_features, validate_flat_features
from neurokernel_seed.replay.exporter import ReplayExporter
from neurokernel_seed.replay.validator import validate_replay
from neurokernel_seed.storage.sqlite_logger import SQLiteEpisodeLogger


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "list-envs":
        for name in list_envs(args.split):
            print(name)
        return 0
    if args.cmd == "eval-env":
        result = _run_eval(args.env_name, args.agent, args.predictor, args.episodes, args.db)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "eval-suite":
        payload = _run_suite(args.split, args.agent, args.predictor, args.episodes, args.db)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "run-suite":
        payload = _run_suite(args.split, args.agent, args.predictor, args.episodes, args.db)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "collect-dataset":
        payload = _collect_dataset(args.split, args.agents, args.predictor, args.episodes, args.db, args.replay_out, args.features_out)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "collect-counterfactual-dataset":
        counterfactual_meta = export_candidate_counterfactuals(
            args.counterfactual_out,
            split=args.split,
            episodes=args.episodes,
            agents=args.agents,
            include_rollouts=not args.no_rollouts,
            include_variants=not args.no_variants,
            lock_code_lengths=args.lock_code_lengths,
            curriculum_profile=args.curriculum_profile,
            curriculum_target_groups_per_family=args.curriculum_target_groups_per_family,
        )
        counterfactual_validation = validate_counterfactual(args.counterfactual_out)
        feature_meta = export_flat_features(args.counterfactual_out, args.features_out)
        feature_validation = validate_flat_features(args.features_out)
        payload = {"counterfactual": counterfactual_meta, "counterfactual_validation": counterfactual_validation, "features": feature_meta, "feature_validation": feature_validation}
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "run-lock":
        result = _run_eval("lock.test", args.agent, args.predictor, args.episodes, args.db)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "export-replay":
        meta = ReplayExporter(args.db).export_jsonl(args.out)
        print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "validate-replay":
        result = validate_replay(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "validate-counterfactual":
        result = validate_counterfactual(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "inspect-replay":
        result = inspect_replay_features(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "export-features":
        result = export_flat_features(args.replay_path, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "validate-features":
        result = validate_flat_features(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "inspect-slot-features":
        from neurokernel_seed.replay.slot_dataset import inspect_slot_features
        result = inspect_slot_features(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "export-slot-features":
        from neurokernel_seed.replay.slot_dataset import export_slot_features
        result = export_slot_features(args.counterfactual_path, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "validate-slot-features":
        from neurokernel_seed.replay.slot_dataset import validate_slot_features
        result = validate_slot_features(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "merge-slot-features":
        from neurokernel_seed.replay.slot_dataset import merge_slot_feature_files
        result = merge_slot_feature_files(args.paths, args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "audit-slot-dataset":
        from neurokernel_seed.replay.slot_dataset import audit_slot_dataset
        result = audit_slot_dataset(args.path)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "audit-maze-grounding":
        from neurokernel_seed.replay.slot_dataset import audit_maze_grounding
        result = audit_maze_grounding(args.path, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.cmd == "check-slot-dataset-gates":
        from neurokernel_seed.replay.slot_dataset import check_slot_dataset_gates
        result = check_slot_dataset_gates(
            args.path,
            lock_max_ratio=args.lock_max_ratio,
            maze_min_ratio=args.maze_min_ratio,
            tool_min_ratio=args.tool_min_ratio,
            hiddenish_min_ratio=args.hiddenish_min_ratio,
            information_gain_group_min_ratio=args.information_gain_group_min_ratio,
            post_reveal_group_min_ratio=args.post_reveal_group_min_ratio,
            post_setup_execution_group_min_ratio=args.post_setup_execution_group_min_ratio,
            post_setup_positive_group_min_ratio=args.post_setup_positive_group_min_ratio,
            post_setup_correct_action_group_min_ratio=args.post_setup_correct_action_group_min_ratio,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.cmd == "export-balanced-slot-dataset":
        from neurokernel_seed.replay.slot_dataset import export_balanced_slot_dataset
        result = export_balanced_slot_dataset(
            args.path,
            args.out,
            target_rows=args.target_rows,
            seed=args.seed,
            include_test=not args.drop_heldout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "train-world-model":
        from neurokernel_seed.model.train import train_world_model
        result = train_world_model(args.features, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers, device=args.device, patience=args.patience)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "eval-world-model":
        from neurokernel_seed.model.evaluate import eval_world_model
        result = eval_world_model(args.checkpoint, args.features, args.split, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "eval-action-ranking":
        from neurokernel_seed.model.ranking import eval_action_ranking
        result = eval_action_ranking(args.checkpoint, args.features, args.split, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "export-world-model-onnx":
        from neurokernel_seed.model.export_onnx import export_onnx
        result = export_onnx(args.checkpoint, args.out, verify=not args.no_verify, dynamic_batch=not args.static_batch)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "eval-learned-gate":
        result = _eval_learned_gate(args.model, args.split, args.episodes, args.db, args.gate_mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "debug-lock-candidates":
        result = _debug_lock_candidates(args.env, args.model, args.gate_mode)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "freeze-baseline":
        from neurokernel_seed.baseline import freeze_baseline
        result = freeze_baseline(
            baseline_name=args.baseline_name,
            artifacts_dir=args.artifacts_dir,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            evaluation_path=args.evaluation,
            artifact_prefix=args.artifact_prefix,
            training_device=args.training_device,
            edge_runtime=args.edge_runtime,
            notes=args.notes,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "eval-hard-heldout":
        from neurokernel_seed.eval.hard_heldout import eval_hard_heldout
        result = eval_hard_heldout(args.model, episodes=args.episodes, gate_mode=args.gate_mode, out=args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "trace-hard-failures":
        from neurokernel_seed.eval.hard_heldout import trace_hard_failures
        result = trace_hard_failures(args.model, groups=args.groups, episodes=args.episodes, max_failures_per_env=args.max_failures_per_env, gate_mode=args.gate_mode, out=args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "debug-model-needed-failures":
        from neurokernel_seed.eval.hard_heldout import debug_model_needed_failures
        result = debug_model_needed_failures(args.model, env_names=args.envs, episodes=args.episodes, gate_mode=args.gate_mode, out=args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "run-benchmark":
        from neurokernel_seed.eval.benchmark import BenchmarkConfig, run_benchmark
        result = run_benchmark(
            args.model,
            out_dir=args.out_dir,
            config=BenchmarkConfig(
                episodes=args.episodes,
                trace_episodes=args.trace_episodes,
                max_failures_per_env=args.max_failures_per_env,
                strict=args.strict,
                gate_mode=args.gate_mode,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    if args.cmd == "run-gate-ablation":
        from neurokernel_seed.eval.gate_ablation import GateAblationConfig, run_gate_ablation
        result = run_gate_ablation(
            args.model,
            out_dir=args.out_dir,
            config=GateAblationConfig(
                episodes=args.episodes,
                trace_episodes=args.trace_episodes,
                max_failures_per_env=args.max_failures_per_env,
                strict=args.strict,
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["mode_passed"].get("hybrid") or result["mode_passed"].get("hybrid_veto") else 1
    if args.cmd == "benchmark-runtime":
        from neurokernel_seed.perf.runtime_benchmark import benchmark_runtime
        result = benchmark_runtime(
            args.model,
            backend=args.backend,
            iterations=args.iterations,
            samples=args.samples,
            batches=args.batches,
            out=args.out,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("available", True) else 1
    if args.cmd == "harness-actions":
        service = _make_harness_service(args.db, args.project_root)
        print(json.dumps(service.actions(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "harness-status":
        service = _make_harness_service(args.db, args.project_root)
        print(json.dumps(service.status(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "harness-create-task":
        service = _make_harness_service(args.db, args.project_root)
        result = service.create_task(_load_json_arg(args.task_json, args.task_file), source="cli")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "harness-dry-run":
        service = _make_harness_service(args.db, args.project_root)
        result = service.dry_run(args.task_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "harness-run":
        service = _make_harness_service(args.db, args.project_root)
        result = service.run(args.task_id)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {"failed"} else 1
    if args.cmd == "harness-approve":
        service = _make_harness_service(args.db, args.project_root)
        result = service.approve(args.task_id, approved_by=args.approved_by, reason=args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "harness-reject":
        service = _make_harness_service(args.db, args.project_root)
        result = service.reject(args.task_id, rejected_by=args.rejected_by, reason=args.reason)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "serve-core-api":
        from neurokernel_seed.api.server import serve
        serve(host=args.host, port=args.port, db_path=args.db, project_root=args.project_root)
        return 0
    if args.cmd == "serve-discord-bot":
        from neurokernel_seed.discord_bot.bot import config_from_env, run_discord_bot
        config = config_from_env(
            token_env=args.token_env,
            channel_id=args.channel_id,
            core_url=args.core_url,
            prefix=args.prefix,
            allow_dms=args.allow_dms,
            allowed_user_ids=tuple(args.allowed_user_id) if args.allowed_user_id is not None else None,
            reply_without_prefix=args.reply_without_prefix,
            auto_do_low_risk=args.auto_do_low_risk,
        )
        run_discord_bot(config)
        return 0
    if args.cmd == "serve-work-worker":
        from neurokernel_seed.harness.worker import serve_worker
        serve_worker(db_path=args.db, project_root=args.project_root, worker_id=args.worker_id, queues=args.queues, block_ms=args.block_ms, once=args.once)
        return 0
    if args.cmd == "language-to-core":
        from neurokernel_seed.language.codex_harness import CodexLanguageHarness, config_from_env
        config = config_from_env()
        result = CodexLanguageHarness(config).to_core(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.cmd == "language-to-human":
        from neurokernel_seed.language.codex_harness import CodexLanguageHarness, config_from_env
        config = config_from_env()
        result = CodexLanguageHarness(config).to_human(_load_json_arg(args.core_result_json, args.core_result_file))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.cmd)


def _collect_dataset(split: str, agents: list[str], predictor_name: str, episodes: int, db_path: str, replay_out: str, features_out: str) -> dict:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(db_path).exists():
        Path(db_path).unlink()
    payload: dict[str, dict] = {}
    with SQLiteEpisodeLogger(db_path) as logger:
        for agent_name in agents:
            payload[agent_name] = {}
            for env_name in list_envs(split):
                payload[agent_name][env_name] = _evaluate(make_env(env_name), agent_name, predictor_name, episodes, logger)
    replay_meta = ReplayExporter(db_path).export_jsonl(replay_out)
    replay_validation = validate_replay(replay_out)
    feature_meta = export_flat_features(replay_out, features_out)
    feature_validation = validate_flat_features(features_out)
    return {"db": db_path, "runs": payload, "replay": replay_meta, "replay_validation": replay_validation, "features": feature_meta, "feature_validation": feature_validation}


def _run_suite(split: str, agent_name: str, predictor_name: str, episodes: int, db_path: str | None) -> dict[str, dict]:
    payload: dict[str, dict] = {}
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with SQLiteEpisodeLogger(db_path) as logger:
            for env_name in list_envs(split):
                payload[env_name] = _evaluate(make_env(env_name), agent_name, predictor_name, episodes, logger)
    else:
        for env_name in list_envs(split):
            payload[env_name] = _evaluate(make_env(env_name), agent_name, predictor_name, episodes, None)
    return payload


def _run_eval(env_name: str, agent_name: str, predictor_name: str, episodes: int, db_path: str | None) -> dict:
    env = make_env(env_name)
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with SQLiteEpisodeLogger(db_path) as logger:
            return _evaluate(env, agent_name, predictor_name, episodes, logger)
    return _evaluate(env, agent_name, predictor_name, episodes, None)


def _eval_learned_gate(model_path: str, split: str, episodes: int, db_path: str | None, gate_mode: str = "hybrid") -> dict[str, dict]:
    from neurokernel_seed.predictors.learned_onnx import OnnxWorldModelPredictor
    predictor = OnnxWorldModelPredictor(model_path)
    payload: dict[str, dict] = {}
    if db_path:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with SQLiteEpisodeLogger(db_path) as logger:
            for env_name in list_envs(split):
                payload[env_name] = Evaluator(logger).run(make_env(env_name), GatedAgent(predictor, gate_mode=gate_mode), episodes=episodes).as_dict()
    else:
        for env_name in list_envs(split):
            payload[env_name] = Evaluator().run(make_env(env_name), GatedAgent(predictor, gate_mode=gate_mode), episodes=episodes).as_dict()
    return payload


def _debug_lock_candidates(env_name: str, model_path: str | None, gate_mode: str = "hybrid") -> dict:
    from neurokernel_seed.core.gate import ActionGate
    from neurokernel_seed.replay.dataset import canonical_action_key

    env = make_env(env_name)
    state = env.reset(0)
    predictor = _make_debug_predictor(model_path)
    gates = {mode: ActionGate(env.action_specs, predictor, mode=mode) for mode in ("model_only", "prior_only", "hybrid", "hybrid_veto")}
    candidates = []
    for action in env.candidate_actions(state):
        prediction = predictor.predict(state, action)
        probe = make_env(env_name)
        probe.restore(env.snapshot())
        actual = probe.step(action)
        scores = {mode: gate.score(state, action, prediction) for mode, gate in gates.items()}
        compatibility = gates["hybrid"].compatibility_info(state, action)
        candidates.append(
            {
                "action": canonical_action_key(action),
                "predicted_reward": prediction.reward,
                "predicted_progress_delta": prediction.progress_delta,
                "predicted_information_gain": prediction.information_gain,
                "predicted_success_probability": prediction.success_probability,
                "predicted_score": scores[gate_mode],
                "gate_scores": scores,
                "compatibility": {
                    "required_action_key": compatibility.required_action_key,
                    "action_matches_required": compatibility.action_matches_required,
                    "executes_current_slot": compatibility.executes_current_slot,
                    "reveals_information": compatibility.reveals_information,
                    "requires_known_slot": compatibility.requires_known_slot,
                    "terminal_only": compatibility.terminal_only,
                    "reset_like": compatibility.reset_like,
                },
                "actual_reward": actual.reward,
                "actual_success": bool(actual.info.get("success")),
                "actual_next_facts": actual.next_state.facts,
            }
        )
    candidates.sort(key=lambda item: item["gate_scores"][gate_mode], reverse=True)
    return {
        "env_name": env_name,
        "predictor": predictor.name,
        "gate_mode": gate_mode,
        "state_facts": state.facts,
        "best_action": candidates[0]["action"] if candidates else None,
        "candidates": candidates,
    }


def _make_debug_predictor(model_path: str | None):
    if model_path:
        from neurokernel_seed.predictors.learned_onnx import OnnxWorldModelPredictor

        return OnnxWorldModelPredictor(model_path)
    return SymbolicPredictor()


def _evaluate(env, agent_name: str, predictor_name: str, episodes: int, logger: SQLiteEpisodeLogger | None) -> dict:
    predictor = _make_predictor(predictor_name)
    agent = _make_agent(agent_name, predictor)
    return Evaluator(logger).run(env, agent, episodes=episodes).as_dict()


def _make_predictor(name: str):
    if name == "noop":
        return NoOpPredictor()
    if name == "symbolic":
        return SymbolicPredictor()
    raise ValueError(name)


def _make_agent(name: str, predictor):
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    if name == "gated":
        return GatedAgent(predictor)
    raise ValueError(name)


def _make_harness_service(db_path: str, project_root: str):
    from neurokernel_seed.harness.service import HarnessService

    return HarnessService(db_path=db_path, project_root=project_root)


def _load_json_arg(task_json: str | None, task_file: str | None) -> dict:
    if bool(task_json) == bool(task_file):
        raise ValueError("provide exactly one of --task-json or --task-file")
    if task_file:
        return json.loads(Path(task_file).read_text(encoding="utf-8-sig"))
    return json.loads(task_json or "{}")


if __name__ == "__main__":
    raise SystemExit(main())
