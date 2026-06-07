import argparse
import json
from pathlib import Path

from neurokernel_seed import nk_cli
from neurokernel_seed.nk_console import menu as nk_menu


def _write_ablation(run_dir: Path, *, hybrid_veto: float, prior: float, signal_count: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_ablation_result.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "macro_success_rate": {
                        "model_only": 0.5,
                        "prior_only": prior,
                        "hybrid": max(prior, hybrid_veto - 0.1),
                        "hybrid_veto": hybrid_veto,
                    },
                    "model_needed_signal_group_count": signal_count,
                    "hybrid_veto_gain_over_prior": hybrid_veto - prior,
                    "hybrid_gain_over_prior": 0.0,
                    "groups_with_model_needed_signal": ["model_needed_probe"] if signal_count else [],
                },
                "interpretation": {"verdict": "hybrid_adds_value"},
            }
        ),
        encoding="utf-8",
    )


def test_nk_top_ranks_training_runs_by_benchmark_quality(tmp_path):
    _write_ablation(tmp_path / "mid", hybrid_veto=0.9, prior=0.7, signal_count=1)
    _write_ablation(tmp_path / "best", hybrid_veto=1.0, prior=0.6, signal_count=3)
    _write_ablation(tmp_path / "weak", hybrid_veto=0.8, prior=0.8, signal_count=0)

    result = nk_cli._run_top_models(argparse.Namespace(run_dir=str(tmp_path), limit=2))

    assert [item["run_name"] for item in result["top"]] == ["best", "mid"]
    assert result["top"][0]["hybrid_veto_gain_over_prior"] == 0.4


def test_nk_help_exposes_menu_commands():
    parser = nk_cli._build_parser()
    help_text = parser.format_help()
    assert "runtime-seed" in help_text
    assert "runtime-auto" in help_text
    assert "runtime-cycle" in help_text
    assert "deploy-use" in help_text
    assert "top" in help_text


def test_nk_dashboard_loops_until_exit(monkeypatch, capsys):
    calls = []

    def fake_run_action(args):
        calls.append(args.action)
        if args.action == "current":
            return {"current": {"release_name": "current_a"}}
        if args.action == "top":
            return {"top": []}
        if args.action == "compare":
            return {"status": "ok", "current_name": "current_a", "best": None, "needs_deploy": False}
        raise AssertionError(args.action)

    answers = iter(["compare", "0"])
    monkeypatch.setattr(nk_cli, "_run_action", fake_run_action)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(nk_cli.sys.stdin, "isatty", lambda: False)

    result = nk_cli._run_dashboard(argparse.Namespace(action=None, run_dir=None, limit=5))

    output = capsys.readouterr().out
    assert result == 0
    assert calls.count("top") >= 1
    assert "NK 학습 콘솔" in output
    assert "비교" in output
    assert "종료" in output


def test_nk_menu_shows_primary_actions(monkeypatch):
    monkeypatch.delenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", raising=False)
    monkeypatch.delenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", raising=False)
    monkeypatch.delenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", raising=False)
    base = argparse.Namespace()

    assert [(item.key, item.action) for item in nk_menu.DASHBOARD_COMMANDS] == [
        ("1", "runtime-seed"),
        ("2", "runtime-auto"),
        ("3", "runtime-cycle"),
        ("4", "compare"),
        ("5", "deploy-best"),
        ("0", "exit"),
    ]
    assert nk_menu.dashboard_action("시드") == "runtime-seed"
    assert nk_menu.dashboard_action("학습") == "runtime-auto"
    assert nk_menu.dashboard_action("배포학습") == "runtime-cycle"
    assert nk_menu.dashboard_action("월드배포") == "deploy-best"
    assert nk_menu.dashboard_action("종료") == "exit"
    seed_args = nk_menu.build_menu_args(base, "runtime-seed")
    auto_args = nk_menu.build_menu_args(base, "runtime-auto")
    data_args = nk_menu.build_menu_args(base, "runtime-data")
    feature_args = nk_menu.build_menu_args(base, "runtime-features")
    train_args = nk_menu.build_menu_args(base, "runtime-train")
    top_args = nk_menu.build_menu_args(base, "top")
    cycle_args = nk_menu.build_menu_args(base, "runtime-cycle")

    assert seed_args.cycles == 8
    assert seed_args.include_failures is True
    assert seed_args.export_dataset is True
    assert seed_args.min_actions == 4
    assert auto_args.limit is None
    assert data_args.limit is None
    assert data_args.out == "data/model_ready/runtime_replay.jsonl"
    assert feature_args.replay == "data/model_ready/runtime_replay.jsonl"
    assert feature_args.out == "data/model_ready/runtime_features.jsonl"
    assert train_args.features == "data/model_ready/runtime_features.jsonl"
    assert train_args.out == "artifacts/runtime_action_model.pt"
    assert top_args.limit == 5
    assert cycle_args.deploy_runtime is True


def test_nk_parser_accepts_korean_runtime_shortcuts():
    parser = nk_cli._build_parser()

    seed = parser.parse_args(["시드", "--no-export"])
    cycle = parser.parse_args(["배포학습", "--no-deploy"])
    deploy = parser.parse_args(["런타임배포", "--model", "artifacts/runtime_action_model.pt"])

    assert seed.action == "시드"
    assert seed.export_dataset is False
    assert cycle.action == "배포학습"
    assert cycle.deploy_runtime is False
    assert deploy.action == "런타임배포"
    assert deploy.model == "artifacts/runtime_action_model.pt"


def test_nk_runtime_seed_local_creates_real_execution_rows(tmp_path):
    db = tmp_path / "harness.db"

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-seed",
            source="local",
            db=str(db),
            remote_db="data/harness.db",
            cache_db=None,
            remote_host=None,
            remote_project=None,
            ssh_connect_timeout=10,
            project_root=".",
            profile="readonly-basic",
            target="local",
            cycles=1,
            include_failures=True,
            export_dataset=False,
            replay_out=str(tmp_path / "runtime_replay.jsonl"),
            features_out=str(tmp_path / "runtime_features.jsonl"),
            test_ratio=0.2,
            min_rows=10,
            min_actions=4,
        )
    )

    seed = result["seed"]
    assert result["status"] == "completed"
    assert seed["tasks_created"] == 14
    assert seed["success_rows"] >= 10
    assert seed["failure_rows"] >= 4
    assert seed["action_counts"]["list_artifacts"] == 1
    assert seed["action_counts"]["tail_logs"] == 3
    assert db.exists()
    assert result["artifacts"] == {}


def test_nk_runtime_seed_local_exports_dataset_after_seed(tmp_path, monkeypatch):
    calls = []
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"

    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_seed_local_action",
        lambda args: {"tasks_created": 14, "success_rows": 10, "failure_rows": 4, "action_counts": {"tail_logs": 3}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_data_action",
        lambda args: calls.append(("data", args.source, args.out, args.min_rows))
        or {"ready_for_runtime_training": True, "validation": {"rows": 14, "success_rows": 10, "failure_rows": 4}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_features_action",
        lambda args: calls.append(("features", args.replay, args.out, args.min_actions))
        or {"ready_for_runtime_model_training": True, "validation": {"rows": 28, "input_dim": 32}},
    )

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-seed",
            source="local",
            db=str(tmp_path / "harness.db"),
            remote_db="data/harness.db",
            cache_db=None,
            remote_host=None,
            remote_project=None,
            ssh_connect_timeout=10,
            project_root=".",
            profile="readonly-basic",
            target="local",
            cycles=1,
            include_failures=True,
            export_dataset=True,
            replay_out=str(replay),
            features_out=str(features),
            test_ratio=0.2,
            min_rows=10,
            min_actions=4,
        )
    )

    assert result["ready_for_runtime_model_training"] is True
    assert result["artifacts"] == {"replay": str(replay), "features": str(features)}
    assert calls == [
        ("data", "local", str(replay), 10),
        ("features", str(replay), str(features), 4),
    ]


def test_nk_runtime_seed_edge_runs_remote_seed_then_exports(tmp_path, monkeypatch):
    calls = []
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"

    monkeypatch.setattr(
        nk_cli,
        "_run_remote_runtime_seed_command",
        lambda args: calls.append(("remote-seed", args.remote_host, args.cycles))
        or {"status": "completed", "seed": {"tasks_created": 28, "success_rows": 20, "failure_rows": 8}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_data_action",
        lambda args: calls.append(("data", args.source, args.remote_project, args.out))
        or {"ready_for_runtime_training": True, "validation": {"rows": 28}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_features_action",
        lambda args: calls.append(("features", args.replay, args.out, args.min_actions))
        or {"ready_for_runtime_model_training": True, "validation": {"rows": 56}},
    )

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-seed",
            source="edge",
            db="data/harness.db",
            remote_db="data/harness.db",
            cache_db=None,
            remote_host="orangepi5",
            remote_project="/remote",
            ssh_connect_timeout=10,
            project_root=".",
            profile="readonly-basic",
            target="orangepi5",
            cycles=2,
            include_failures=True,
            export_dataset=True,
            replay_out=str(replay),
            features_out=str(features),
            test_ratio=0.2,
            min_rows=10,
            min_actions=4,
        )
    )

    assert result["source"] == "edge"
    assert result["ready_for_runtime_model_training"] is True
    assert calls == [
        ("remote-seed", "orangepi5", 2),
        ("data", "edge", "/remote", str(replay)),
        ("features", str(replay), str(features), 4),
    ]


def test_nk_runtime_seed_remote_command_prefers_venv_python(monkeypatch):
    captured = {}

    class Result:
        returncode = 0
        stdout = json.dumps({"status": "completed", "seed": {"tasks_created": 14}})
        stderr = ""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return Result()

    monkeypatch.setattr(nk_cli.subprocess, "run", fake_run)

    result = nk_cli._run_remote_runtime_seed_command(
        argparse.Namespace(
            remote_host="orangepi5",
            remote_project="/remote",
            remote_db="data/harness.db",
            remote_python=None,
            ssh_connect_timeout=10,
            profile="readonly-basic",
            cycles=2,
            include_failures=True,
        )
    )

    remote_command = captured["cmd"][-1]
    assert result["seed"]["tasks_created"] == 14
    assert "test -x venv/bin/python" in remote_command
    assert "PYTHONPATH=src" in remote_command
    assert "runtime-seed --source local" in remote_command
    assert captured["kwargs"]["encoding"] == "utf-8"


def test_nk_compare_reports_current_best_and_deploy_need(tmp_path, monkeypatch):
    _write_ablation(tmp_path / "best", hybrid_veto=1.0, prior=0.6, signal_count=3)
    monkeypatch.setattr(
        nk_cli,
        "current_model_release",
        lambda config: {"current": {"release_name": "old_model"}, "remote_host": "orangepi5", "remote_project": "/remote"},
    )

    result = nk_cli._run_action(argparse.Namespace(action="compare", run_dir=str(tmp_path), limit=5, remote_host=None, remote_project=None, ssh_connect_timeout=10))

    assert result["current_name"] == "old_model"
    assert result["best_name"] == "best"
    assert result["needs_deploy"] is True


def test_nk_current_reports_world_and_runtime_model_slots(monkeypatch):
    monkeypatch.setattr(
        nk_cli,
        "current_model_release",
        lambda config: {
            "status": "ok",
            "remote_host": "orangepi5",
            "remote_project": "/remote",
            "current": {
                "release_name": "world_a",
                "primary_model": "/remote/artifacts/current_world_model.onnx",
                "remote_release_dir": "/remote/artifacts/model_releases/world_a",
            },
        },
    )
    monkeypatch.setattr(
        nk_cli,
        "_runtime_model_slot",
        lambda config: {
            "slot": "runtime_action",
            "status": "active",
            "model": "/remote/artifacts/current_runtime_action_model.pt",
            "data_origin": "runtime_experience_log",
        },
    )

    result = nk_cli._run_action(
        argparse.Namespace(
            action="current",
            remote_host="orangepi5",
            remote_project="/remote",
            ssh_connect_timeout=10,
        )
    )

    assert result["model_slots"]["world"]["slot"] == "world"
    assert result["model_slots"]["world"]["release_name"] == "world_a"
    assert result["model_slots"]["runtime"]["slot"] == "runtime_action"
    assert result["model_slots"]["runtime"]["status"] == "active"
    assert result["model_slots"]["runtime"]["data_origin"] == "runtime_experience_log"


def test_nk_deploy_best_skips_when_current_matches_best(tmp_path, monkeypatch):
    _write_ablation(tmp_path / "best", hybrid_veto=1.0, prior=0.6, signal_count=3)
    monkeypatch.setattr(
        nk_cli,
        "current_model_release",
        lambda config: {"current": {"release_name": "best"}, "remote_host": "orangepi5", "remote_project": "/remote"},
    )

    result = nk_cli._run_action(argparse.Namespace(action="deploy-best", run_dir=str(tmp_path), limit=5, remote_host=None, remote_project=None, ssh_connect_timeout=10))

    assert result["status"] == "already_current"


def test_nk_deploy_best_deploys_when_best_differs(tmp_path, monkeypatch):
    _write_ablation(tmp_path / "best", hybrid_veto=1.0, prior=0.6, signal_count=3)
    captured = {}
    monkeypatch.setattr(
        nk_cli,
        "current_model_release",
        lambda config: {"current": {"release_name": "old_model"}, "remote_host": "orangepi5", "remote_project": "/remote"},
    )

    def fake_deploy(source_dir, **kwargs):
        captured["source_dir"] = source_dir
        captured["kwargs"] = kwargs
        return {"remote_release_dir": "/remote/best"}

    monkeypatch.setattr(nk_cli, "deploy_training_run", fake_deploy)

    result = nk_cli._run_action(argparse.Namespace(action="deploy-best", run_dir=str(tmp_path), limit=5, remote_host="orangepi5", remote_project="/remote", ssh_connect_timeout=10))

    assert result["status"] == "deployed_and_activated"
    assert result["run_name"] == "best"
    assert captured["source_dir"] == tmp_path / "best"
    assert captured["kwargs"]["activate"] is True


def test_nk_runtime_auto_runs_data_features_and_training(tmp_path, monkeypatch):
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    model = tmp_path / "runtime_action_model.pt"
    report_dir = tmp_path / "reports"
    calls = []

    def fake_runtime_data(args):
        calls.append(("data", args.out))
        return {
            "ready_for_runtime_training": True,
            "source": "edge",
            "validation": {"rows": 12},
            "gates": {"gates": {}},
        }

    def fake_runtime_features(args):
        calls.append(("features", args.replay, args.out))
        return {
            "ready_for_runtime_model_training": True,
            "validation": {"input_dim": 24},
            "gates": {"gates": {}},
        }

    def fake_runtime_train(args):
        calls.append(("train", args.features, args.out))
        return {
            "checkpoint": str(model),
            "metrics": str(model.with_suffix(".metrics.json")),
            "test": {"success_accuracy": 0.8, "reward_mae": 0.2},
        }

    monkeypatch.setenv("NEUROKERNEL_RUNTIME_PIPELINE_DIR", str(report_dir))
    monkeypatch.setattr(nk_cli, "_run_runtime_data_action", fake_runtime_data)
    monkeypatch.setattr(nk_cli, "_run_runtime_features_action", fake_runtime_features)
    monkeypatch.setattr(nk_cli, "_run_runtime_training_action", fake_runtime_train)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-auto",
            source="edge",
            db="data/harness.db",
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
            remote_db="data/harness.db",
            cache_db=None,
            replay_out=str(replay),
            features_out=str(features),
            model_out=str(model),
            test_ratio=0.2,
            min_rows=10,
            min_actions=1,
            epochs=1,
            batch_size=8,
            device="cpu",
            patience=3,
        )
    )

    assert result["status"] == "completed"
    assert calls == [
        ("data", str(replay)),
        ("features", str(replay), str(features)),
        ("train", str(features), str(model)),
    ]
    assert Path(result["report"]).exists()


def test_nk_runtime_cycle_trains_deploys_and_rechecks_current(tmp_path, monkeypatch):
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    model = tmp_path / "runtime_action_model.pt"
    report_dir = tmp_path / "reports"
    calls = []

    monkeypatch.setenv("NEUROKERNEL_RUNTIME_PIPELINE_DIR", str(report_dir))
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_data_action",
        lambda args: calls.append(("data", args.out)) or {"ready_for_runtime_training": True, "validation": {"rows": 12}, "gates": {"gates": {}}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_features_action",
        lambda args: calls.append(("features", args.replay, args.out))
        or {"ready_for_runtime_model_training": True, "validation": {"input_dim": 24}, "gates": {"gates": {}}},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_runtime_training_action",
        lambda args: calls.append(("train", args.features, args.out))
        or {"checkpoint": str(model), "manifest": str(model.with_suffix(".manifest.json")), "metrics": str(model.with_suffix(".metrics.json"))},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_deploy_runtime_action",
        lambda args: calls.append(("deploy", args.model))
        or {"status": "deployed_and_activated", "remote_model": "/remote/artifacts/current_runtime_action_model.pt"},
    )
    monkeypatch.setattr(
        nk_cli,
        "_run_current",
        lambda args: calls.append(("current", args.remote_host))
        or {"model_slots": {"runtime": {"status": "active", "model": "/remote/artifacts/current_runtime_action_model.pt"}}},
    )

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-cycle",
            source="edge",
            db="data/harness.db",
            remote_host="orangepi5",
            remote_project="/remote",
            remote_db="data/harness.db",
            cache_db=None,
            replay_out=str(replay),
            features_out=str(features),
            model_out=str(model),
            test_ratio=0.2,
            min_rows=10,
            min_actions=1,
            epochs=1,
            batch_size=8,
            device="cpu",
            patience=3,
            deploy_runtime=True,
            ssh_connect_timeout=10,
        )
    )

    assert result["pipeline"] == "runtime_action_cycle_v1"
    assert result["steps"]["runtime_deploy"]["status"] == "deployed_and_activated"
    assert result["steps"]["current_after"]["model_slots"]["runtime"]["status"] == "active"
    assert calls == [
        ("data", str(replay)),
        ("features", str(replay), str(features)),
        ("train", str(features), str(model)),
        ("deploy", str(model)),
        ("current", "orangepi5"),
    ]


def test_nk_train_runs_local_pipeline_without_deploy(tmp_path, monkeypatch):
    captured = {}

    def fake_pipeline(config):
        captured["config"] = config
        return {
            "status": "completed",
            "run_name": "local_run",
            "run_dir": str(tmp_path / "runs" / "local_run"),
            "artifacts": {"gate_ablation": str(tmp_path / "runs" / "local_run" / "gate_ablation" / "gate_ablation_v4_4.json")},
        }

    monkeypatch.setattr(nk_cli, "run_training_pipeline", fake_pipeline)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="train",
            features=str(tmp_path / "features.jsonl"),
            out_dir=str(tmp_path / "runs"),
            run_name="local_run",
            epochs=1,
            batch_size=1024,
            device="cuda",
            patience=10,
            skip_gate_ablation=False,
            overwrite_local_run=False,
        )
    )

    assert result["status"] == "completed"
    assert captured["config"].run_name == "local_run"
    assert captured["config"].run_gate_ablation is True


def test_nk_data_checks_training_feature_contract(tmp_path, monkeypatch):
    features = tmp_path / "features.jsonl"
    features.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_validate(path):
        captured["validate"] = Path(path)
        return {"rows": 10, "schema_version": "neurokernel-slot-transition-v2", "input_dim": 41, "target_dim": 9}

    def fake_gates(path):
        captured["gates"] = Path(path)
        return {"passed": True, "audit_shortage_report": []}

    monkeypatch.setattr(nk_cli, "validate_slot_features", fake_validate)
    monkeypatch.setattr(nk_cli, "check_slot_dataset_gates", fake_gates)

    result = nk_cli._run_action(argparse.Namespace(action="data", features=str(features)))

    assert result["ready_for_training"] is True
    assert captured == {"validate": features, "gates": features}


def test_nk_runtime_features_exports_and_checks_contract(tmp_path, monkeypatch):
    replay = tmp_path / "runtime_replay.jsonl"
    features = tmp_path / "runtime_features.jsonl"
    captured = {}

    def fake_export(source, out, *, test_ratio):
        captured["export"] = (Path(source), Path(out), test_ratio)
        return {"rows": 3, "schema_version": "neurokernel-runtime-action-feature-v1", "input_dim": 10, "target_dim": 4}

    def fake_validate(path):
        captured["validate"] = Path(path)
        return {"accepted": True, "rows": 3, "schema_version": "neurokernel-runtime-action-feature-v1", "input_dim": 10, "target_dim": 4}

    def fake_gates(path, *, min_rows, min_actions):
        captured["gates"] = (Path(path), min_rows, min_actions)
        return {"passed": True, "ready_for_runtime_model_training": True, "warnings": []}

    monkeypatch.setattr(nk_cli, "export_runtime_features", fake_export)
    monkeypatch.setattr(nk_cli, "validate_runtime_features", fake_validate)
    monkeypatch.setattr(nk_cli, "check_runtime_feature_gates", fake_gates)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-features",
            replay=str(replay),
            out=str(features),
            test_ratio=0.1,
            min_rows=2,
            min_actions=1,
        )
    )

    assert result["ready_for_runtime_model_training"] is True
    assert captured["export"] == (replay, features, 0.1)
    assert captured["validate"] == features
    assert captured["gates"] == (features, 2, 1)


def test_nk_runtime_data_pulls_edge_db_to_cache(tmp_path, monkeypatch):
    cache_db = tmp_path / "cache" / "harness.db"
    replay = tmp_path / "runtime_replay.jsonl"
    captured = {}

    def fake_copy(remote_host, remote_path, local_path):
        captured["copy"] = (remote_host, remote_path, Path(local_path))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_text("db", encoding="utf-8")

    def fake_etl(config):
        captured["etl"] = config
        return {
            "status": "passed",
            "out": str(config.out_path),
            "validation": {"rows": 1, "success_rows": 1, "failure_rows": 0, "schema_version": "neurokernel-runtime-action-v1"},
            "gates": {"passed": True, "gates": {}},
            "ready_for_runtime_training": True,
        }

    monkeypatch.setattr(nk_cli, "_copy_remote_file", fake_copy)
    monkeypatch.setattr(nk_cli, "run_runtime_replay_etl", fake_etl)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-data",
            source="edge",
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
            remote_db="data/harness.db",
            cache_db=str(cache_db),
            db="data/harness.db",
            out=str(replay),
            limit=None,
            min_rows=1,
            allow_no_execution=False,
        )
    )

    assert result["source"] == "edge"
    assert captured["copy"] == ("orangepi5", "/home/ubuntu/projects/neurokernel-agi-seed/data/harness.db", cache_db)
    assert captured["etl"].db_path == cache_db


def test_nk_runtime_data_reports_missing_local_db(tmp_path):
    missing = tmp_path / "missing.db"

    try:
        nk_cli._run_action(
            argparse.Namespace(
                action="runtime-data",
                source="local",
                db=str(missing),
                out=str(tmp_path / "runtime_replay.jsonl"),
                limit=None,
                min_rows=1,
                allow_no_execution=False,
            )
        )
    except nk_cli.ModelReleaseError as exc:
        assert "사용데이터 입력 실패" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing local db should fail explicitly")


def test_nk_runtime_train_uses_runtime_action_model(tmp_path, monkeypatch):
    features = tmp_path / "runtime_features.jsonl"
    out = tmp_path / "runtime_action_model.pt"
    captured = {}

    def fake_train(feature_path, out_path, **kwargs):
        captured["feature_path"] = Path(feature_path)
        captured["out_path"] = Path(out_path)
        captured["kwargs"] = kwargs
        return {
            "checkpoint": str(out_path),
            "metrics": str(Path(out_path).with_suffix(".metrics.json")),
            "device": "cpu",
            "best_epoch": 2,
            "train": {"success_accuracy": 1.0, "reward_mae": 0.1},
            "test": {"success_accuracy": 0.9, "reward_mae": 0.2},
        }

    monkeypatch.setattr(nk_cli, "train_runtime_action_model", fake_train)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="runtime-train",
            features=str(features),
            out=str(out),
            epochs=5,
            batch_size=8,
            lr=0.001,
            weight_decay=0.0001,
            hidden_dim=16,
            hidden_layers=1,
            device="cpu",
            patience=3,
            min_rows=4,
            min_actions=2,
        )
    )

    assert result["status"] == "completed"
    assert captured["feature_path"] == features
    assert captured["out_path"] == out
    assert captured["kwargs"]["min_actions"] == 2
    assert result["test"]["success_accuracy"] == 0.9


def test_nk_deploy_runtime_activates_runtime_slot(tmp_path, monkeypatch):
    model = tmp_path / "runtime_action_model.pt"
    manifest = tmp_path / "runtime_action_model.manifest.json"
    model.write_text("model", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "neurokernel-runtime-action-model-v1",
                "model_slot": "runtime_action_model",
                "data_origin": "runtime_experience_log",
                "activation_status": "candidate",
            }
        ),
        encoding="utf-8",
    )
    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Result()

    monkeypatch.setattr(nk_cli.subprocess, "run", fake_run)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="deploy-runtime",
            model=str(model),
            remote_host="orangepi5",
            remote_project="/remote",
            ssh_connect_timeout=10,
        )
    )

    assert result["status"] == "deployed_and_activated"
    assert result["remote_model"] == "/remote/artifacts/current_runtime_action_model.pt"
    assert result["remote_manifest"] == "/remote/artifacts/current_runtime_action_model.manifest.json"
    assert result["model_slots"]["runtime"]["status"] == "active"
    assert any(call[0] == "scp" and str(model) in call for call in calls)
    assert any(call[0] == "scp" and str(manifest) in call for call in calls)
    assert any(call[0] == "ssh" and "current_runtime_action_model.pt" in call[-1] for call in calls)


def test_nk_deploy_uses_existing_training_run(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "candidate"
    run_dir.mkdir(parents=True)
    captured = {}

    def fake_deploy(source_dir, **kwargs):
        captured["source_dir"] = source_dir
        captured["kwargs"] = kwargs
        return {"remote_release_dir": "/remote/candidate"}

    monkeypatch.setattr(nk_cli, "deploy_training_run", fake_deploy)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="deploy-use",
            run_name="candidate",
            run_dir=str(tmp_path / "runs"),
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
            ssh_connect_timeout=10,
        )
    )

    assert result["status"] == "deployed_and_activated"
    assert captured["source_dir"] == run_dir
    assert captured["kwargs"]["activate"] is True


def test_nk_bench_current_saves_result_as_comparable_training_run(tmp_path, monkeypatch):
    copied: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        nk_cli,
        "current_model_release",
        lambda config: {
            "remote_host": "orangepi5",
            "remote_project": "/home/ubuntu/projects/neurokernel-agi-seed",
            "current": {"release_name": "current_a"},
        },
    )

    def fake_copy(remote_host, remote_path, local_path):
        copied.append((remote_path, local_path))
        local_path.write_text("model", encoding="utf-8")

    def fake_ablation(model_path, *, out_dir, config):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {
            "aggregate": {"macro_success_rate": {"hybrid_veto": 1.0, "prior_only": 0.5}},
            "interpretation": {"verdict": "hybrid_adds_value"},
            "out": str(Path(out_dir) / "gate_ablation_v4_4.json"),
        }

    monkeypatch.setattr(nk_cli, "_copy_remote_file", fake_copy)
    monkeypatch.setattr(nk_cli, "run_gate_ablation", fake_ablation)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="bench-current",
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
            ssh_connect_timeout=10,
            out_dir=str(tmp_path / "current"),
            episodes=1,
            trace_episodes=1,
            max_failures_per_env=1,
            strict=True,
        )
    )

    assert result["release_name"] == "current_a"
    assert (tmp_path / "current" / "gate_ablation_result.json").exists()
    assert copied[0][0].endswith("/artifacts/current_world_model.onnx")


def test_nk_bench_writes_top_comparable_result(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_text("model", encoding="utf-8")

    def fake_ablation(model_path, *, out_dir, config):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {
            "aggregate": {
                "macro_success_rate": {"hybrid_veto": 0.9, "prior_only": 0.7},
                "model_needed_signal_group_count": 1,
                "hybrid_veto_gain_over_prior": 0.2,
            },
            "interpretation": {"verdict": "hybrid_adds_value"},
            "out": str(Path(out_dir) / "gate_ablation_v4_4.json"),
        }

    monkeypatch.setattr(nk_cli, "run_gate_ablation", fake_ablation)

    result = nk_cli._run_action(
        argparse.Namespace(
            action="bench",
            model=str(model),
            run_name=None,
            out_dir=str(tmp_path / "baseline"),
            episodes=1,
            trace_episodes=1,
            max_failures_per_env=1,
            strict=True,
        )
    )

    assert result["interpretation"]["verdict"] == "hybrid_adds_value"
    assert (tmp_path / "baseline" / "gate_ablation_result.json").exists()
