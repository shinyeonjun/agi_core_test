import argparse
import json
from pathlib import Path

from neurokernel_seed import nk_cli


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
    assert "deploy-use" in help_text
    assert "top" in help_text


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
