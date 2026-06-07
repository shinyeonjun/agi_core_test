import json
from pathlib import Path

import pytest

from neurokernel_seed.model import release


def _make_releasable_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "training_pipeline_manifest.json",
        "world_model.pt",
        "world_model.onnx",
        "world_model.manifest.json",
        "train_result.json",
        "onnx_export.json",
    ):
        (run_dir / name).write_text("{}", encoding="utf-8")


def test_train_deploy_model_creates_versioned_remote_release(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    run_dir = tmp_path / "runs" / "release_a"

    def fake_training(config):
        _make_releasable_run(run_dir)
        return {"status": "completed", "run_name": "release_a", "run_dir": str(run_dir)}

    def fake_run_native(command):
        commands.append(command)
        class Result:
            stdout = ""
            stderr = ""
            returncode = 0
        return Result()

    monkeypatch.setattr(release, "run_training_pipeline", fake_training)
    monkeypatch.setattr(release, "_run_native", fake_run_native)

    result = release.train_deploy_model(
        release.TrainDeployModelConfig(
            features=tmp_path / "features.jsonl",
            out_dir=tmp_path / "runs",
            run_name="release_a",
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
            activate=False,
        )
    )

    assert result["status"] == "completed"
    assert result["release"]["remote_release_dir"].endswith("/artifacts/model_releases/release_a")
    assert result["release"]["status"] == "deployed"
    assert (run_dir / "model_release_manifest.json").exists()
    joined = "\n".join(" ".join(command) for command in commands)
    assert "model release already exists" in joined
    assert "artifacts/model_releases/release_a" in joined
    assert "artifacts/world_model.onnx" not in joined


def test_train_deploy_model_requires_features_when_env_missing(monkeypatch):
    monkeypatch.delenv("NEUROKERNEL_TRAIN_FEATURES", raising=False)
    with pytest.raises(release.ModelReleaseError, match="missing features"):
        release.train_deploy_model(release.TrainDeployModelConfig(run_name="release_a"))


def test_deploy_training_run_rejects_unsafe_release_name(tmp_path):
    run_dir = tmp_path / "run"
    _make_releasable_run(run_dir)
    with pytest.raises(release.ModelReleaseError):
        release.deploy_training_run(
            run_dir,
            release_name="../bad",
            remote_host="orangepi5",
            remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
        )


def test_deploy_training_run_activation_updates_pointer_only(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    _make_releasable_run(run_dir)
    commands: list[list[str]] = []
    monkeypatch.setattr(release, "_run_native", lambda command: commands.append(command))

    release.deploy_training_run(
        run_dir,
        release_name="release_b",
        remote_host="orangepi5",
        remote_project="/home/ubuntu/projects/neurokernel-agi-seed",
        activate=True,
    )

    joined = "\n".join(" ".join(command) for command in commands)
    assert "current.json" in joined
    assert "ln -sfn" in joined
    manifest = json.loads((run_dir / "model_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["activate_requested"] is True
