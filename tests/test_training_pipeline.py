import json
from pathlib import Path

import pytest

from neurokernel_seed.model import pipeline


def _write_slot_features(tmp_path: Path) -> Path:
    features = tmp_path / "features_slot.jsonl"
    manifest = features.with_suffix(features.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "rows": 1,
                "schema_version": "neurokernel-slot-transition-v2",
                "dataset_profile": "test",
                "input_dim": 3,
                "target_dim": 2,
                "splits": ["train", "test"],
            }
        ),
        encoding="utf-8",
    )
    features.write_text('{"row_index": 0}\n', encoding="utf-8")
    return features


def test_training_pipeline_runs_full_model_lifecycle(tmp_path, monkeypatch):
    features = _write_slot_features(tmp_path)
    calls: list[str] = []

    def fake_validate(path):
        calls.append("validate")
        return {"rows": 1, "schema_version": "neurokernel-slot-transition-v2", "input_dim": 3, "target_dim": 2}

    def fake_gates(path):
        calls.append("gates")
        return {"passed": True, "candidate_groups": 1, "schema_version": "neurokernel-slot-transition-v2"}

    def fake_train(features_path, out_path, **kwargs):
        calls.append("train")
        Path(out_path).write_text("checkpoint", encoding="utf-8")
        return {"checkpoint": str(out_path), "device": kwargs["device"]}

    def fake_eval(checkpoint_path, features_path, split, device):
        calls.append("eval")
        return {"checkpoint": str(checkpoint_path), "features": str(features_path), "split": split, "device": device, "loss": 0.1}

    def fake_ranking(checkpoint_path, features_path, split, device):
        calls.append("ranking")
        return {
            "checkpoint": str(checkpoint_path),
            "features": str(features_path),
            "split": split,
            "device": device,
            "candidate_groups": 1,
            "grouped_top1_action_accuracy": 1.0,
            "pairwise_ranking_accuracy": 1.0,
        }

    def fake_export(checkpoint_path, out_path, verify, dynamic_batch):
        calls.append("export")
        out = Path(out_path)
        out.write_text("onnx", encoding="utf-8")
        out.with_suffix(".manifest.json").write_text("{}", encoding="utf-8")
        return {"checkpoint": str(checkpoint_path), "onnx": str(out), "dynamic_batch": dynamic_batch}

    def fake_ablation(model_path, *, out_dir, config):
        calls.append("ablation")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        report = Path(out_dir) / "gate_ablation_v4_4.json"
        report.write_text("{}", encoding="utf-8")
        return {"mode_passed": {"hybrid": False, "hybrid_veto": True}, "out": str(report)}

    monkeypatch.setattr(pipeline, "validate_slot_features", fake_validate)
    monkeypatch.setattr(pipeline, "check_slot_dataset_gates", fake_gates)
    monkeypatch.setattr(pipeline, "train_world_model", fake_train)
    monkeypatch.setattr(pipeline, "eval_world_model", fake_eval)
    monkeypatch.setattr(pipeline, "eval_action_ranking", fake_ranking)
    monkeypatch.setattr(pipeline, "export_onnx", fake_export)
    monkeypatch.setattr(pipeline, "run_gate_ablation", fake_ablation)

    result = pipeline.run_training_pipeline(
        pipeline.TrainingPipelineConfig(
            features=features,
            out_dir=tmp_path / "runs",
            run_name="slot_auto",
            device="cuda",
            strict=True,
        )
    )

    assert result["status"] == "completed"
    assert calls == ["validate", "gates", "train", "eval", "ranking", "export", "ablation"]
    run_dir = tmp_path / "runs" / "slot_auto"
    assert (run_dir / "world_model.pt").exists()
    assert (run_dir / "world_model.onnx").exists()
    assert json.loads((run_dir / "training_pipeline_manifest.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_training_pipeline_stops_when_dataset_gates_fail(tmp_path, monkeypatch):
    features = _write_slot_features(tmp_path)
    monkeypatch.setattr(pipeline, "validate_slot_features", lambda path: {"rows": 1, "schema_version": "neurokernel-slot-transition-v2"})
    monkeypatch.setattr(pipeline, "check_slot_dataset_gates", lambda path: {"passed": False, "audit_shortage_report": [{"family": "tool"}]})

    with pytest.raises(pipeline.TrainingPipelineError):
        pipeline.run_training_pipeline(
            pipeline.TrainingPipelineConfig(
                features=features,
                out_dir=tmp_path / "runs",
                run_name="bad_data",
            )
        )

    manifest = json.loads((tmp_path / "runs" / "bad_data" / "training_pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "TrainingPipelineError"
    assert manifest["error_payload"]["passed"] is False
