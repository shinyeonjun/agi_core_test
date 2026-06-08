from neurokernel_seed.harness.action_catalog import default_action_catalog
from neurokernel_seed.harness.executors.readonly_system import ReadOnlyExecutor
from neurokernel_seed.harness.model_usage import inspect_model_usage


def test_inspect_model_usage_reports_runtime_and_world_roles(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "current_runtime_action_model.pt").write_bytes(b"model")
    (artifacts / "current_world_model.onnx").write_bytes(b"world")

    report = inspect_model_usage(project_root=tmp_path, db_path=tmp_path / "harness.db")

    assert report["schema_version"] == "neurokernel-model-usage-audit-v1"
    assert report["model_roles"]["world"] == "future_state_risk_reward_predictor"
    assert report["model_roles"]["runtime_action"] == "safety_gated_action_ranker"
    assert report["slots"]["runtime_action"]["active_model"]["exists"] is True
    assert report["slots"]["world"]["live_runtime_planner"]["usable"] is False
    assert any(gap["reason"] == "world_not_live_runtime_compatible" for gap in report["gaps"])


def test_inspect_model_usage_action_is_read_only(tmp_path):
    executor = ReadOnlyExecutor(project_root=tmp_path, memory_path=tmp_path / "harness.db")

    result = executor.execute("inspect_model_usage")

    assert result.success
    assert result.result["status"] == "completed"
    assert "runtime_action" in result.result["slots"]


def test_default_catalog_includes_model_usage_inspection():
    action = default_action_catalog()["inspect_model_usage"]
    assert action.executor == "readonly_system"
    assert action.risk_level == "low"
    assert action.side_effect is False
    assert action.requires_approval is False
