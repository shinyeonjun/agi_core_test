import json

import pytest

from neurokernel_seed.harness.action_catalog import default_action_catalog
from neurokernel_seed.harness.executors import readonly_system
from neurokernel_seed.harness.executors.readonly_system import ReadOnlyExecutor
from neurokernel_seed.harness.service import HarnessService


def test_readonly_executor_blocks_path_escape(tmp_path):
    executor = ReadOnlyExecutor(project_root=tmp_path)
    result = executor.execute("tail_logs", {"path": "../outside.log"})
    assert not result.success
    assert result.error_type


def test_readonly_executor_redacts_secret_like_output(tmp_path):
    log = tmp_path / "safe.log"
    log.write_text("token=abc123\nok\n", encoding="utf-8")
    executor = ReadOnlyExecutor(project_root=tmp_path)
    result = executor.execute("tail_logs", {"path": "safe.log", "lines": 10})
    assert result.success
    assert "[REDACTED]" in result.result["text"]


def test_readonly_executor_returns_each_cpu_core(monkeypatch, tmp_path):
    monkeypatch.setattr(readonly_system.psutil, "cpu_percent", lambda *, interval, percpu: [0.0, 12.3456, 100.0])
    executor = ReadOnlyExecutor(project_root=tmp_path)

    result = executor.execute("get_cpu_per_core_usage")

    assert result.success
    assert result.result == {
        "per_core_percent": [
            {"core": 0, "used_percent": 0.0},
            {"core": 1, "used_percent": 12.346},
            {"core": 2, "used_percent": 100.0},
        ],
        "core_count": 3,
    }


def test_readonly_executor_cpu_per_core_usage_is_read_only(monkeypatch, tmp_path):
    calls = []

    def fake_cpu_percent(*, interval, percpu):
        calls.append({"interval": interval, "percpu": percpu})
        return [7.5]

    monkeypatch.setattr(readonly_system.psutil, "cpu_percent", fake_cpu_percent)
    monkeypatch.setattr(
        readonly_system.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("get_cpu_per_core_usage must not run subprocesses"),
    )
    executor = ReadOnlyExecutor(project_root=tmp_path)

    result = executor.execute("get_cpu_per_core_usage")

    assert result.success
    assert calls == [{"interval": 0.2, "percpu": True}]


def test_default_catalog_includes_cpu_per_core_usage():
    action = default_action_catalog()["get_cpu_per_core_usage"]
    assert action.executor == "readonly_system"
    assert action.risk_level == "low"
    assert action.side_effect is False
    assert action.requires_approval is False


def test_default_catalog_includes_symptom_diagnosis_action():
    action = default_action_catalog()["diagnose_system_symptoms"]
    assert action.executor == "readonly_system"
    assert action.risk_level == "low"
    assert action.side_effect is False
    assert action.requires_approval is False
    assert action.params_schema["symptoms"]["default"] == ""
    assert action.params_schema["log_paths"]["default"] == []


def test_readonly_executor_diagnoses_symptoms_from_composed_probes(monkeypatch, tmp_path):
    log = tmp_path / "train.log"
    log.write_text("step 10 ok\nERROR out of memory while loading model\n", encoding="utf-8")
    executor = ReadOnlyExecutor(project_root=tmp_path, memory_path=tmp_path / "harness.db")
    monkeypatch.setattr(executor, "_get_uptime", lambda: {"uptime_seconds": 123.0})
    monkeypatch.setattr(executor, "_get_disk_usage", lambda path: {"path": path, "used_percent": 95.0})
    monkeypatch.setattr(executor, "_get_memory_usage", lambda: {"used_percent": 96.0, "available_bytes": 128 * 1024 * 1024})
    monkeypatch.setattr(executor, "_get_cpu_temp", lambda: {"celsius": 82.0})
    monkeypatch.setattr(executor, "_get_cpu_per_core_usage", lambda: {"per_core_percent": [{"core": 0, "used_percent": 99.0}], "core_count": 1})
    monkeypatch.setattr(executor, "_list_artifacts", lambda path: {"path": path, "exists": True, "items": []})
    monkeypatch.setattr(executor, "_get_recent_trace", lambda limit: {"traces": [{"status": "failed", "error_type": "RuntimeError"}]})

    result = executor.execute(
        "diagnose_system_symptoms",
        {
            "symptoms": "학습이 멈춘 것 같고 모델 파일도 이상해",
            "log_paths": ["train.log"],
            "log_lines": 20,
            "artifact_path": "artifacts",
            "trace_limit": 3,
        },
    )

    assert result.success
    summary = result.result["summary"]
    candidate_ids = {candidate["id"] for candidate in summary["cause_candidates"]}
    assert summary["status"] == "attention_needed"
    assert {
        "memory_pressure",
        "disk_pressure",
        "cpu_saturation",
        "thermal_throttling",
        "log_errors",
        "recent_failures",
        "artifact_path_empty",
    } <= candidate_ids
    assert "next_actions" in summary
    assert result.result["observations"]["logs"][0]["signal_counts"]["error"] == 1
    assert result.result["observations"]["logs"][0]["signal_counts"]["oom"] >= 1


def test_default_catalog_includes_code_structure_inspection():
    action = default_action_catalog()["inspect_code_structure"]
    assert action.executor == "readonly_system"
    assert action.risk_level == "low"
    assert action.side_effect is False
    assert action.requires_approval is False


def test_readonly_executor_inspects_code_structure(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "module.py").write_text("\n".join(["def f():", *["    x = 1" for _ in range(20)], "    return x"]), encoding="utf-8")
    executor = ReadOnlyExecutor(project_root=tmp_path)

    result = executor.execute(
        "inspect_code_structure",
        {"paths": ["src"], "long_file_lines": 10, "large_function_lines": 10},
    )

    assert result.success
    assert result.result["candidate_count"] == 1
    assert result.result["candidates"][0]["path"] == "src/module.py"


def test_harness_service_runs_readonly_task_and_records_result(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "list artifacts",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    task_id = created["task"]["task_id"]
    result = service.run(task_id)
    assert result["status"] == "completed"
    fetched = service.get_task(task_id)
    assert fetched["task"]["status"] == "completed"
    assert any(event["event_type"] == "completed" for event in fetched["events"])


def test_harness_service_keeps_approval_task_waiting(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "write something",
            "target": "orangepi5",
            "allowed_actions": ["write_file"],
            "risk_level": "medium",
            "requires_approval": True,
        }
    )
    result = service.run(created["task"]["task_id"])
    assert result["status"] == "waiting_approval"


def test_harness_create_task_rejects_raw_shell(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)
    try:
        service.create_task({"goal": "bad", "target": "orangepi5", "allowed_actions": ["raw_shell"]})
    except Exception as exc:
        assert "unknown action" in str(exc)
    else:
        raise AssertionError("raw_shell should be rejected")
