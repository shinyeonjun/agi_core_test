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
