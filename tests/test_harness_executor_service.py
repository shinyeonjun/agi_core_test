import json

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

