import json

import pytest

from neurokernel_seed.harness.action_catalog import default_action_catalog
from neurokernel_seed.harness.executors import readonly_system
from neurokernel_seed.harness.executors.readonly_system import ReadOnlyExecutor
from neurokernel_seed.harness.executors.benchmark import BenchmarkExecutor
from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.service import HarnessService


class FakeRuntimePolicy:
    def __init__(self, ranked_actions=None):
        self.ranked_actions = ranked_actions or []

    def rank(self, *, task, decisions):
        scores = {}
        for index, item in enumerate(decisions):
            action_id = item["action_id"]
            scores[action_id] = {
                "model_used": True,
                "reason": "test_runtime_policy",
                "score": 10.0 - self.ranked_actions.index(action_id) if action_id in self.ranked_actions else float(index),
            }
        return {
            "model_used": True,
            "reason": "ranked_by_runtime_action_model",
            "ranked_actions": list(self.ranked_actions),
            "scores": scores,
            "top_action": self.ranked_actions[0] if self.ranked_actions else None,
        }


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


def test_benchmark_executor_requires_explicit_model(tmp_path):
    executor = BenchmarkExecutor(project_root=tmp_path)

    result = executor.execute("run_safe_benchmark", {"episodes": 1})

    assert not result.success
    assert result.error_type == "ValueError"


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


def test_default_catalog_includes_work_pipeline_inspection():
    action = default_action_catalog()["inspect_work_pipeline"]
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


def test_readonly_executor_inspects_work_pipeline(tmp_path):
    executor = ReadOnlyExecutor(project_root=tmp_path, memory_path=tmp_path / "harness.db")

    result = executor.execute("inspect_work_pipeline", {"limit": 5, "stale_after_seconds": 60})

    assert result.success
    assert result.result["schema_version"] == "neurokernel-work-pipeline-status-v1"
    assert "summary" in result.result


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

    with HarnessMemory(db) as memory:
        experiences = memory.recent_experiences(task_id=task_id)
        assert len(experiences) == 1
        experience = memory.get_experience(experiences[0]["experience_id"])

    assert experience["phase"] == "run"
    assert experience["status"] == "completed"
    assert experience["decision_policy"] == "deterministic_safety_first"
    assert experience["model_used"] is False
    assert experience["model_unavailable_reason"] == "model_unavailable"
    assert len(experience["candidates"]) == 1
    candidate = experience["candidates"][0]
    assert candidate["action_id"] == "list_artifacts"
    assert candidate["selected"] is True
    assert candidate["executed"] is True
    assert candidate["execution_result_known"] is True
    assert candidate["target_mask_json"] == {
        "success": True,
        "reward": True,
        "duration_seconds": True,
        "failure_present": True,
    }


def test_harness_service_uses_runtime_model_to_rank_safe_candidates(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path, runtime_policy=FakeRuntimePolicy(["get_memory_usage", "list_artifacts"]))
    created = service.create_task(
        {
            "goal": "choose the better readonly action",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts", "get_memory_usage"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )

    result = service.run(created["task"]["task_id"])

    assert result["status"] == "completed"
    assert result["action"] == "get_memory_usage"
    with HarnessMemory(db) as memory:
        experience = memory.get_experience(memory.recent_experiences(task_id=created["task"]["task_id"])[0]["experience_id"])
    assert experience["decision_policy"] == "runtime_model_ranked_safety_gated"
    assert experience["model_used"] is True
    selected = [item for item in experience["candidates"] if item["selected"]]
    assert selected[0]["action_id"] == "get_memory_usage"
    assert selected[0]["model_score_json"]["model_used"] is True


def test_harness_service_runtime_model_does_not_bypass_approval(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, runtime_policy=FakeRuntimePolicy(["write_file"]))
    created = service.create_task(
        {
            "goal": "write something",
            "target": "orangepi5",
            "allowed_actions": ["write_file"],
            "risk_level": "low",
            "requires_approval": False,
        }
    )

    result = service.run(created["task"]["task_id"])

    assert result["status"] == "waiting_approval"
    assert result["safety"]["decision"] == "requires_approval"


def test_harness_service_rejects_unknown_preference_key(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)

    with pytest.raises(ValueError, match="unknown preference key"):
        service.set_preference({"user_id": "discord:1", "key": "private_token", "value": "abc"})


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
