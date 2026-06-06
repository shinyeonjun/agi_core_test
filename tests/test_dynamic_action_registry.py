import json
from pathlib import Path

import pytest

from neurokernel_seed.harness.action_catalog import REGISTRY_SCHEMA_VERSION, build_action_catalog, load_action_registry
from neurokernel_seed.harness.service import HarnessService


def test_dynamic_registry_action_can_be_loaded_and_executed(tmp_path):
    registry = tmp_path / "actions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "actions": [
                    {
                        "action_id": "get_test_probe",
                        "title": "테스트 probe",
                        "risk_level": "low",
                        "side_effect": False,
                        "requires_approval": False,
                        "executor": "readonly_command",
                        "allowed_targets": ["local"],
                        "executor_config": {
                            "command": ["python", "-c", "import json; print(json.dumps({'ok': True, 'value': 7}))"],
                            "output": "json",
                            "timeout_seconds": 5,
                        },
                        "test_plan": [{"name": "returns_ok", "assertions": ["ok is true"]}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    catalog = build_action_catalog(registry_path=registry)
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, catalog=catalog)
    created = service.create_task(
        {
            "goal": "run dynamic action",
            "target": "local",
            "allowed_actions": ["get_test_probe"],
            "success_criteria": ["probe returns ok"],
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )

    result = service.run(created["task"]["task_id"])

    assert result["status"] == "completed"
    assert result["action"] == "get_test_probe"
    assert result["execution_result"]["result"] == {"ok": True, "value": 7}


def test_dynamic_registry_rejects_entries_without_tests(tmp_path):
    registry = tmp_path / "actions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "actions": [
                    {
                        "action_id": "get_no_tests",
                        "title": "테스트 없는 action",
                        "risk_level": "low",
                        "side_effect": False,
                        "requires_approval": False,
                        "executor": "readonly_command",
                        "executor_config": {"command": ["python", "-c", "print('{}')"], "output": "json"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="test_plan is required"):
        load_action_registry(registry)


def test_dynamic_registry_rejects_shell_commands(tmp_path):
    registry = tmp_path / "actions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "actions": [
                    {
                        "action_id": "get_shell_probe",
                        "title": "shell probe",
                        "risk_level": "low",
                        "side_effect": False,
                        "requires_approval": False,
                        "executor": "readonly_command",
                        "executor_config": {"command": ["bash", "-lc", "echo '{}'"], "output": "json"},
                        "test_plan": [{"name": "blocked", "assertions": ["never loads"]}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="executable is not allowed"):
        load_action_registry(registry)
