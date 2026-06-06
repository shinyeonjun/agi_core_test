import json
from pathlib import Path

import pytest

from neurokernel_seed.harness.action_catalog import REGISTRY_SCHEMA_VERSION, REGISTRY_SCHEMA_VERSION_V2, build_action_catalog, load_action_registry
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


def test_dynamic_registry_v2_loads_contract_fields(tmp_path):
    registry = tmp_path / "actions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION_V2,
                "actions": [
                    {
                        "action_id": "get_test_probe_v2",
                        "version": "1.2.3",
                        "title": "테스트 probe v2",
                        "description": "계약 기반 registry v2 동작 확인",
                        "status": "active",
                        "risk_level": "low",
                        "side_effect": False,
                        "requires_approval": False,
                        "role": "readonly_action",
                        "executor": "readonly_command",
                        "target": ["local"],
                        "inputs_schema": {"type": "object", "properties": {}, "required": []},
                        "outputs_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                        "executor_config": {
                            "command": ["python", "-c", "import json; print(json.dumps({'ok': True}))"],
                            "output": "json",
                            "timeout_seconds": 5,
                        },
                        "test_plan": ["ok 필드를 반환한다"],
                        "examples": ["테스트 probe 확인"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = build_action_catalog(registry_path=registry)
    action = catalog["get_test_probe_v2"]

    assert action.version == "1.2.3"
    assert action.description == "계약 기반 registry v2 동작 확인"
    assert action.status == "active"
    assert action.inputs_schema["type"] == "object"
    assert action.outputs_schema["required"] == ["ok"]
    assert action.test_plan == ({"name": "check_1", "assertions": ["ok 필드를 반환한다"]},)
    assert action.examples == ("테스트 probe 확인",)


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


def test_dynamic_registry_v2_rejects_missing_contract_fields(tmp_path):
    registry = tmp_path / "actions.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION_V2,
                "actions": [
                    {
                        "action_id": "get_incomplete_probe",
                        "version": "1.0.0",
                        "title": "불완전 probe",
                        "status": "active",
                        "risk_level": "low",
                        "side_effect": False,
                        "requires_approval": False,
                        "role": "readonly_action",
                        "executor": "readonly_command",
                        "target": ["local"],
                        "inputs_schema": {},
                        "outputs_schema": {},
                        "test_plan": ["never loads"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required action spec fields"):
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
