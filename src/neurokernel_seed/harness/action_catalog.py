from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RiskLevel = Literal["none", "low", "medium", "high", "forbidden"]
ActionStatus = Literal["proposed", "drafted", "tested", "activation_candidate", "active", "disabled", "deprecated", "rejected"]
REGISTRY_SCHEMA_VERSION = "neurokernel-action-registry-v1"
REGISTRY_SCHEMA_VERSION_V2 = "neurokernel-action-registry-v2"
ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ALLOWED_DYNAMIC_EXECUTORS = {"readonly_system", "benchmark", "readonly_command"}
ALLOWED_ACTION_STATUSES = {"proposed", "drafted", "tested", "activation_candidate", "active", "disabled", "deprecated", "rejected"}


@dataclass(frozen=True)
class ActionDefinition:
    action_id: str
    title: str
    risk_level: RiskLevel
    side_effect: bool
    requires_approval: bool
    executor: str
    params_schema: dict[str, Any] = field(default_factory=dict)
    role: str = "observe"
    allowed_targets: tuple[str, ...] = ("local", "orangepi5")
    executor_config: dict[str, Any] = field(default_factory=dict)
    test_plan: tuple[dict[str, Any], ...] = ()
    source: str = "builtin"
    version: str = "1.0.0"
    description: str = ""
    status: ActionStatus = "active"
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    outputs_schema: dict[str, Any] = field(default_factory=dict)
    examples: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_targets"] = list(self.allowed_targets)
        payload["test_plan"] = list(self.test_plan)
        payload["examples"] = list(self.examples)
        return payload


def default_action_catalog() -> dict[str, ActionDefinition]:
    actions = [
        ActionDefinition("get_uptime", "시스템 업타임 조회", "low", False, False, "readonly_system", description="시스템 업타임을 read-only로 조회한다."),
        ActionDefinition("get_disk_usage", "디스크 사용량 조회", "low", False, False, "readonly_system", {"path": {"type": "string", "default": "."}}, description="디스크 사용량을 read-only로 조회한다."),
        ActionDefinition("get_memory_usage", "메모리 사용량 조회", "low", False, False, "readonly_system", description="메모리 사용량을 read-only로 조회한다."),
        ActionDefinition("get_cpu_temp", "CPU 온도 조회", "low", False, False, "readonly_system", description="CPU 온도를 read-only로 조회한다."),
        ActionDefinition("get_cpu_per_core_usage", "CPU 코어별 사용률 조회", "low", False, False, "readonly_system", description="CPU 코어별 사용률을 read-only로 조회한다."),
        ActionDefinition("get_service_status", "서비스 상태 조회", "low", False, False, "readonly_system", {"service": {"type": "string"}}, description="허용된 서비스 상태를 조회한다."),
        ActionDefinition("tail_logs", "허용된 로그 꼬리 조회", "low", False, False, "readonly_system", {"path": {"type": "string"}, "lines": {"type": "integer", "default": 80}}, description="허용된 경로의 로그 일부를 조회한다."),
        ActionDefinition("list_artifacts", "artifact 목록 조회", "low", False, False, "readonly_system", {"path": {"type": "string", "default": "artifacts"}}, description="artifact 디렉터리 목록을 조회한다."),
        ActionDefinition("get_recent_trace", "최근 trace 조회", "low", False, False, "readonly_system", {"limit": {"type": "integer", "default": 5}}, description="최근 실행 trace를 조회한다."),
        ActionDefinition(
            "diagnose_system_symptoms",
            "느려짐 학습 중단 모델 파일 이상 자동 진단",
            "low",
            False,
            False,
            "readonly_system",
            {
                "symptoms": {"type": "string", "default": ""},
                "log_paths": {"type": "array", "items": {"type": "string"}, "default": []},
                "log_lines": {"type": "integer", "default": 80},
                "artifact_path": {"type": "string", "default": "artifacts"},
                "trace_limit": {"type": "integer", "default": 5},
                "service": {"type": "string", "default": ""},
            },
            description="CPU, 메모리, 디스크, 온도, 로그, 최근 trace를 묶어 증상 기반 진단 요약을 만든다.",
        ),
        ActionDefinition(
            "inspect_code_structure",
            "Code structure inspection",
            "low",
            False,
            False,
            "readonly_system",
            {
                "paths": {"type": "array", "items": {"type": "string"}, "default": ["src", "tests"]},
                "max_files": {"type": "integer", "default": 250},
                "max_results": {"type": "integer", "default": 20},
                "long_file_lines": {"type": "integer", "default": 300},
                "very_long_file_lines": {"type": "integer", "default": 700},
                "large_function_lines": {"type": "integer", "default": 80},
                "many_functions": {"type": "integer", "default": 18},
                "many_classes": {"type": "integer", "default": 8},
            },
            description="Read-only scan for long files, large functions, and responsibility-heavy Python modules.",
            examples=("Find refactor candidates before self-patch.",),
        ),
        ActionDefinition(
            "inspect_work_pipeline",
            "Work pipeline inspection",
            "low",
            False,
            False,
            "readonly_system",
            {
                "limit": {"type": "integer", "default": 20},
                "stale_after_seconds": {"type": "integer", "default": 300},
            },
            description="Read-only inspection of work items, jobs, recent agent events, and stale running jobs.",
            examples=("Explain what is waiting, running, stale, or ready for approval.",),
        ),
        ActionDefinition("run_safe_benchmark", "허용된 벤치마크 실행", "low", False, False, "benchmark", {"model": {"type": "string"}, "episodes": {"type": "integer", "default": 5}}, description="허용된 미니월드 벤치마크를 실행한다."),
        ActionDefinition("write_file", "파일 쓰기", "medium", True, True, "write_file", role="mutate", description="파일을 쓴다."),
        ActionDefinition("reboot", "시스템 재부팅", "high", True, True, "system_control", role="control", description="시스템을 재부팅한다."),
        ActionDefinition("delete_file", "파일 삭제", "forbidden", True, True, "forbidden", role="mutate", description="파일을 삭제한다."),
        ActionDefinition("read_secret", "시크릿 조회", "forbidden", False, True, "forbidden", role="secret", description="시크릿을 조회한다."),
        ActionDefinition("git_push", "원격 저장소 push", "high", True, True, "git", role="external", description="원격 저장소에 push한다."),
    ]
    return {action.action_id: action for action in actions}


def build_action_catalog(*, registry_path: str | Path | None = None) -> dict[str, ActionDefinition]:
    catalog = default_action_catalog()
    resolved = registry_path if registry_path is not None else os.environ.get("NEUROKERNEL_ACTION_REGISTRY")
    if resolved:
        catalog.update(load_action_registry(resolved))
    return catalog


def load_action_registry(path: str | Path) -> dict[str, ActionDefinition]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"action registry not found: {registry_path}")
    data = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("action registry must be a JSON object")
    schema_version = data.get("schema_version")
    if schema_version not in {REGISTRY_SCHEMA_VERSION, REGISTRY_SCHEMA_VERSION_V2}:
        raise ValueError(f"unsupported action registry schema: {data.get('schema_version')}")
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("action registry actions must be a list")
    actions: dict[str, ActionDefinition] = {}
    for index, raw in enumerate(raw_actions):
        action = _action_from_registry_item(raw, source=f"registry:{registry_path}", schema_version=str(schema_version))
        if action.action_id in actions:
            raise ValueError(f"duplicate dynamic action_id: {action.action_id}")
        actions[action.action_id] = action
    return actions


def _action_from_registry_item(raw: Any, *, source: str, schema_version: str) -> ActionDefinition:
    if not isinstance(raw, dict):
        raise ValueError("action registry item must be an object")
    strict_v2 = schema_version == REGISTRY_SCHEMA_VERSION_V2
    if strict_v2:
        _require_fields(
            raw,
            (
                "action_id",
                "version",
                "title",
                "description",
                "status",
                "risk_level",
                "side_effect",
                "requires_approval",
                "role",
                "executor",
                "target",
                "inputs_schema",
                "outputs_schema",
                "test_plan",
            ),
        )
    action_id = str(raw.get("action_id") or "").strip()
    if not ACTION_ID_RE.match(action_id):
        raise ValueError(f"invalid action_id: {action_id}")
    version = str(raw.get("version") or "1.0.0").strip()
    if strict_v2 and not version:
        raise ValueError(f"version is required for action: {action_id}")
    title = str(raw.get("title") or "").strip()
    if not title:
        raise ValueError(f"title is required for action: {action_id}")
    description = str(raw.get("description") or raw.get("purpose") or "").strip()
    if strict_v2 and not description:
        raise ValueError(f"description is required for action: {action_id}")
    status = str(raw.get("status") or "active").strip()
    if status not in ALLOWED_ACTION_STATUSES:
        raise ValueError(f"invalid status for {action_id}: {status}")
    risk_level = str(raw.get("risk_level") or ("low" if not strict_v2 else ""))
    if risk_level not in {"none", "low", "medium", "high", "forbidden"}:
        raise ValueError(f"invalid risk_level for {action_id}: {risk_level}")
    executor = str(raw.get("executor") or "").strip()
    if executor not in ALLOWED_DYNAMIC_EXECUTORS:
        raise ValueError(f"dynamic action executor is not allowed for {action_id}: {executor}")
    side_effect = bool(raw.get("side_effect", False))
    requires_approval = bool(raw.get("requires_approval", risk_level in {"medium", "high", "forbidden"} or side_effect))
    if executor == "readonly_command" and (side_effect or requires_approval or risk_level not in {"none", "low"}):
        raise ValueError(f"readonly_command action must be low-risk read-only: {action_id}")
    params_schema = raw.get("inputs_schema") if "inputs_schema" in raw else raw.get("params_schema") or raw.get("inputs") or {}
    if not isinstance(params_schema, dict):
        raise ValueError(f"inputs_schema must be an object for {action_id}")
    outputs_schema = raw.get("outputs_schema") if "outputs_schema" in raw else raw.get("outputs") or {}
    if not isinstance(outputs_schema, dict):
        raise ValueError(f"outputs_schema must be an object for {action_id}")
    executor_config = raw.get("executor_config") or {}
    if not isinstance(executor_config, dict):
        raise ValueError(f"executor_config must be an object for {action_id}")
    test_plan = raw.get("test_plan")
    if status == "active" and (not isinstance(test_plan, list) or not test_plan):
        raise ValueError(f"test_plan is required for dynamic action: {action_id}")
    test_items = _normalize_test_plan(test_plan or [], action_id=action_id)
    raw_targets = raw.get("target") if "target" in raw else raw.get("allowed_targets")
    allowed_targets = raw_targets or ["local", "orangepi5"]
    if not isinstance(allowed_targets, list) or not all(isinstance(item, str) for item in allowed_targets):
        raise ValueError(f"target must be a string list for {action_id}")
    examples = raw.get("examples") or []
    if not isinstance(examples, list) or not all(isinstance(item, str) for item in examples):
        raise ValueError(f"examples must be a string list for {action_id}")
    _validate_readonly_command_config(action_id, executor, executor_config)
    return ActionDefinition(
        action_id=action_id,
        title=title,
        risk_level=risk_level,  # type: ignore[arg-type]
        side_effect=side_effect,
        requires_approval=requires_approval,
        executor=executor,
        params_schema=params_schema,
        role=str(raw.get("role") or "observe"),
        allowed_targets=tuple(allowed_targets),
        executor_config=executor_config,
        test_plan=test_items,
        source=source,
        version=version,
        description=description or title,
        status=status,  # type: ignore[arg-type]
        inputs_schema=params_schema,
        outputs_schema=outputs_schema,
        examples=tuple(examples),
    )


def _require_fields(raw: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in raw]
    if missing:
        raise ValueError(f"missing required action spec fields: {missing}")


def _normalize_test_plan(test_plan: list[Any], *, action_id: str) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(test_plan):
        if isinstance(item, str):
            text = item.strip()
            if not text:
                raise ValueError(f"test_plan entries must not be empty for {action_id}")
            items.append({"name": f"check_{index + 1}", "assertions": [text]})
            continue
        if isinstance(item, dict):
            items.append(item)
            continue
        raise ValueError(f"test_plan entries must be strings or objects for {action_id}")
    return tuple(items)


def _validate_readonly_command_config(action_id: str, executor: str, config: dict[str, Any]) -> None:
    if executor != "readonly_command":
        return
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError(f"readonly_command requires argv command for {action_id}")
    executable = Path(command[0]).name.lower()
    if executable not in {"python", "python3", "python.exe"}:
        raise ValueError(f"readonly_command executable is not allowed for {action_id}: {command[0]}")
    timeout = int(config.get("timeout_seconds", 5))
    if timeout < 1 or timeout > 30:
        raise ValueError(f"readonly_command timeout must be between 1 and 30 for {action_id}")
    output = str(config.get("output") or "json")
    if output not in {"json", "text"}:
        raise ValueError(f"readonly_command output must be json or text for {action_id}")


def public_catalog(catalog: dict[str, ActionDefinition] | None = None) -> list[dict[str, Any]]:
    return [action.as_dict() for action in sorted((catalog or default_action_catalog()).values(), key=lambda item: item.action_id)]


def require_action(action_id: str, catalog: dict[str, ActionDefinition] | None = None) -> ActionDefinition:
    catalog = catalog or default_action_catalog()
    if action_id not in catalog:
        raise KeyError(f"unknown action: {action_id}")
    return catalog[action_id]
