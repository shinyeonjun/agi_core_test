from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

RiskLevel = Literal["none", "low", "medium", "high", "forbidden"]
REGISTRY_SCHEMA_VERSION = "neurokernel-action-registry-v1"
ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
ALLOWED_DYNAMIC_EXECUTORS = {"readonly_system", "benchmark", "readonly_command"}


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

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_targets"] = list(self.allowed_targets)
        payload["test_plan"] = list(self.test_plan)
        return payload


def default_action_catalog() -> dict[str, ActionDefinition]:
    actions = [
        ActionDefinition("get_uptime", "시스템 업타임 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_disk_usage", "디스크 사용량 조회", "low", False, False, "readonly_system", {"path": {"type": "string", "default": "."}}),
        ActionDefinition("get_memory_usage", "메모리 사용량 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_cpu_temp", "CPU 온도 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_cpu_per_core_usage", "CPU 코어별 사용률 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_service_status", "서비스 상태 조회", "low", False, False, "readonly_system", {"service": {"type": "string"}}),
        ActionDefinition("tail_logs", "허용된 로그 꼬리 조회", "low", False, False, "readonly_system", {"path": {"type": "string"}, "lines": {"type": "integer", "default": 80}}),
        ActionDefinition("list_artifacts", "artifact 목록 조회", "low", False, False, "readonly_system", {"path": {"type": "string", "default": "artifacts"}}),
        ActionDefinition("get_recent_trace", "최근 trace 조회", "low", False, False, "readonly_system", {"limit": {"type": "integer", "default": 5}}),
        ActionDefinition("run_safe_benchmark", "허용된 벤치마크 실행", "low", False, False, "benchmark", {"model": {"type": "string"}, "episodes": {"type": "integer", "default": 5}}),
        ActionDefinition("write_file", "파일 쓰기", "medium", True, True, "write_file", role="mutate"),
        ActionDefinition("reboot", "시스템 재부팅", "high", True, True, "system_control", role="control"),
        ActionDefinition("delete_file", "파일 삭제", "forbidden", True, True, "forbidden", role="mutate"),
        ActionDefinition("read_secret", "시크릿 조회", "forbidden", False, True, "forbidden", role="secret"),
        ActionDefinition("git_push", "원격 저장소 push", "high", True, True, "git", role="external"),
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
    if data.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"unsupported action registry schema: {data.get('schema_version')}")
    raw_actions = data.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError("action registry actions must be a list")
    actions: dict[str, ActionDefinition] = {}
    for index, raw in enumerate(raw_actions):
        action = _action_from_registry_item(raw, source=f"registry:{registry_path}")
        if action.action_id in actions:
            raise ValueError(f"duplicate dynamic action_id: {action.action_id}")
        actions[action.action_id] = action
    return actions


def _action_from_registry_item(raw: Any, *, source: str) -> ActionDefinition:
    if not isinstance(raw, dict):
        raise ValueError("action registry item must be an object")
    action_id = str(raw.get("action_id") or "").strip()
    if not ACTION_ID_RE.match(action_id):
        raise ValueError(f"invalid action_id: {action_id}")
    title = str(raw.get("title") or "").strip()
    if not title:
        raise ValueError(f"title is required for action: {action_id}")
    risk_level = str(raw.get("risk_level") or "low")
    if risk_level not in {"none", "low", "medium", "high", "forbidden"}:
        raise ValueError(f"invalid risk_level for {action_id}: {risk_level}")
    executor = str(raw.get("executor") or "").strip()
    if executor not in ALLOWED_DYNAMIC_EXECUTORS:
        raise ValueError(f"dynamic action executor is not allowed for {action_id}: {executor}")
    side_effect = bool(raw.get("side_effect", False))
    requires_approval = bool(raw.get("requires_approval", risk_level in {"medium", "high", "forbidden"} or side_effect))
    if executor == "readonly_command" and (side_effect or requires_approval or risk_level not in {"none", "low"}):
        raise ValueError(f"readonly_command action must be low-risk read-only: {action_id}")
    params_schema = raw.get("params_schema") or {}
    if not isinstance(params_schema, dict):
        raise ValueError(f"params_schema must be an object for {action_id}")
    executor_config = raw.get("executor_config") or {}
    if not isinstance(executor_config, dict):
        raise ValueError(f"executor_config must be an object for {action_id}")
    test_plan = raw.get("test_plan")
    if not isinstance(test_plan, list) or not test_plan:
        raise ValueError(f"test_plan is required for dynamic action: {action_id}")
    test_items = tuple(item for item in test_plan if isinstance(item, dict))
    if len(test_items) != len(test_plan):
        raise ValueError(f"test_plan entries must be objects for {action_id}")
    allowed_targets = raw.get("allowed_targets") or ["local", "orangepi5"]
    if not isinstance(allowed_targets, list) or not all(isinstance(item, str) for item in allowed_targets):
        raise ValueError(f"allowed_targets must be a string list for {action_id}")
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
    )


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
