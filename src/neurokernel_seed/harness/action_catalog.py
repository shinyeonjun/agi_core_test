from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RiskLevel = Literal["none", "low", "medium", "high", "forbidden"]


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

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_targets"] = list(self.allowed_targets)
        return payload


def default_action_catalog() -> dict[str, ActionDefinition]:
    actions = [
        ActionDefinition("get_uptime", "시스템 업타임 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_disk_usage", "디스크 사용량 조회", "low", False, False, "readonly_system", {"path": {"type": "string", "default": "."}}),
        ActionDefinition("get_memory_usage", "메모리 사용량 조회", "low", False, False, "readonly_system"),
        ActionDefinition("get_cpu_temp", "CPU 온도 조회", "low", False, False, "readonly_system"),
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


def public_catalog(catalog: dict[str, ActionDefinition] | None = None) -> list[dict[str, Any]]:
    return [action.as_dict() for action in sorted((catalog or default_action_catalog()).values(), key=lambda item: item.action_id)]


def require_action(action_id: str, catalog: dict[str, ActionDefinition] | None = None) -> ActionDefinition:
    catalog = catalog or default_action_catalog()
    if action_id not in catalog:
        raise KeyError(f"unknown action: {action_id}")
    return catalog[action_id]

