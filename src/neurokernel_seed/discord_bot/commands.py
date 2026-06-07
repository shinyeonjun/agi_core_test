from __future__ import annotations

import json
from typing import Any


DEFAULT_MODEL_PATH = "artifacts/world_model_slot_v2_model_needed_v3.onnx"

PRESET_ACTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "uptime": ("get_uptime", {}),
    "disk": ("get_disk_usage", {"path": "."}),
    "memory": ("get_memory_usage", {}),
    "mem": ("get_memory_usage", {}),
    "temp": ("get_cpu_temp", {}),
    "artifacts": ("list_artifacts", {"path": "artifacts"}),
    "trace": ("get_recent_trace", {"limit": 5}),
    "code": ("inspect_code_structure", {"paths": ["src", "tests"]}),
    "code-structure": ("inspect_code_structure", {"paths": ["src", "tests"]}),
    "structure": ("inspect_code_structure", {"paths": ["src", "tests"]}),
}


def build_preset_task(name: str, *, target: str = "orangepi5") -> dict[str, Any]:
    key = name.strip().lower()
    if key not in PRESET_ACTIONS:
        raise ValueError(f"unknown preset: {name}")
    action_id, params = PRESET_ACTIONS[key]
    return {
        "goal": f"discord preset: {key}",
        "target": target,
        "context": {"params": params},
        "allowed_actions": [action_id],
        "blocked_actions": [],
        "success_criteria": [f"{action_id} completes"],
        "risk_level": "low",
        "requires_approval": False,
        "timeout_seconds": 60,
        "mode": "readonly",
    }


def build_benchmark_task(*, model: str = DEFAULT_MODEL_PATH, episodes: int = 3, target: str = "orangepi5") -> dict[str, Any]:
    return {
        "goal": "discord preset: safe benchmark",
        "target": target,
        "context": {"params": {"model": model, "episodes": episodes}},
        "allowed_actions": ["run_safe_benchmark"],
        "blocked_actions": [],
        "success_criteria": ["benchmark completes"],
        "risk_level": "low",
        "requires_approval": False,
        "timeout_seconds": 300,
        "mode": "readonly",
    }


def parse_task_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("task JSON must be an object")
    return payload


def compact_json(payload: Any, *, max_chars: int = 1600) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n... truncated ..."


def format_code_block(payload: Any, *, max_chars: int = 1600) -> str:
    return f"```json\n{compact_json(payload, max_chars=max_chars)}\n```"


def summarize_status(payload: dict[str, Any]) -> str:
    tasks = payload.get("recent_tasks") or []
    traces = payload.get("recent_traces") or []
    lines = ["Core 상태: OK" if payload.get("health", {}).get("ok") else "Core 상태: 확인 필요"]
    lines.append(f"최근 task: {len(tasks)}개 / 최근 trace: {len(traces)}개")
    for task in tasks[:3]:
        lines.append(f"- {task.get('task_id')} · {task.get('status')} · {task.get('goal')}")
    return "\n".join(lines)


def help_text(prefix: str) -> str:
    return "\n".join(
        [
            "NeuroKernel Discord v0",
            f"`{prefix} status` - Core 상태 확인",
            f"`{prefix} actions` - 허용된 액션 목록",
            f"`{prefix} run uptime|disk|memory|temp|artifacts|trace` - 안전한 읽기 작업 실행",
            f"`{prefix} benchmark [episodes]` - 안전 벤치마크 실행",
            f"`{prefix} ask <말>` - 실행 없이 대화",
            f"`{prefix} plan <말>` - 자연어를 작업 계획으로 바꾸고 미리 점검",
            f"`{prefix} do <말>` - 안전한 조회 작업만 실행",
            f"`{prefix} task {{...json...}}` - 개발자용 작업 생성 후 미리 점검",
            f"`{prefix} run-task <task_id>` - 생성된 task 실행",
            f"`{prefix} approve <task_id> [reason]` / `{prefix} reject <task_id> [reason]`",
            "`prefs show|set|forget|reset` - 답변 선호 저장/조회",
            "`memory recent|context` - 최근 대화 기억 확인",
        ]
    )
