from __future__ import annotations

from typing import Any


OUTPUT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cpu_usage": ("cpu", "core", "load", "usage", "코어", "부하", "사용률"),
    "cpu_temperature": ("temperature", "temp", "thermal", "온도", "발열"),
    "memory_usage": ("memory", "mem", "ram", "메모리", "램"),
    "disk_usage": ("disk", "storage", "space", "capacity", "저장", "공간", "용량", "디스크"),
    "uptime": ("uptime", "boot", "부팅", "켜진", "업타임"),
    "service_status": ("service", "systemd", "서비스"),
    "logs": ("log", "logs", "tail", "로그"),
    "artifacts": ("artifact", "artifacts", "model file", "모델파일", "아티팩트", "파일"),
    "traces": ("trace", "history", "recent", "트레이스", "최근", "이력"),
    "system_diagnosis": ("diagnose", "diagnosis", "symptom", "진단", "증상", "원인"),
    "code_structure": ("code", "structure", "코드", "구조"),
    "work_pipeline": ("work", "pipeline", "job", "작업", "파이프라인", "큐"),
}

ACTION_OUTPUTS: dict[str, tuple[str, ...]] = {
    "diagnose_system_symptoms": ("system_diagnosis", "uptime", "disk_usage", "memory_usage", "cpu_temperature", "cpu_usage", "artifacts", "traces"),
    "get_cpu_per_core_usage": ("cpu_usage",),
    "get_cpu_temp": ("cpu_temperature",),
    "get_disk_usage": ("disk_usage",),
    "get_memory_usage": ("memory_usage",),
    "get_recent_trace": ("traces",),
    "get_service_status": ("service_status",),
    "get_uptime": ("uptime",),
    "inspect_code_structure": ("code_structure",),
    "inspect_work_pipeline": ("work_pipeline",),
    "list_artifacts": ("artifacts",),
    "tail_logs": ("logs",),
}


def infer_required_outputs(request_text: str, task: dict[str, Any] | None = None) -> list[str]:
    text = str(request_text or "").lower()
    required = {name for name, keywords in OUTPUT_KEYWORDS.items() if any(keyword.lower() in text for keyword in keywords)}
    if not required and task:
        for action_id in _task_allowed_actions(task):
            required.update(ACTION_OUTPUTS.get(action_id, ()))
    if "cpu_usage" in required and "cpu_temperature" in required and not any(token in text for token in ("온도", "발열", "temp", "thermal", "temperature")):
        required.discard("cpu_temperature")
    return sorted(required)


def answered_outputs_from_core_result(core_result: dict[str, Any] | None) -> list[str]:
    if not isinstance(core_result, dict):
        return []
    answered: set[str] = set()
    execution = core_result.get("execution_result") if isinstance(core_result.get("execution_result"), dict) else {}
    action_id = str(core_result.get("action") or execution.get("action_id") or "")
    if execution.get("success") is True:
        answered.update(ACTION_OUTPUTS.get(action_id, ()))
        answered.update(_outputs_from_result_payload(execution.get("result")))
    return sorted(answered)


def answer_quality(required_outputs: list[str], answered_outputs: list[str], core_result: dict[str, Any] | None) -> str:
    required = set(required_outputs)
    answered = set(answered_outputs)
    status = str((core_result or {}).get("status") or "").lower() if isinstance(core_result, dict) else ""
    if status in {"waiting_approval", "ready", "validated"}:
        return "pending"
    if status == "failed":
        return "failed"
    if required:
        if required.issubset(answered):
            return "complete"
        if answered:
            return "partial"
        return "failed"
    execution = (core_result or {}).get("execution_result") if isinstance(core_result, dict) else {}
    if isinstance(execution, dict) and execution.get("success") is True:
        return "complete"
    if status == "completed":
        return "complete"
    return "unknown"


def interaction_contract_from_runtime(
    *,
    request_text: str,
    response_text: str,
    task: dict[str, Any] | None,
    core_result: dict[str, Any] | None,
) -> dict[str, Any]:
    required = infer_required_outputs(request_text, task)
    answered = answered_outputs_from_core_result(core_result)
    missing = sorted(set(required) - set(answered))
    quality = answer_quality(required, answered, core_result)
    execution = core_result.get("execution_result") if isinstance(core_result, dict) and isinstance(core_result.get("execution_result"), dict) else {}
    return {
        "request_text": request_text,
        "response_text": response_text,
        "required_outputs": required,
        "answered_outputs": answered,
        "missing_outputs": missing,
        "answer_quality": quality,
        "task_status": str(core_result.get("status") or "") if isinstance(core_result, dict) else "",
        "action_id": str(core_result.get("action") or execution.get("action_id") or "") if isinstance(core_result, dict) else "",
        "success": bool(execution.get("success")) if execution else None,
    }


def _task_allowed_actions(task: dict[str, Any]) -> list[str]:
    allowed = task.get("allowed_actions")
    if isinstance(allowed, list):
        return [str(action) for action in allowed]
    spec = task.get("task_spec_json")
    if isinstance(spec, dict):
        nested = spec.get("allowed_actions")
        if isinstance(nested, list):
            return [str(action) for action in nested]
    return []


def _outputs_from_result_payload(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    outputs: set[str] = set()
    if "per_core_percent" in value or "core_count" in value:
        outputs.add("cpu_usage")
    if "celsius" in value:
        outputs.add("cpu_temperature")
    if {"total_bytes", "available_bytes", "used_percent"} <= set(value):
        outputs.add("memory_usage")
    if {"total_bytes", "free_bytes", "used_percent"} <= set(value):
        outputs.add("disk_usage")
    if "uptime_seconds" in value:
        outputs.add("uptime")
    if "active" in value and "service" in value:
        outputs.add("service_status")
    if "text" in value and "lines" in value:
        outputs.add("logs")
    if "items" in value and "path" in value:
        outputs.add("artifacts")
    if "traces" in value:
        outputs.add("traces")
    if "summary" in value and "observations" in value:
        outputs.add("system_diagnosis")
        observations = value.get("observations")
        if isinstance(observations, dict):
            for child in observations.values():
                outputs.update(_outputs_from_result_payload(child))
    if "findings" in value or "file_count" in value:
        outputs.add("code_structure")
    if "queue" in value or "stale_jobs" in value or "recent_work" in value:
        outputs.add("work_pipeline")
    return outputs
