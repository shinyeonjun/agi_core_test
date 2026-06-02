from __future__ import annotations

import json
from typing import Any

FAILURE_CATEGORIES = {
    "success",
    "intent_misread",
    "bad_plan",
    "tool_error",
    "tool_unavailable",
    "permission_block",
    "policy_block",
    "profile_block",
    "missing_context",
    "verification_failed",
    "timeout",
    "environment_issue",
    "input_insufficient",
    "unknown",
}

FAILURE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("policy_block", ("policy", "denied", "blocked", "secret", "token", ".env", "ssh_key", "root_delete")),
    ("permission_block", ("permission denied", "access denied", "requires_approval", "approval_required", "sudo")),
    ("profile_block", ("profile_not_full_device_lab", "safe profile", "worker_not_available")),
    ("timeout", ("timeout", "timed out", "returncode: 124", "rc=124")),
    ("tool_unavailable", ("command_not_found", "returncode: 127", "rc=127", "no such file", "not found")),
    ("verification_failed", ("pytest", "assertionerror", "test failed", "verification_failed")),
    ("environment_issue", ("modulenotfounderror", "venv", "pythonpath", "environment", "no module named")),
    ("missing_context", ("missing_context", "not enough context", "context unavailable")),
    ("input_insufficient", ("too broad", "unclear", "insufficient", "need more detail")),
    ("bad_plan", ("bad_plan", "plan_invalid", "wrong step", "invalid project step")),
    ("tool_error", ("returncode", "stderr", "exception", "traceback")),
)

RECOVERY_HINTS = {
    "success": "추가 복구 조치 없음",
    "intent_misread": "사용자 의도를 다시 해석하고 목표, 대상, 완료 조건을 분리한다.",
    "bad_plan": "계획 단계를 더 작게 쪼개고 검증 조건을 먼저 정의한다.",
    "tool_error": "stderr, returncode, cwd, 입력 인자를 기록한 뒤 가장 작은 재현부터 다시 시도한다.",
    "tool_unavailable": "도구 설치 여부와 실행 경로를 확인하고 대체 도구를 선택한다.",
    "permission_block": "승인 채널로 넘기거나 권한이 덜 필요한 대체 작업으로 바꾼다.",
    "policy_block": "금지된 경로를 제거하고 읽기 전용 또는 제안 작업으로 대체한다.",
    "profile_block": "현재 autonomy profile을 확인하고 필요하면 사용자 승인 뒤 전환한다.",
    "missing_context": "self-map, memory, repo map, 최근 이벤트를 보강한 뒤 다시 판단한다.",
    "verification_failed": "실패 테스트를 최소 재현하고 변경 범위를 줄여 다시 구현한다.",
    "timeout": "작업을 더 작은 단계로 나누거나 timeout과 비동기 처리 방식을 조정한다.",
    "environment_issue": "venv, PYTHONPATH, cwd, OS 차이를 확인하고 환경 복구 action을 만든다.",
    "input_insufficient": "불확실한 완료 조건을 질문 또는 보류 사유로 남긴다.",
    "unknown": "원시 결과를 요약해 기록하고 새 failure rule 후보로 남긴다.",
}


def _text(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False).lower()
    return str(value or "").lower()


def classify_failure(value: object) -> str:
    text = _text(value)
    if any(token in text for token in ("completed", "done", "rc=0", 'returncode": 0', "codex_work_completed")):
        return "success"
    for category, tokens in FAILURE_RULES:
        if any(token in text for token in tokens):
            return category
    return "unknown"


def recovery_hint(category: str) -> str:
    return RECOVERY_HINTS.get(category, RECOVERY_HINTS["unknown"])


def failure_report(value: object) -> dict[str, Any]:
    category = classify_failure(value)
    return {
        "category": category,
        "recovery_hint": recovery_hint(category),
        "known": category in FAILURE_CATEGORIES,
    }
