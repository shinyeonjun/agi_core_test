from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agent.core.decisions import list_decisions
from agent.core.task_queue import list_tasks
from agent.tools.action_log import list_action_runs


ACTION_CATEGORY_LABELS = {
    "success": "성공",
    "policy_block": "정책 차단",
    "approval_required": "승인 필요",
    "profile_block": "모드 제한",
    "timeout": "시간 초과",
    "command_not_found": "명령 없음",
    "command_failed": "명령 실패",
    "running": "진행 중",
    "unknown": "미분류",
}

ACTION_CATEGORY_SEVERITY = {
    "success": "info",
    "policy_block": "warning",
    "approval_required": "warning",
    "profile_block": "info",
    "timeout": "warning",
    "command_not_found": "warning",
    "command_failed": "warning",
    "running": "info",
    "unknown": "info",
}

SUMMARY_TO_CATEGORY = {
    "rc=0": "success",
    "timeout": "timeout",
    "command_not_found": "command_not_found",
    "profile_not_full_device_lab": "profile_block",
    "approval_required": "approval_required",
    "ssh_key_access_denied": "policy_block",
    "root_delete_denied": "policy_block",
    "env_access_denied": "policy_block",
    "secret_access_denied": "policy_block",
    "credential_exfiltration_denied": "policy_block",
    "remote_script_execution_denied": "policy_block",
    "external_harm_denied": "policy_block",
}

SUMMARY_LABELS = {
    "rc=0": "정상 종료",
    "timeout": "제한 시간 초과",
    "command_not_found": "명령을 찾지 못함",
    "profile_not_full_device_lab": "현재 모드에서 로컬 실행 차단",
    "approval_required": "사람 승인 필요",
    "ssh_key_access_denied": "SSH 키 접근 차단",
    "root_delete_denied": "위험한 삭제 차단",
    "env_access_denied": ".env 접근 차단",
    "secret_access_denied": "민감정보 접근 차단",
    "credential_exfiltration_denied": "민감정보 외부 전송 차단",
    "remote_script_execution_denied": "원격 스크립트 실행 차단",
    "external_harm_denied": "외부 피해 가능 작업 차단",
    "no_approved_proposal": "실행 가능한 제안 없음",
    "approval_rejected": "사람이 승인 거절",
    "stale_running_recovered": "멈춘 작업 복구",
    "stale_running_max_attempts": "재시도 한도 초과",
    "max_attempts_exceeded": "재시도 한도 초과",
}


def decode_json_value(value: object, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def decode_command(value: object) -> list[str]:
    decoded = decode_json_value(value, fallback=None)
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    if decoded is not None:
        return [str(decoded)]
    if value is None:
        return []
    return [str(value)]


def command_text(command_json: object) -> str:
    return " ".join(part for part in decode_command(command_json) if part and part != "-").strip()


def summary_label(summary: object) -> str:
    raw = str(summary or "").strip()
    if not raw:
        return "없음"
    return SUMMARY_LABELS.get(raw, raw.replace("_", " "))


def _category_for_action(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "")
    summary = str(row.get("result_summary") or "")
    returncode = row.get("returncode")
    if status == "running":
        return "running"
    if status == "completed" and (returncode == 0 or summary == "rc=0"):
        return "success"
    if status == "timeout" or summary == "timeout":
        return "timeout"
    if summary in SUMMARY_TO_CATEGORY:
        return SUMMARY_TO_CATEGORY[summary]
    if returncode == 127:
        return "command_not_found"
    if status == "blocked":
        return "policy_block"
    if status == "failed" or (isinstance(returncode, int) and returncode != 0):
        return "command_failed"
    return "unknown"


def action_observation(row: dict[str, Any]) -> dict[str, Any]:
    category = _category_for_action(row)
    command = command_text(row.get("command_json"))
    summary = str(row.get("result_summary") or "")
    return {
        "id": row.get("id"),
        "status": row.get("status"),
        "category": category,
        "label": ACTION_CATEGORY_LABELS[category],
        "severity": ACTION_CATEGORY_SEVERITY[category],
        "summary": summary,
        "summary_label": summary_label(summary),
        "returncode": row.get("returncode"),
        "command": command,
        "profile": row.get("profile"),
        "risk_level": row.get("risk_level"),
        "explanation": _action_explanation(category, summary, command),
        "next_step": _action_next_step(category, summary),
    }


def _action_explanation(category: str, summary: str, command: str) -> str:
    if category == "success":
        return "명령이 정상 종료됐어."
    if category == "profile_block":
        return "현재 자율 모드가 로컬 실행을 허용하지 않아서 막았어."
    if category == "approval_required":
        return "시스템 변경 가능성이 있어서 사람 승인이 필요해."
    if category == "policy_block":
        return f"{summary_label(summary)} 규칙에 걸려 실행하지 않았어."
    if category == "timeout":
        return "제한 시간 안에 끝나지 않아서 중단했어."
    if category == "command_not_found":
        return f"`{command}` 실행 파일이나 명령을 찾지 못했어."
    if category == "command_failed":
        return "명령은 실행됐지만 0이 아닌 종료 코드로 끝났어."
    if category == "running":
        return "아직 실행 중으로 기록돼 있어."
    return "아직 분류 규칙이 없는 결과야."


def _action_next_step(category: str, summary: str) -> str:
    if category == "success":
        return "후속 조치 없음"
    if category == "profile_block":
        return "필요하면 full_device_lab 모드와 승인 조건을 확인"
    if category == "approval_required":
        return "#승인 채널에서 승인 또는 거절"
    if category == "policy_block":
        return "요청 의도와 정책 규칙을 다시 확인"
    if category == "timeout":
        return "timeout 증가, 명령 축소, 재시도 여부 판단"
    if category == "command_not_found":
        return "패키지 설치 여부나 명령 경로 확인"
    if category == "command_failed":
        return "stderr와 returncode 기준으로 원인 분류"
    if category == "running":
        return "tasks doctor 또는 action 상태 확인"
    return "분류 규칙 추가 검토"


def action_failure_breakdown(actions: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(action_observation(row)["category"] for row in actions)
    counter.pop("success", None)
    return dict(counter)


def task_observation(task: dict[str, Any]) -> dict[str, Any]:
    status = str(task.get("status") or "")
    queue_type = str(task.get("queue_type") or "")
    result = task.get("result") if isinstance(task.get("result"), dict) else decode_json_value(task.get("result_json"), fallback={})
    reason = str((result or {}).get("reason") or "")
    return {
        "id": task.get("id"),
        "queue_type": queue_type,
        "status": status,
        "title": task.get("title"),
        "attempts": int(task.get("attempts") or 0),
        "approval_id": task.get("approval_id"),
        "waiting_reason": _task_waiting_reason(status, queue_type, task.get("approval_id"), reason),
        "next_step": _task_next_step(status, queue_type, task.get("approval_id"), reason),
    }


def _task_waiting_reason(status: str, queue_type: str, approval_id: object, reason: str) -> str:
    if status == "queued" and queue_type == "user":
        return "사용자 작업 큐에서 즉시 처리 worker를 기다리는 중"
    if status == "queued" and queue_type == "autonomous":
        return "자율 스케줄러 실행 주기를 기다리는 중"
    if status == "waiting_approval":
        return f"승인 #{approval_id} 대기 중" if approval_id else "사람 승인 대기 중"
    if status == "running":
        return "worker가 실행 중으로 claim한 상태"
    if status == "blocked":
        return summary_label(reason) if reason else "차단됨"
    if status == "done":
        return "완료됨"
    if status == "skipped":
        return summary_label(reason) if reason else "이번 주기에서 건너뜀"
    return "상태 설명 규칙 없음"


def _task_next_step(status: str, queue_type: str, approval_id: object, reason: str) -> str:
    if status == "queued" and queue_type == "user":
        return "대화 트리거나 agentctl tasks run-user로 처리"
    if status == "queued" and queue_type == "autonomous":
        return "lab tick 주기에서 처리"
    if status == "waiting_approval":
        return f"!approve {approval_id} 또는 !reject {approval_id}" if approval_id else "승인 항목 확인"
    if status == "running":
        return "오래 멈춰 있으면 agentctl tasks doctor 실행"
    if status == "blocked":
        return "reason을 보고 재시도/폐기/승인 요청 판단"
    if status == "done":
        return "후속 조치 없음"
    if status == "skipped":
        return "조건 충족 후 재큐잉 여부 확인"
    return "상태 분류 추가"


def decision_trace(decision: dict[str, Any]) -> dict[str, Any]:
    policy = decision.get("policy_summary") or {}
    interpretation = decision.get("language_interpretation") or {}
    user_goal = decision.get("user_directed_goal") or {}
    self_map = decision.get("runtime_self_map") or {}
    return {
        "facts": [
            f"renderer={decision.get('renderer')}",
            f"intent={interpretation.get('intent') or 'unknown'}",
            f"target={interpretation.get('target') or 'unknown'}",
            f"risk={policy.get('risk_level') or decision.get('risk_level')}",
            f"requires_approval={bool(policy.get('requires_approval'))}",
            f"self_map={self_map.get('summary') or 'unavailable'}",
        ],
        "inferences": [
            "사용자 목표로 분류됨" if user_goal else "대화 응답으로 분류됨",
            f"selected_goal_id={decision.get('selected_goal_id')}",
        ],
        "guards": [
            "민감정보 원문은 renderer/summary에 전달하지 않음",
            "정책상 거부/승인 필요 작업은 실행하지 않음",
        ],
        "next_step": _decision_next_step(policy, user_goal),
    }


def _decision_next_step(policy: dict[str, Any], user_goal: dict[str, Any]) -> str:
    if policy.get("denied"):
        return "정책 차단 사유를 사용자에게 설명"
    if user_goal and user_goal.get("status") == "waiting_approval":
        return "승인 채널에서 사람 판단 대기"
    if user_goal and user_goal.get("status") == "active":
        return "사용자 작업 큐에서 우선 처리"
    return "대화 응답 후 상태/기억만 갱신"


def latest_decision_traces(limit: int = 5) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for row in list_decisions(limit):
        decision = decode_json_value(row.get("decision_json"), fallback={})
        if isinstance(decision, dict):
            traces.append({"id": row.get("id"), "trace": decision.get("decision_trace") or decision_trace(decision)})
    return traces


def observability_snapshot(limit: int = 10) -> dict[str, Any]:
    actions = list_action_runs(limit)
    tasks = list_tasks(limit)
    return {
        "actions": [action_observation(row) for row in actions],
        "action_failure_breakdown": action_failure_breakdown(actions),
        "tasks": [task_observation(row) for row in tasks],
        "decision_traces": latest_decision_traces(min(5, limit)),
    }
