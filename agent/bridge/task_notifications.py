from __future__ import annotations

import json
import os
from typing import Any

from agent.bridge.formatter import compact_text, redact_discord_content
from agent.bridge.notifier import post_bot_channel, post_webhook, webhook_url
from agent.core.database import connect, init_db
from agent.core.events import log_event


PHASE_LABELS = {
    "queued": "대기 중",
    "planning": "준비 중",
    "executing": "작업 중",
    "verifying": "검증 중",
    "reporting": "보고 중",
    "learned": "기록 완료",
}

STATUS_LABELS = {
    "queued": "대기 중",
    "running": "진행 중",
    "started": "시작",
    "passed": "통과",
    "done": "완료",
    "blocked": "막힘",
    "failed": "실패",
    "skipped": "건너뜀",
    "waiting_approval": "승인 대기",
    "codex_work_completed": "코드 작업 완료",
    "codex_work_failed": "코드 작업 실패",
    "codex_work_blocked": "코드 작업 차단",
}


def _enabled() -> bool:
    return os.getenv("AGENT_DISCORD_TASK_NOTIFICATIONS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _decode_json(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(str(value or "")) if value else {}
    except json.JSONDecodeError:
        return {}


def _task_brief(task_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM task_queue WHERE id = ?", (int(task_id),)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["payload"] = _decode_json(data.get("payload_json"))
    data["result"] = _decode_json(data.get("result_json"))
    return data


def _is_user_visible(task: dict[str, Any] | None, queue_type: str | None) -> bool:
    if queue_type == "user":
        return True
    if not task:
        return False
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    source = str(task.get("source") or "")
    return bool(payload.get("requires_native_loop")) or source.startswith(("discord_self_improvement", "self_improvement"))


def _task_kind_label(task: dict[str, Any] | None) -> str:
    kind = compact_text((task or {}).get("task_kind"), "")
    source = compact_text((task or {}).get("source"), "")
    if source.startswith("discord_self_improvement") or kind == "code_change" and (task or {}).get("payload", {}).get("requires_native_loop"):
        return "자가개선"
    if kind == "code_change":
        return "코드 작업"
    if kind == "report":
        return "보고서"
    if kind == "project_spec":
        return "프로젝트 설계"
    return "작업"


def _human_title(task: dict[str, Any] | None) -> str:
    title = compact_text((task or {}).get("title"), "작업")
    return title.replace("사용자 요청 자가개선: ", "").replace("Core self-improvement", "Core 자가개선")


def _phase_message(task: dict[str, Any] | None, *, task_id: int, phase: str, status: str, summary: str) -> str:
    phase_label = PHASE_LABELS.get(phase, phase)
    status_label = STATUS_LABELS.get(status, status.replace("_", " "))
    kind = _task_kind_label(task)
    title = _human_title(task)
    lines = [
        f"**{kind} {phase_label} #{task_id}**",
        f"{title}",
        f"상태: {status_label}",
    ]
    clean_summary = compact_text(summary, "")
    if clean_summary:
        lines.append(f"지금: {clean_summary}")
    if phase == "queued":
        lines.append("다음: 작업자가 별도 작업공간을 잡고 시작할 거야.")
    elif phase == "planning":
        lines.append("다음: 수정 범위와 실행 방식을 정리해.")
    elif phase == "executing":
        lines.append("다음: 코드 수정 뒤 검증으로 넘어가.")
    elif phase == "verifying":
        lines.append("다음: 테스트/audit/eval 결과를 확인해.")
    return "\n".join(lines)


def notify_task_phase(task_id: int, phase: str, status: str, summary: str, *, queue_type: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _enabled() or phase == "learned" or not webhook_url("update"):
        return {"sent": False, "reason": "disabled_or_missing_webhook"}
    task = _task_brief(task_id)
    if not _is_user_visible(task, queue_type):
        return {"sent": False, "reason": "not_user_visible"}
    content = _phase_message(task, task_id=task_id, phase=phase, status=status, summary=summary)
    result = post_webhook("update", content)
    log_event("discord", "task_phase_update_notify", str(task_id), {"task_id": task_id, "phase": phase, "status": status, "sent": result.get("sent")}, 0.5)
    return result


def _finish_message(task: dict[str, Any] | None, task_id: int, status: str, result: dict[str, Any]) -> str:
    kind = _task_kind_label(task)
    title = _human_title(task)
    result_status = compact_text(result.get("status") or status)
    approval_id = result.get("approval_id")
    lines = [
        f"**{kind} 결과 #{task_id}**",
        title,
        f"결과: {STATUS_LABELS.get(status, status)}",
        f"상세: {STATUS_LABELS.get(result_status, result_status.replace('_', ' '))}",
    ]
    if approval_id:
        lines.append(f"다음: 승인 채널에서 #{approval_id} 확인이 필요해.")
    elif status == "done":
        lines.append("다음: 결과를 기록했고 필요하면 다음 작업으로 이어갈 수 있어.")
    elif status == "blocked":
        lines.append(f"막힌 이유: {compact_text(result.get('reason'), '확인 필요')}")
    return "\n".join(lines)


def notify_task_finished(task_id: int, status: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _enabled():
        return {"sent": False, "reason": "disabled"}
    payload = result or {}
    task = _task_brief(task_id)
    if not _is_user_visible(task, payload.get("queue_type") or (task or {}).get("queue_type")):
        return {"sent": False, "reason": "not_user_visible"}
    content = _finish_message(task, task_id, status, payload)
    update_result = post_webhook("update", content) if webhook_url("update") else {"sent": False, "reason": "missing_update_webhook"}
    summary_result = {"sent": False, "reason": "not_summary_worthy"}
    report = compact_text(payload.get("report"), "")
    if status == "done" and report and webhook_url("summary"):
        summary_content = "\n".join([f"**작업 결과 요약 #{task_id}**", _human_title(task), "", redact_discord_content(report[:1600])])
        summary_result = post_webhook("summary", summary_content)
    log_event("discord", "task_finished_notify", str(task_id), {"task_id": task_id, "status": status, "update": update_result, "summary": summary_result}, 0.56)
    return {"update": update_result, "summary": summary_result}


def notify_approval_required(approval_id: int, proposal: dict[str, Any]) -> dict[str, Any]:
    if not _enabled():
        return {"sent": False, "reason": "disabled"}
    description = compact_text(proposal.get("description"), "승인이 필요한 작업")
    risk = compact_text(proposal.get("risk_level"), "medium")
    content = "\n".join(
        [
            f"**승인 필요 #{approval_id}**",
            f"작업: {description}",
            f"위험도: {risk}",
            "상태: 아직 실행/반영 안 함",
            f"명령: `!approve {approval_id}` 또는 `!reject {approval_id}`",
        ]
    )
    approval_result = post_bot_channel("approval", content)
    update_result = post_webhook("update", content) if webhook_url("update") else {"sent": False, "reason": "missing_update_webhook"}
    log_event("discord", "approval_required_notify", str(approval_id), {"approval_id": approval_id, "approval": approval_result, "update": update_result}, 0.62)
    return {"approval": approval_result, "update": update_result}
