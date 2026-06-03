from __future__ import annotations

import json
import os
from typing import Any

from agent.bridge.formatter import compact_text, redact_discord_content
from agent.bridge.notifier import post_bot_channel, post_webhook, webhook_url
from agent.core.database import connect, init_db
from agent.core.events import log_event


PHASE_LABELS = {
    "planning": "\uc2dc\uc791",
}

STATUS_LABELS = {
    "queued": "\ub300\uae30 \uc911",
    "running": "\uc9c4\ud589 \uc911",
    "claimed": "\uc2dc\uc791\ud568",
    "started": "\uc2dc\uc791\ud568",
    "passed": "\ud1b5\uacfc",
    "done": "\uc644\ub8cc",
    "blocked": "\ucc28\ub2e8\ub428",
    "failed": "\uc2e4\ud328",
    "skipped": "\uac74\ub108\ub700",
    "waiting_approval": "\uc2b9\uc778 \ub300\uae30",
    "codex_work_completed": "\ucf54\ub4dc \uc791\uc5c5 \uc644\ub8cc",
    "codex_work_failed": "\ucf54\ub4dc \uc791\uc5c5 \uc2e4\ud328",
    "codex_work_blocked": "\ucf54\ub4dc \uc791\uc5c5 \ucc28\ub2e8",
}

REASON_LABELS = {
    "secret_access_denied": "\uc548\uc804 \uaddc\uce59\uc774 \ubbfc\uac10\uc815\ubcf4 \uc811\uadfc\uc73c\ub85c \ud310\ub2e8\ud574 \uba48\ucdc4",
    "self_improvement_requires_native_loop": "\uc790\uac00\uac1c\uc120\uc740 \uaca9\ub9ac\ub41c \uc791\uc5c5\uacf5\uac04\uc774 \ud544\uc694\ud574 \uba48\ucdc4",
    "codex_worker_not_available": "\ucf54\ub4dc \uc791\uc5c5\uc790\ub97c \uc2e4\ud589\ud560 \uc218 \uc5c6\uc5b4 \uba48\ucdc4",
    "profile_not_full_device_lab": "\ud604\uc7ac \ubaa8\ub4dc\uc5d0\uc11c \uc2e4\ud589\uc774 \ud5c8\uc6a9\ub418\uc9c0 \uc54a\uc544 \uba48\ucdc4",
}


def _enabled() -> bool:
    return os.getenv("AGENT_DISCORD_TASK_NOTIFICATIONS", "1").strip().lower() not in {"0", "false", "no", "off"}


def _verbose() -> bool:
    return os.getenv("AGENT_DISCORD_TASK_NOTIFY_DETAIL", "compact").strip().lower() in {"verbose", "all", "1", "true"}


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


def _is_self_improvement(task: dict[str, Any] | None) -> bool:
    payload = (task or {}).get("payload") if isinstance((task or {}).get("payload"), dict) else {}
    source = str((task or {}).get("source") or "")
    return source.startswith(("discord_self_improvement", "self_improvement")) or bool(payload.get("requires_native_loop"))


def _task_kind_label(task: dict[str, Any] | None) -> str:
    kind = compact_text((task or {}).get("task_kind"), "")
    if _is_self_improvement(task):
        return "\uc790\uac00\uac1c\uc120"
    if kind == "code_change":
        return "\ucf54\ub4dc \uc791\uc5c5"
    if kind == "report":
        return "\ubcf4\uace0\uc11c"
    if kind == "project_spec":
        return "\ud504\ub85c\uc81d\ud2b8 \uc124\uacc4"
    return "\uc791\uc5c5"


def _human_title(task: dict[str, Any] | None) -> str:
    title = compact_text((task or {}).get("title"), "\uc791\uc5c5")
    title = title.replace("\uc0ac\uc6a9\uc790 \uc694\uccad \uc790\uac00\uac1c\uc120: ", "")
    title = title.replace("Core self-improvement", "Core \uc790\uac00\uac1c\uc120")
    return redact_discord_content(title)


def _reason_text(result: dict[str, Any]) -> str:
    reason = compact_text(result.get("reason") or result.get("denied_reason") or "")
    if not reason:
        return "\uc0c1\uc138 \uc0ac\uc720 \ud655\uc778 \ud544\uc694"
    return REASON_LABELS.get(reason, reason.replace("_", " "))


def _should_notify_phase(phase: str, status: str) -> bool:
    if _verbose():
        return phase != "learned"
    return phase == "planning" and status == "claimed"


def _phase_message(task: dict[str, Any] | None, *, task_id: int, phase: str, status: str) -> str:
    kind = _task_kind_label(task)
    title = _human_title(task)
    phase_label = PHASE_LABELS.get(phase, "\uc9c4\ud589")
    status_label = STATUS_LABELS.get(status, status.replace("_", " "))
    return "\n".join(
        [
            f"**{kind} {phase_label} #{task_id}**",
            title,
            f"\uc0c1\ud0dc: {status_label}",
            "\uc9c0\uae08 \uaca9\ub9ac\ub41c \uc791\uc5c5\uacf5\uac04\uc5d0\uc11c \ucc98\ub9ac\ud558\ub294 \uc911\uc774\uc57c.",
        ]
    )


def notify_task_phase(task_id: int, phase: str, status: str, summary: str, *, queue_type: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _enabled() or not webhook_url("update"):
        return {"sent": False, "reason": "disabled_or_missing_webhook"}
    if not _should_notify_phase(phase, status):
        return {"sent": False, "reason": "compact_mode_suppressed"}
    task = _task_brief(task_id)
    if not _is_user_visible(task, queue_type):
        return {"sent": False, "reason": "not_user_visible"}
    content = _phase_message(task, task_id=task_id, phase=phase, status=status)
    result = post_webhook("update", content)
    log_event("discord", "task_phase_update_notify", str(task_id), {"task_id": task_id, "phase": phase, "status": status, "sent": result.get("sent")}, 0.5)
    return result


def _finish_message(task: dict[str, Any] | None, task_id: int, status: str, result: dict[str, Any]) -> str:
    kind = _task_kind_label(task)
    title = _human_title(task)
    result_status = compact_text(result.get("status") or status)
    lines = [
        f"**{kind} \uacb0\uacfc #{task_id}**",
        title,
        f"\uacb0\uacfc: {STATUS_LABELS.get(status, status)}",
    ]
    if status == "done":
        lines.append("\uc694\uc57d: \uc791\uc5c5\uc774 \ub05d\ub0ac\uace0 \uacb0\uacfc\ub97c \uae30\ub85d\ud588\uc5b4.")
        approval_id = result.get("approval_id")
        if approval_id:
            lines.append(f"\ub2e4\uc74c: main \ubc18\uc601\uc740 #\uc2b9\uc778\uc5d0\uc11c #{approval_id} \ud655\uc778\uc774 \ud544\uc694\ud574.")
    elif status == "blocked":
        lines.append(f"\uc694\uc57d: {_reason_text(result)}")
    elif status == "waiting_approval":
        lines.append("\uc694\uc57d: \uc2e4\ud589 \uc804\uc5d0 \uc2b9\uc778\uc774 \ud544\uc694\ud574.")
    else:
        lines.append(f"\uc694\uc57d: {STATUS_LABELS.get(result_status, result_status.replace('_', ' '))}")
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
        summary_content = "\n".join([f"**\uc791\uc5c5 \uacb0\uacfc \uc694\uc57d #{task_id}**", _human_title(task), "", redact_discord_content(report[:1600])])
        summary_result = post_webhook("summary", summary_content)
    log_event("discord", "task_finished_notify", str(task_id), {"task_id": task_id, "status": status, "update": update_result, "summary": summary_result}, 0.56)
    return {"update": update_result, "summary": summary_result}


def notify_approval_required(approval_id: int, proposal: dict[str, Any]) -> dict[str, Any]:
    if not _enabled():
        return {"sent": False, "reason": "disabled"}
    description = redact_discord_content(compact_text(proposal.get("description"), "\uc2b9\uc778\uc774 \ud544\uc694\ud55c \uc791\uc5c5"))
    risk = compact_text(proposal.get("risk_level"), "medium")
    content = "\n".join(
        [
            f"**\uc2b9\uc778 \ud544\uc694 #{approval_id}**",
            f"\uc791\uc5c5: {description}",
            f"\uc704\ud5d8\ub3c4: {risk}",
            "\uc0c1\ud0dc: \uc544\uc9c1 \uc2e4\ud589/\ubc18\uc601 \uc548 \ud588\uc5b4.",
            f"\uba85\ub839: `!approve {approval_id}` \ub610\ub294 `!reject {approval_id}`",
        ]
    )
    approval_result = post_bot_channel("approval", content)
    log_event("discord", "approval_required_notify", str(approval_id), {"approval_id": approval_id, "approval": approval_result}, 0.62)
    return {"approval": approval_result, "update": {"sent": False, "reason": "approval_channel_only"}}
