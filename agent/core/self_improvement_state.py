from __future__ import annotations

from typing import Any

from agent.core.approvals import ApprovalStore
from agent.core.task_queue import list_tasks


def _is_self_improvement_task(task: dict[str, Any]) -> bool:
    payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}
    source = str(task.get("source") or "")
    return (
        task.get("task_kind") == "code_change"
        and (
            source.startswith(("discord_self_improvement", "self_improvement"))
            or bool(payload.get("requires_native_loop"))
        )
    )


def classify_self_improvement_phase(task: dict[str, Any]) -> dict[str, Any]:
    status = str(task.get("status") or "")
    result = task.get("result") if isinstance(task.get("result"), dict) else {}
    review = result.get("code_review") if isinstance(result.get("code_review"), dict) else {}
    verdict = str(review.get("verdict") or "")
    approval_id = result.get("approval_id") or task.get("approval_id")

    if status == "queued":
        phase = "queued"
        next_step = "작업자가 별도 작업공간에서 수정을 시작해야 해."
    elif status == "running":
        phase = "executing"
        next_step = "작업자가 수정, 테스트, 검증 결과를 만드는 중이야."
    elif status == "waiting_approval" or approval_id or verdict in {"needs_approval", "ready_for_approval"}:
        phase = "approval"
        next_step = f"승인 채널에서 #{approval_id} 승인 또는 거절이 필요해." if approval_id else "승인 항목 확인이 필요해."
    elif status == "done" and verdict in {"needs_validation", "needs_review"}:
        phase = "review"
        next_step = str(review.get("next_action") or "검증이나 리뷰 보강이 필요해.")
    elif status == "done":
        phase = "done"
        next_step = "완료 기록을 확인하면 돼."
    elif status == "blocked":
        phase = "blocked"
        recovery = result.get("recovery_plan") if isinstance(result.get("recovery_plan"), dict) else {}
        next_step = str(recovery.get("next_action") or result.get("reason") or (review.get("next_action") if review else "") or "막힌 이유를 확인해야 해.")
    else:
        phase = "unknown"
        next_step = "상태 확인이 필요해."

    return {
        "task_id": task.get("id"),
        "goal_id": task.get("goal_id"),
        "title": task.get("title"),
        "queue_type": task.get("queue_type"),
        "status": status,
        "phase": phase,
        "review_verdict": verdict or None,
        "approval_id": approval_id,
        "next_step": next_step,
    }


def self_improvement_status(limit: int = 10) -> dict[str, Any]:
    safe_limit = max(1, int(limit))
    tasks = [task for task in list_tasks(limit=max(20, safe_limit * 3)) if _is_self_improvement_task(task)]
    items = [classify_self_improvement_phase(task) for task in tasks[:safe_limit]]
    approvals = [
        row
        for row in ApprovalStore().list_pending()
        if (row.get("proposal") or {}).get("action_type") == "self_improvement_apply"
    ]
    return {
        "items": items,
        "pending_apply_approvals": [
            {
                "id": row.get("id"),
                "description": row.get("description"),
                "risk_level": row.get("risk_level"),
            }
            for row in approvals
        ],
        "counts": {
            "queued": sum(1 for item in items if item["phase"] == "queued"),
            "executing": sum(1 for item in items if item["phase"] == "executing"),
            "review": sum(1 for item in items if item["phase"] == "review"),
            "approval": sum(1 for item in items if item["phase"] == "approval"),
            "blocked": sum(1 for item in items if item["phase"] == "blocked"),
            "done": sum(1 for item in items if item["phase"] == "done"),
        },
    }
