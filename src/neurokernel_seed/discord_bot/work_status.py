from __future__ import annotations

import json
from typing import Any


def format_work_notification(payload: dict[str, Any]) -> tuple[str, str | None]:
    item = payload.get("work_item") if isinstance(payload.get("work_item"), dict) else {}
    children = payload.get("child_work_items") if isinstance(payload.get("child_work_items"), list) else []
    work_id = str(item.get("work_id") or "")
    title = str(item.get("title") or work_id or "작업")
    status = str(item.get("status") or "unknown")
    result = latest_self_patch_result(payload.get("events"))
    result_status = str(result.get("status") or "")
    changed_files = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
    changed_text = ", ".join(str(path) for path in changed_files[:5])
    proposal = payload.get("capability_proposal") if isinstance(payload.get("capability_proposal"), dict) else {}

    if status == "proposed" and str(item.get("type") or "") == "self_patch":
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        view_kind = f"proposal:{proposal_id}" if proposal_id else "work"
        return (
            f"새 자기개선 후보가 올라왔어: {title}\n승인하면 개발 worker가 구현/테스트를 시작하고, 장착은 다시 승인받아.",
            view_kind,
        )
    if status == "proposed" and str(item.get("type") or "") == "training_pipeline":
        metadata = item.get("metadata_json") if isinstance(item.get("metadata_json"), dict) else {}
        slot = str(metadata.get("model_slot") or "model")
        return (
            f"모델 학습 후보가 올라왔어: {title}\n슬롯: `{slot}`\n승인하면 학습 큐로 들어가고, 노트북 training worker가 NK 벤치/배포 파이프라인을 실행해.",
            "work",
        )
    if status == "proposed" and str(item.get("type") or "") == "mcp_plugin_skill":
        return (
            f"MCP/플러그인/스킬 작업 후보가 올라왔어: {title}\n승인하면 격리된 개발 worker가 구현/테스트 패치를 만든 뒤 다시 승인받아.",
            "work",
        )
    if status == "proposed":
        return f"작업 후보가 올라왔어: {title}\n승인하면 큐에 들어가고 worker가 다음 단계를 진행해.", "work"
    child_summary = external_work_child_summary(children)
    if status == "planned" and str(item.get("type") or "") == "external_work" and child_summary:
        return child_summary, None
    if status == "planned" and str(item.get("type") or "") == "external_work":
        return f"계획이 접수됐어: {title}\n실제 코드 구현으로 넘기려면 개발 작업으로 전환해야 해.", "promote"
    if status == "waiting_approval" and result_status == "patch_ready":
        lines = [
            f"개발 후보가 테스트를 통과했어: {title}",
            "이제 장착 승인만 남았어.",
        ]
        if changed_text:
            lines.append(f"바뀐 파일: {changed_text}")
        return "\n".join(lines), "activation"
    if status == "reviewing" and result_status in {"test_failed", "diff_check_failed", "codex_failed", "codex_failed_no_patch"}:
        reason_by_status = {
            "test_failed": "테스트 실패",
            "diff_check_failed": "패치 형식 검사 실패",
            "codex_failed": "개발 워커 실행 실패",
            "codex_failed_no_patch": "개발 워커가 패치 없이 종료 실패",
        }
        reason = reason_by_status.get(result_status, "수정 필요")
        lines = [
            f"개발 시도는 끝났는데 바로 장착하면 안 돼: {title}",
            f"이유: {reason}",
        ]
        if changed_text:
            lines.append(f"건드린 파일: {changed_text}")
        lines.append("패치는 보존했고, 다음엔 실패 로그를 보고 수정해야 해.")
        return "\n".join(lines), "retry"
    if status == "blocked":
        return f"작업이 막혔어: {title}\n패치가 없거나 워커가 더 진행할 수 없는 상태야.", "retry"
    if status == "failed":
        return f"작업이 실패했어: {title}\n상태를 확인해서 원인부터 봐야 해.", "retry"
    if status == "completed":
        return f"작업이 완료됐어: {title}", None
    return f"작업 상태가 바뀌었어: {title}\n현재 상태: {status}", None


def latest_self_patch_result(events: Any) -> dict[str, Any]:
    if not isinstance(events, list):
        return {}
    for event in reversed(events):
        if not isinstance(event, dict) or event.get("event_type") not in {"job_completed", "self_patch_failed"}:
            continue
        payload = event.get("payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                continue
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if isinstance(result, dict):
            return result
    return {}


def build_work_status_payload(items: list[Any], jobs: list[Any]) -> dict[str, Any]:
    clean_items = [item for item in items[:10] if isinstance(item, dict)]
    clean_jobs = [job for job in jobs[:10] if isinstance(job, dict)]
    jobs_by_work: dict[str, list[dict[str, Any]]] = {}
    for job in clean_jobs:
        work_id = str(job.get("work_id") or "")
        if work_id:
            jobs_by_work.setdefault(work_id, []).append(job)
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for item in clean_items:
        parent_id = str(item.get("parent_work_id") or "").strip()
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(item)
    return {
        "kind": "work_status",
        "lifecycle": {
            "proposed": "waiting_for_user_decision",
            "accepted": "queued_or_ready_for_worker",
            "planned": "recorded_plan_without_active_worker",
            "running": "worker_is_processing",
            "reviewing": "worker_finished_but_needs_fix",
            "waiting_approval": "patch_or_result_ready_for_user_approval",
            "completed": "finished",
            "blocked": "cannot_continue_without_review",
            "failed": "failed",
            "cancelled": "cancelled",
        },
        "work_items": clean_items,
        "jobs": clean_jobs[:5],
        "progress": [
            work_progress(
                item,
                jobs_by_work.get(str(item.get("work_id") or ""), []),
                child_items=children_by_parent.get(str(item.get("work_id") or ""), []),
            )
            for item in clean_items
        ],
    }


def work_progress(item: dict[str, Any], jobs: list[dict[str, Any]], *, child_items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    child_items = child_items or []
    work_type = str(item.get("type") or "")
    status = str(item.get("status") or "")
    latest_job = jobs[0] if jobs else {}
    latest_job_status = str(latest_job.get("status") or "") if latest_job else None
    active_child = first_live_self_patch_child(child_items)
    child_status = str(active_child.get("status") or "") if active_child else None
    user_action_required = status in {"proposed", "reviewing", "waiting_approval", "blocked", "failed"} or child_status in {"waiting_approval", "reviewing", "blocked", "failed"}
    if work_type == "self_patch":
        stage = self_patch_stage(status, latest_job_status)
    elif work_type == "mcp_plugin_skill":
        stage = self_patch_stage(status, latest_job_status)
    elif work_type == "training_pipeline":
        stage = training_pipeline_stage(status, latest_job_status)
    elif work_type == "external_work":
        stage = external_work_stage(status, latest_job_status, child_status=child_status)
    else:
        stage = generic_work_stage(status, latest_job_status)
    promotion_possible = status == "planned" and work_type == "external_work" and active_child is None
    return {
        "work_id": item.get("work_id"),
        "title": item.get("title"),
        "type": work_type,
        "status": status,
        "priority": item.get("priority"),
        "risk_level": item.get("risk_level"),
        "latest_job_id": latest_job.get("job_id") if latest_job else None,
        "latest_job_status": latest_job_status,
        "child_work_id": active_child.get("work_id") if active_child else None,
        "child_status": child_status,
        "automation_stage": stage,
        "user_action_required": user_action_required,
        "worker_action_required": status in {"accepted", "running"} or (status == "planned" and work_type == "external_work" and active_child is None),
        "activation_possible": status == "waiting_approval" or child_status == "waiting_approval",
        "promotion_possible": promotion_possible,
        "retry_possible": status in {"reviewing", "blocked", "failed"} and work_type in {"self_patch", "mcp_plugin_skill", "training_pipeline"},
    }


def first_promotable_work_id(status_payload: dict[str, Any]) -> str | None:
    progress = status_payload.get("progress") if isinstance(status_payload.get("progress"), list) else []
    for item in progress:
        if isinstance(item, dict) and item.get("promotion_possible"):
            work_id = str(item.get("work_id") or "").strip()
            if work_id:
                return work_id
    return None


def self_patch_stage(status: str, latest_job_status: str | None) -> str:
    if status == "proposed":
        return "waiting_for_user_to_accept_development"
    if status == "accepted":
        return "accepted_and_waiting_for_development_worker"
    if status == "running" or latest_job_status == "running":
        return "development_worker_running"
    if status == "waiting_approval":
        return "patch_ready_waiting_for_activation_approval"
    if status == "reviewing":
        return "development_attempt_finished_needs_fix"
    if status in {"blocked", "failed"}:
        return "development_blocked_or_failed"
    if status == "completed":
        return "capability_attached_or_work_completed"
    return "not_active"


def external_work_stage(status: str, latest_job_status: str | None, *, child_status: str | None = None) -> str:
    if child_status == "waiting_approval":
        return "implementation_patch_ready_waiting_for_activation"
    if child_status == "completed":
        return "implementation_child_completed"
    if child_status == "running":
        return "implementation_worker_running"
    if child_status in {"reviewing", "blocked", "failed"}:
        return "implementation_child_needs_review"
    if status == "proposed":
        return "waiting_for_user_to_accept_work"
    if status == "accepted":
        return "accepted_and_waiting_for_planning_worker"
    if status == "running" or latest_job_status == "running":
        return "planning_worker_running"
    if status == "planned":
        return "plan_recorded_no_implementation_worker_running"
    if status == "completed":
        return "work_completed"
    if status in {"blocked", "failed"}:
        return "work_blocked_or_failed"
    return "not_active"


def training_pipeline_stage(status: str, latest_job_status: str | None) -> str:
    if status == "proposed":
        return "waiting_for_user_to_accept_training"
    if status == "accepted":
        return "accepted_and_waiting_for_training_worker"
    if status == "running" or latest_job_status == "running":
        return "training_worker_running"
    if status == "reviewing":
        return "training_attempt_finished_needs_review"
    if status == "blocked":
        return "waiting_for_laptop_training_worker_or_manual_review"
    if status == "completed":
        return "training_pipeline_completed"
    if status == "failed":
        return "training_pipeline_failed"
    return "not_active"


def first_live_self_patch_child(children: list[dict[str, Any]]) -> dict[str, Any] | None:
    terminal = {"rejected", "cancelled", "failed", "archived"}
    for child in children:
        if str(child.get("type") or "") == "self_patch" and str(child.get("status") or "") not in terminal:
            return child
    return None


def external_work_child_summary(children: list[Any]) -> str | None:
    live_children = [child for child in children if isinstance(child, dict)]
    child = first_live_self_patch_child(live_children)
    if not child:
        return None
    title = str(child.get("title") or "작업")
    status = str(child.get("status") or "")
    if status == "waiting_approval":
        return f"개발 후보가 준비됐어: {title}\n이제 새 개발 전환이 아니라 장착 승인 단계야."
    if status == "completed":
        return f"작업이 이미 장착됐어: {title}\n새 개발 전환은 필요 없어."
    if status == "running":
        return f"개발 작업이 이미 진행 중이야: {title}"
    if status in {"reviewing", "blocked", "failed"}:
        return f"개발 작업이 수정 대기 중이야: {title}\n전환을 다시 누르는 게 아니라 수정/재시도 단계야."
    return f"개발 작업이 이미 만들어져 있어: {title}\n새 개발 전환은 필요 없어."


def generic_work_stage(status: str, latest_job_status: str | None) -> str:
    if latest_job_status == "running":
        return "worker_running"
    if status in {"accepted", "running"}:
        return "worker_pending_or_running"
    if status == "waiting_approval":
        return "waiting_for_user_approval"
    if status == "completed":
        return "completed"
    if status in {"blocked", "failed", "reviewing"}:
        return "needs_review"
    return "not_active"
