from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory import HarnessMemory


ACTIVE_SELF_IMPROVEMENT_STATUSES = {"accepted", "planned", "running", "reviewing", "waiting_approval", "blocked"}


def summarize_self_improvement_slate(db_path: str | Path, *, max_size: int) -> dict[str, Any]:
    with HarnessMemory(db_path) as memory:
        rows = [compact_work_item(row) for row in memory.list_work_items(limit=100) if is_self_improvement_work(row)]
    proposed = [row for row in rows if str(row.get("status") or "") == "proposed"]
    active = [row for row in rows if str(row.get("status") or "") in ACTIVE_SELF_IMPROVEMENT_STATUSES]
    return {
        "schema_version": "neurokernel-self-improvement-slate-v1",
        "max_open_proposals": max(1, int(max_size)),
        "proposed_count": len(proposed),
        "active_count": len(active),
        "proposed_work": proposed,
        "active_work": active,
        "state": "active_work_in_progress" if active else ("waiting_for_user_decision" if proposed else "empty"),
    }


def defer_sibling_self_improvement_proposals(memory: HarnessMemory, accepted_item: dict[str, Any], *, actor: str) -> list[dict[str, Any]]:
    if not is_self_improvement_work(accepted_item):
        return []
    accepted_work_id = str(accepted_item.get("work_id") or "")
    deferred: list[dict[str, Any]] = []
    for item in memory.list_work_items(limit=100):
        if str(item.get("work_id") or "") == accepted_work_id:
            continue
        if str(item.get("status") or "") != "proposed":
            continue
        if not is_self_improvement_work(item):
            continue
        updated = memory.transition_work_item(
            str(item["work_id"]),
            "deferred",
            actor=actor,
            payload={"reason": "self_improvement_slate_reset", "accepted_work_id": accepted_work_id},
        )
        deferred.append(compact_work_item(updated))
    if deferred:
        memory.add_work_event(accepted_work_id, "proposal_slate_reset", actor=actor, payload={"deferred_work_ids": [item["work_id"] for item in deferred]})
        memory.conn.commit()
    return deferred


def is_self_improvement_work(row: dict[str, Any]) -> bool:
    if str(row.get("linked_entity_type") or "") == "self_improvement_deficit":
        return True
    metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
    return str(metadata.get("execution_kind") or "") == "self_improvement"


def compact_work_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "work_id": row.get("work_id"),
        "type": row.get("type"),
        "title": row.get("title"),
        "status": row.get("status"),
        "priority": row.get("priority"),
        "risk_level": row.get("risk_level"),
        "linked_entity_id": row.get("linked_entity_id"),
        "updated_at": row.get("updated_at"),
    }
