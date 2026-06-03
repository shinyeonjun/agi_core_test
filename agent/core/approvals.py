from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.policy import ActionProposal
from agent.core.task_queue import block_tasks_for_approval, resume_tasks_for_approval
from agent.core.wake_signals import emit_wake_signal


class ApprovalStore:
    def create_approval(self, proposal: ActionProposal) -> int:
        init_db()
        ts = now_kst()
        with connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO approvals (
                    created_at, updated_at, action_type, description,
                    proposed_payload_json, risk_level, status, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL)
                """,
                (
                    ts,
                    ts,
                    proposal.action_type,
                    proposal.description,
                    json.dumps(proposal.to_dict(), ensure_ascii=False),
                    proposal.risk_level,
                ),
            )
            conn.commit()
            approval_id = int(cur.lastrowid)
        try:
            from agent.bridge.task_notifications import notify_approval_required

            notify_approval_required(approval_id, proposal.to_dict())
        except Exception as exc:
            emit_wake_signal(
                "approval_notify_failed",
                "approval_store",
                priority=0.45,
                payload={"approval_id": approval_id, "error": type(exc).__name__},
                dedupe_key=f"approval_notify_failed:{approval_id}",
            )
        return approval_id

    def list_pending(self) -> list[dict[str, Any]]:
        return self.list(status="pending")

    def list(self, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        init_db()
        if status:
            query = "SELECT * FROM approvals WHERE status = ? ORDER BY id DESC LIMIT ?"
            params: tuple[Any, ...] = (status, limit)
        else:
            query = "SELECT * FROM approvals ORDER BY id DESC LIMIT ?"
            params = (limit,)
        with connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._decode_row(dict(row)) for row in rows]

    def approve(self, approval_id: int) -> bool:
        return self._set_status(approval_id, "approved")

    def reject(self, approval_id: int) -> bool:
        return self._set_status(approval_id, "rejected")

    def _set_status(self, approval_id: int, status: str) -> bool:
        init_db()
        with connect() as conn:
            cur = conn.execute(
                "UPDATE approvals SET status = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
                (status, now_kst(), approval_id),
            )
            conn.commit()
            ok = cur.rowcount > 0
        if ok and status == "approved":
            resume_tasks_for_approval(approval_id)
        if ok and status == "rejected":
            block_tasks_for_approval(approval_id)
        if ok:
            emit_wake_signal(
                "approval_changed",
                "approval_store",
                priority=0.9 if status == "approved" else 0.75,
                payload={"approval_id": approval_id, "status": status},
                dedupe_key=f"approval_changed:{approval_id}:{status}",
            )
        return ok

    def _decode_row(self, row: dict[str, Any]) -> dict[str, Any]:
        raw = row.get("proposed_payload_json")
        try:
            row["proposal"] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            row["proposal"] = None
        return row
