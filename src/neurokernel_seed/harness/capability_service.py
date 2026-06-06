from __future__ import annotations

from pathlib import Path
from typing import Any

from .action_catalog import ActionDefinition
from .ids import new_id
from .memory import HarnessMemory
from .work_service import WorkItemService


class CapabilityProposalService:
    def __init__(self, *, db_path: str | Path, catalog: dict[str, ActionDefinition], work_items: WorkItemService):
        self.db_path = Path(db_path)
        self.catalog = catalog
        self.work_items = work_items

    def list_gaps(self, *, limit: int = 20, status: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            gaps = memory.list_capability_gaps(limit=limit, status=status)
        return {"gaps": gaps}

    def list_proposals(self, *, limit: int = 20, status: str | None = None) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            proposals = memory.list_capability_proposals(limit=limit, status=status)
        return {"proposals": proposals}

    def get_proposal(self, proposal_id: str) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            proposal = memory.get_capability_proposal(proposal_id)
            events = memory.proposal_events(proposal_id)
        return {"proposal": proposal, "events": events}

    def create_from_intent(
        self,
        *,
        user_text: str,
        capability_intent: dict[str, Any],
        user_id: str | None = None,
        channel_id: str | None = None,
        language_intent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = str(capability_intent.get("kind") or "none")
        if kind == "none":
            return {"created": False, "kind": "none", "reply": capability_intent.get("reply") or ""}

        gap = capability_intent.get("gap")
        if not isinstance(gap, dict):
            return {"created": False, "kind": "none", "reply": "능력 후보를 만들 정보가 아직 부족해."}

        gap_row = self._record_gap(user_text=user_text, kind=kind, gap=gap, capability_intent=capability_intent, user_id=user_id, channel_id=channel_id)
        if kind in {"ambiguous", "forbidden"}:
            return {
                "created": False,
                "kind": kind,
                "gap": gap_row,
                "reply": capability_intent.get("clarifying_question") or capability_intent.get("reply") or "이건 바로 능력 후보로 만들기 애매해.",
            }

        proposal = capability_intent.get("proposal")
        if not isinstance(proposal, dict):
            return {"created": False, "kind": "none", "gap": gap_row, "reply": "능력 후보 초안이 비어 있어서 저장하지 않았어."}

        action_id = str(proposal.get("action_id") or "").strip()
        target = str(proposal.get("target") or "orangepi5").strip()
        if action_id in self.catalog:
            return {
                "created": False,
                "kind": "existing_action",
                "gap": gap_row,
                "existing_action": self.catalog[action_id].as_dict(),
                "reply": "이미 비슷한 기능이 등록되어 있어. 새 후보로 만들지는 않을게.",
            }

        duplicate = self._find_duplicate(action_id=action_id, target=target, gap_id=str(gap_row["gap_id"]), user_text=user_text, actor=user_id or "api")
        if duplicate:
            return {
                "created": False,
                "kind": "duplicate",
                "gap": gap_row,
                "proposal": duplicate,
                "reply": "이미 같은 능력 후보가 올라와 있어. 새로 만들지 않고 기존 후보에 묶어둘게.",
            }

        return self._create_proposal_work_item(
            user_text=user_text,
            capability_intent=capability_intent,
            gap=gap,
            gap_row=gap_row,
            proposal=proposal,
            action_id=action_id,
            user_id=user_id,
            channel_id=channel_id,
            language_intent=language_intent,
        )

    def transition(self, proposal_id: str, next_status: str, *, actor: str = "api", reason: str | None = None) -> dict[str, Any]:
        if next_status == "active":
            raise ValueError("capability proposals cannot become active in v1")
        enqueue_work_id = None
        with HarnessMemory(self.db_path) as memory:
            proposal = memory.transition_capability_proposal(proposal_id, next_status, actor=actor, payload={"reason": reason})
            work_id = proposal.get("work_id")
            if work_id:
                work_next = {"approved_for_dev": "accepted", "deferred": "deferred", "rejected": "rejected"}.get(next_status)
                if work_next:
                    try:
                        memory.transition_work_item(str(work_id), work_next, actor=actor, payload={"proposal_id": proposal_id, "reason": reason})
                        if work_next == "accepted":
                            enqueue_work_id = str(work_id)
                    except ValueError:
                        memory.add_work_event(str(work_id), "proposal_transition", actor=actor, payload={"proposal_status": next_status, "reason": reason})
            events = memory.proposal_events(proposal_id)
        result: dict[str, Any] = {"proposal": proposal, "events": events}
        if enqueue_work_id:
            result["queue"] = self.work_items.enqueue(enqueue_work_id, actor=actor)
        return result

    def _record_gap(
        self,
        *,
        user_text: str,
        kind: str,
        gap: dict[str, Any],
        capability_intent: dict[str, Any],
        user_id: str | None,
        channel_id: str | None,
    ) -> dict[str, Any]:
        status = "unsafe" if kind == "forbidden" else "needs_clarification" if kind == "ambiguous" else "detected"
        with HarnessMemory(self.db_path) as memory:
            return memory.create_capability_gap(
                gap_id=new_id("gap"),
                user_id=user_id,
                channel_id=channel_id,
                request_text=user_text,
                normalized_request=str(gap.get("normalized_request") or user_text),
                gap_type=str(gap.get("gap_type") or "missing_action"),
                requested_capability=str(gap.get("requested_capability") or ""),
                matched_actions=gap.get("matched_existing_actions") if isinstance(gap.get("matched_existing_actions"), list) else [],
                confidence=float(gap.get("confidence") or capability_intent.get("confidence") or 0.0),
                status=status,
            )

    def _find_duplicate(self, *, action_id: str, target: str, gap_id: str, user_text: str, actor: str) -> dict[str, Any] | None:
        with HarnessMemory(self.db_path) as memory:
            duplicate = memory.find_open_capability_proposal(action_id=action_id, target=target)
            if not duplicate:
                return None
            memory.add_proposal_event(
                str(duplicate["proposal_id"]),
                "deduplicated",
                actor=actor,
                payload={"gap_id": gap_id, "request_text": user_text[:500]},
            )
            memory.conn.commit()
            return duplicate

    def _create_proposal_work_item(
        self,
        *,
        user_text: str,
        capability_intent: dict[str, Any],
        gap: dict[str, Any],
        gap_row: dict[str, Any],
        proposal: dict[str, Any],
        action_id: str,
        user_id: str | None,
        channel_id: str | None,
        language_intent: dict[str, Any] | None,
    ) -> dict[str, Any]:
        actor = user_id or "language_organ"
        with HarnessMemory(self.db_path) as memory:
            work_row = memory.create_work_item(
                work_id=new_id("work", action_id),
                work_type="self_patch",
                title=str(proposal.get("capability_name") or action_id),
                goal=str(proposal.get("purpose") or user_text),
                status="proposed",
                priority="medium",
                risk_level=str(proposal.get("risk_level") or "low"),
                owner_user_id=user_id,
                channel_id=channel_id,
                linked_entity_type="capability_gap",
                linked_entity_id=str(gap_row["gap_id"]),
                route_reason=str(gap.get("normalized_request") or user_text),
                confidence=float(proposal.get("confidence") or capability_intent.get("confidence") or 0.0),
                metadata={
                    "action_id": action_id,
                    "request_text": user_text[:1000],
                    "deliverables": ["implementation_patch", "tests", "activation_candidate"],
                    "test_plan": proposal.get("test_plan") or [],
                    "capability_intent": capability_intent,
                },
                actor=actor,
            )
            proposal_row = memory.create_capability_proposal(
                proposal_id=new_id("prop", action_id),
                gap_id=str(gap_row["gap_id"]),
                proposal=proposal,
                status="proposed",
                actor=actor,
                work_id=str(work_row["work_id"]),
            )
            memory.add_work_event(str(work_row["work_id"]), "linked_capability_proposal", actor=actor, payload={"proposal_id": proposal_row["proposal_id"]})
            memory.conn.commit()
        return {
            "created": True,
            "kind": "gap",
            "gap": gap_row,
            "proposal": proposal_row,
            "work_item": work_row,
            "reply": capability_intent.get("reply") or "지금은 그 능력이 없어. 능력 후보로 올려둘게.",
            "language_intent": language_intent or {},
        }
