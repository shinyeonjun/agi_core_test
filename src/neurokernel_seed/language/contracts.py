from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from neurokernel_seed.harness.action_catalog import ActionDefinition, default_action_catalog
from neurokernel_seed.harness.task_spec import TaskSpecError, task_spec_from_dict


LanguageIntentKind = Literal["chat", "question", "task", "dev_task", "approval", "reject", "unknown"]
PreferenceIntentKind = Literal["none", "preference_update", "question", "reject"]
CapabilityIntentKind = Literal["none", "gap", "ambiguous", "forbidden"]
WorkRouteKind = Literal["runtime_task", "self_patch", "external_work", "unsafe", "clarify"]


class LanguageContractError(ValueError):
    pass


@dataclass(frozen=True)
class LanguageIntent:
    intent: LanguageIntentKind
    reply: str
    task_spec: dict[str, Any] | None = None
    dev_task: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    confidence: float = 0.0
    requires_confirmation: bool = False
    clarifying_question: str | None = None
    safety_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "reply": self.reply,
            "task_spec": self.task_spec,
            "dev_task": self.dev_task,
            "approval": self.approval,
            "confidence": self.confidence,
            "requires_confirmation": self.requires_confirmation,
            "clarifying_question": self.clarifying_question,
            "safety_notes": list(self.safety_notes),
        }


ALLOWED_INTENTS = {"chat", "question", "task", "dev_task", "approval", "reject", "unknown"}
LANGUAGE_INTENT_FIELDS = set(LanguageIntent.__dataclass_fields__)


def validate_language_intent(data: dict[str, Any], *, catalog: dict[str, ActionDefinition] | None = None) -> LanguageIntent:
    if not isinstance(data, dict):
        raise LanguageContractError("LanguageIntent must be an object")
    unknown = sorted(set(data) - LANGUAGE_INTENT_FIELDS)
    if unknown:
        raise LanguageContractError(f"unknown language intent fields: {unknown}")
    intent = str(data.get("intent") or "unknown")
    if intent not in ALLOWED_INTENTS:
        raise LanguageContractError(f"unknown intent: {intent}")
    reply = str(data.get("reply") or "").strip()
    if not reply:
        raise LanguageContractError("reply is required")
    confidence = float(data.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise LanguageContractError("confidence must be between 0 and 1")
    task_spec = data.get("task_spec")
    if task_spec is not None:
        if not isinstance(task_spec, dict):
            raise LanguageContractError("task_spec must be an object or null")
        try:
            task_spec = task_spec_from_dict(task_spec, catalog=catalog or default_action_catalog()).as_dict()
        except TaskSpecError as exc:
            raise LanguageContractError(f"invalid task_spec: {exc}") from exc
    dev_task = data.get("dev_task")
    if dev_task is not None and not isinstance(dev_task, dict):
        raise LanguageContractError("dev_task must be an object or null")
    approval = data.get("approval")
    if approval is not None and not isinstance(approval, dict):
        raise LanguageContractError("approval must be an object or null")
    clarifying_question = data.get("clarifying_question")
    if clarifying_question is not None:
        clarifying_question = str(clarifying_question)
    safety_notes = data.get("safety_notes") or []
    if not isinstance(safety_notes, list):
        raise LanguageContractError("safety_notes must be an array")
    return LanguageIntent(
        intent=intent,  # type: ignore[arg-type]
        reply=reply,
        task_spec=task_spec,
        dev_task=dev_task,
        approval=approval,
        confidence=confidence,
        requires_confirmation=bool(data.get("requires_confirmation", False)),
        clarifying_question=clarifying_question,
        safety_notes=[str(item) for item in safety_notes],
    )


def validate_core_reply(data: dict[str, Any]) -> str:
    if not isinstance(data, dict):
        raise LanguageContractError("CoreReply must be an object")
    unknown = sorted(set(data) - {"reply"})
    if unknown:
        raise LanguageContractError(f"unknown core reply fields: {unknown}")
    reply = str(data.get("reply") or "").strip()
    if not reply:
        raise LanguageContractError("reply is required")
    return reply


ALLOWED_PREFERENCE_KINDS = {"none", "preference_update", "question", "reject"}
ALLOWED_PREFERENCE_KEYS = {
    "response_length",
    "tone",
    "technical_depth",
    "avoid_internal_terms",
    "avoid_emoji",
    "language",
    "avoid_phrases",
}
ALLOWED_PREFERENCE_SCOPES = {"global", "channel"}
ALLOWED_CAPABILITY_KINDS = {"none", "gap", "ambiguous", "forbidden"}
ALLOWED_GAP_TYPES = {
    "missing_action",
    "missing_parameter",
    "missing_target",
    "missing_permission",
    "missing_executor",
    "composite_skill_gap",
    "ambiguous_request",
    "forbidden_request",
}
ALLOWED_CAPABILITY_RISK_LEVELS = {"none", "low", "medium", "high", "forbidden"}
ALLOWED_CAPABILITY_TARGETS = {"local", "orangepi5"}
ALLOWED_WORK_ROUTES = {"runtime_task", "self_patch", "external_work", "unsafe", "clarify"}
ALLOWED_WORK_TYPES = {"runtime_task", "self_patch", "external_work"}
ALLOWED_WORK_PRIORITIES = {"low", "medium", "high"}
ALLOWED_WORK_RISK_LEVELS = {"none", "low", "medium", "high", "forbidden"}


def validate_preference_intent(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LanguageContractError("PreferenceIntent must be an object")
    required = {"kind", "reply", "candidates", "confidence", "requires_confirmation", "clarifying_question", "safety_notes"}
    unknown = sorted(set(data) - required)
    if unknown:
        raise LanguageContractError(f"unknown preference intent fields: {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise LanguageContractError(f"missing preference intent fields: {missing}")
    kind = str(data.get("kind") or "none")
    if kind not in ALLOWED_PREFERENCE_KINDS:
        raise LanguageContractError(f"unknown preference kind: {kind}")
    reply = str(data.get("reply") or "").strip()
    if kind != "none" and not reply:
        raise LanguageContractError("preference reply is required")
    confidence = float(data.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise LanguageContractError("preference confidence must be between 0 and 1")
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise LanguageContractError("preference candidates must be an array")
    checked_candidates = [_validate_preference_candidate(item) for item in candidates]
    safety_notes = data.get("safety_notes") or []
    if not isinstance(safety_notes, list):
        raise LanguageContractError("preference safety_notes must be an array")
    clarifying_question = data.get("clarifying_question")
    return {
        "kind": kind,
        "reply": reply,
        "candidates": checked_candidates,
        "confidence": confidence,
        "requires_confirmation": bool(data.get("requires_confirmation")),
        "clarifying_question": str(clarifying_question) if clarifying_question is not None else None,
        "safety_notes": [str(item) for item in safety_notes],
    }


def _validate_preference_candidate(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise LanguageContractError("preference candidate must be an object")
    required = {"key", "value", "confidence", "evidence", "scope", "source"}
    unknown = sorted(set(item) - required)
    if unknown:
        raise LanguageContractError(f"unknown preference candidate fields: {unknown}")
    missing = sorted(required - set(item))
    if missing:
        raise LanguageContractError(f"missing preference candidate fields: {missing}")
    key = str(item.get("key") or "")
    if key not in ALLOWED_PREFERENCE_KEYS:
        raise LanguageContractError(f"unknown preference key: {key}")
    scope = str(item.get("scope") or "global")
    if scope not in ALLOWED_PREFERENCE_SCOPES:
        raise LanguageContractError(f"unknown preference scope: {scope}")
    source = str(item.get("source") or "")
    if source != "explicit_user_request":
        raise LanguageContractError("preference source must be explicit_user_request")
    confidence = float(item.get("confidence", 0.0))
    if confidence < 0.0 or confidence > 1.0:
        raise LanguageContractError("candidate confidence must be between 0 and 1")
    evidence = str(item.get("evidence") or "").strip()
    if not evidence:
        raise LanguageContractError("candidate evidence is required")
    return {
        "key": key,
        "value": item.get("value"),
        "confidence": confidence,
        "evidence": evidence[:500],
        "scope": scope,
        "source": source,
    }


def validate_capability_intent(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LanguageContractError("CapabilityIntent must be an object")
    required = {"kind", "reply", "gap", "proposal", "confidence", "requires_confirmation", "clarifying_question", "safety_notes"}
    unknown = sorted(set(data) - required)
    if unknown:
        raise LanguageContractError(f"unknown capability intent fields: {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise LanguageContractError(f"missing capability intent fields: {missing}")
    kind = str(data.get("kind") or "none")
    if kind not in ALLOWED_CAPABILITY_KINDS:
        raise LanguageContractError(f"unknown capability intent kind: {kind}")
    confidence = _confidence(data.get("confidence"), "capability confidence")
    gap = data.get("gap")
    proposal = data.get("proposal")
    if kind == "gap":
        if not isinstance(gap, dict):
            raise LanguageContractError("gap intent requires gap object")
        if not isinstance(proposal, dict):
            raise LanguageContractError("gap intent requires proposal object")
    elif kind in {"ambiguous", "forbidden"}:
        if not isinstance(gap, dict):
            raise LanguageContractError(f"{kind} intent requires gap object")
        proposal = None
    else:
        gap = None
        proposal = None
    safety_notes = data.get("safety_notes") or []
    if not isinstance(safety_notes, list):
        raise LanguageContractError("capability safety_notes must be an array")
    clarifying_question = data.get("clarifying_question")
    return {
        "kind": kind,
        "reply": str(data.get("reply") or "").strip(),
        "gap": _validate_gap(gap) if isinstance(gap, dict) else None,
        "proposal": _validate_proposal(proposal) if isinstance(proposal, dict) else None,
        "confidence": confidence,
        "requires_confirmation": bool(data.get("requires_confirmation")),
        "clarifying_question": str(clarifying_question) if clarifying_question is not None else None,
        "safety_notes": [str(item) for item in safety_notes],
    }


def validate_work_route_decision(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LanguageContractError("WorkRouteDecision must be an object")
    required = {"route", "reason", "confidence", "work_item", "requires_confirmation", "clarifying_question", "safety_notes"}
    unknown = sorted(set(data) - required)
    if unknown:
        raise LanguageContractError(f"unknown work route fields: {unknown}")
    missing = sorted(required - set(data))
    if missing:
        raise LanguageContractError(f"missing work route fields: {missing}")
    route = str(data.get("route") or "clarify")
    if route not in ALLOWED_WORK_ROUTES:
        raise LanguageContractError(f"unknown work route: {route}")
    confidence = _confidence(data.get("confidence"), "work route confidence")
    work_item = data.get("work_item")
    if route in {"self_patch", "external_work"}:
        if not isinstance(work_item, dict):
            raise LanguageContractError(f"{route} requires work_item")
        checked_work_item = _validate_work_item(work_item)
        expected_type = "self_patch" if route == "self_patch" else "external_work"
        if checked_work_item["type"] != expected_type:
            raise LanguageContractError(f"{route} requires work_item.type={expected_type}")
    elif work_item is None:
        checked_work_item = None
    elif isinstance(work_item, dict):
        checked_work_item = _validate_work_item(work_item)
    else:
        raise LanguageContractError("work_item must be an object or null")
    safety_notes = data.get("safety_notes") or []
    if not isinstance(safety_notes, list):
        raise LanguageContractError("work route safety_notes must be an array")
    clarifying_question = data.get("clarifying_question")
    reason = str(data.get("reason") or "").strip()
    if not reason:
        raise LanguageContractError("work route reason is required")
    return {
        "route": route,
        "reason": reason[:1000],
        "confidence": confidence,
        "work_item": checked_work_item,
        "requires_confirmation": bool(data.get("requires_confirmation")),
        "clarifying_question": str(clarifying_question).strip() if clarifying_question is not None else None,
        "safety_notes": [str(item)[:500] for item in safety_notes][:20],
    }


def _validate_work_item(item: dict[str, Any]) -> dict[str, Any]:
    required = {"type", "title", "goal", "priority", "risk_level", "deliverables", "open_questions"}
    unknown = sorted(set(item) - required)
    if unknown:
        raise LanguageContractError(f"unknown work item fields: {unknown}")
    missing = sorted(required - set(item))
    if missing:
        raise LanguageContractError(f"missing work item fields: {missing}")
    work_type = str(item.get("type") or "")
    if work_type not in ALLOWED_WORK_TYPES:
        raise LanguageContractError(f"unknown work type: {work_type}")
    priority = str(item.get("priority") or "medium")
    if priority not in ALLOWED_WORK_PRIORITIES:
        raise LanguageContractError(f"unknown work priority: {priority}")
    risk_level = str(item.get("risk_level") or "low")
    if risk_level not in ALLOWED_WORK_RISK_LEVELS:
        raise LanguageContractError(f"unknown work risk_level: {risk_level}")
    deliverables = item.get("deliverables")
    if not isinstance(deliverables, list):
        raise LanguageContractError("work deliverables must be an array")
    open_questions = item.get("open_questions")
    if not isinstance(open_questions, list):
        raise LanguageContractError("work open_questions must be an array")
    title = str(item.get("title") or "").strip()
    goal = str(item.get("goal") or "").strip()
    if not title:
        raise LanguageContractError("work title is required")
    if not goal:
        raise LanguageContractError("work goal is required")
    return {
        "type": work_type,
        "title": title[:200],
        "goal": goal[:2_000],
        "priority": priority,
        "risk_level": risk_level,
        "deliverables": [str(value).strip()[:500] for value in deliverables if str(value).strip()][:20],
        "open_questions": [str(value).strip()[:500] for value in open_questions if str(value).strip()][:20],
    }


def _validate_gap(gap: dict[str, Any]) -> dict[str, Any]:
    required = {"gap_type", "requested_capability", "normalized_request", "matched_existing_actions", "confidence"}
    unknown = sorted(set(gap) - required)
    if unknown:
        raise LanguageContractError(f"unknown capability gap fields: {unknown}")
    missing = sorted(required - set(gap))
    if missing:
        raise LanguageContractError(f"missing capability gap fields: {missing}")
    gap_type = str(gap.get("gap_type") or "")
    if gap_type not in ALLOWED_GAP_TYPES:
        raise LanguageContractError(f"unknown gap_type: {gap_type}")
    matches = gap.get("matched_existing_actions")
    if not isinstance(matches, list):
        raise LanguageContractError("matched_existing_actions must be an array")
    checked_matches = []
    for item in matches[:10]:
        if not isinstance(item, dict):
            raise LanguageContractError("matched action must be an object")
        checked_matches.append(
            {
                "action_id": str(item.get("action_id") or "").strip()[:80],
                "match_score": _confidence(item.get("match_score"), "match_score"),
                "reason": str(item.get("reason") or "").strip()[:500],
            }
        )
    requested = str(gap.get("requested_capability") or "").strip()
    normalized = str(gap.get("normalized_request") or "").strip()
    if not requested:
        raise LanguageContractError("requested_capability is required")
    if not normalized:
        raise LanguageContractError("normalized_request is required")
    return {
        "gap_type": gap_type,
        "requested_capability": requested[:160],
        "normalized_request": normalized[:500],
        "matched_existing_actions": checked_matches,
        "confidence": _confidence(gap.get("confidence"), "gap confidence"),
    }


def _validate_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    required = {
        "action_id",
        "capability_name",
        "purpose",
        "target",
        "risk_level",
        "side_effect",
        "requires_approval",
        "inputs",
        "outputs",
        "implementation_hint",
        "test_plan",
        "safety_notes",
        "confidence",
        "approval_required_for_implementation",
        "activation_requires_tests",
    }
    unknown = sorted(set(proposal) - required)
    if unknown:
        raise LanguageContractError(f"unknown capability proposal fields: {unknown}")
    missing = sorted(required - set(proposal))
    if missing:
        raise LanguageContractError(f"missing capability proposal fields: {missing}")
    action_id = str(proposal.get("action_id") or "").strip()
    if not _valid_action_id(action_id):
        raise LanguageContractError("proposal action_id must be snake_case and 3-64 chars")
    target = str(proposal.get("target") or "")
    if target not in ALLOWED_CAPABILITY_TARGETS:
        raise LanguageContractError(f"unknown proposal target: {target}")
    risk_level = str(proposal.get("risk_level") or "")
    if risk_level not in ALLOWED_CAPABILITY_RISK_LEVELS:
        raise LanguageContractError(f"unknown proposal risk_level: {risk_level}")
    for schema_field in ("inputs", "outputs", "implementation_hint"):
        if not isinstance(proposal.get(schema_field), dict):
            raise LanguageContractError(f"{schema_field} must be an object")
    test_plan = proposal.get("test_plan")
    if not isinstance(test_plan, list):
        raise LanguageContractError("test_plan must be an array")
    safety_notes = proposal.get("safety_notes") or []
    if not isinstance(safety_notes, list):
        raise LanguageContractError("proposal safety_notes must be an array")
    return {
        "action_id": action_id,
        "capability_name": str(proposal.get("capability_name") or "").strip()[:160],
        "purpose": str(proposal.get("purpose") or "").strip()[:1000],
        "target": target,
        "risk_level": risk_level,
        "side_effect": bool(proposal.get("side_effect")),
        "requires_approval": bool(proposal.get("requires_approval")),
        "inputs": proposal.get("inputs") or {},
        "outputs": proposal.get("outputs") or {},
        "implementation_hint": proposal.get("implementation_hint") or {},
        "test_plan": [item for item in test_plan if isinstance(item, dict)][:20],
        "safety_notes": [str(item)[:500] for item in safety_notes][:20],
        "confidence": _confidence(proposal.get("confidence"), "proposal confidence"),
        "approval_required_for_implementation": bool(proposal.get("approval_required_for_implementation")),
        "activation_requires_tests": bool(proposal.get("activation_requires_tests")),
    }


def _confidence(value: Any, field: str) -> float:
    confidence = float(value if value is not None else 0.0)
    if confidence < 0.0 or confidence > 1.0:
        raise LanguageContractError(f"{field} must be between 0 and 1")
    return confidence


def _valid_action_id(action_id: str) -> bool:
    if len(action_id) < 3 or len(action_id) > 64:
        return False
    if not action_id[0].isalpha():
        return False
    return all(char.islower() or char.isdigit() or char == "_" for char in action_id)
