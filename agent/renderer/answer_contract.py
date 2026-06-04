from __future__ import annotations

import re
from typing import Any

ADVICE_TARGETS = {"idea", "question", "self_improvement", "architecture", "capabilities"}
STATUS_TARGETS = {"status", "capabilities", "architecture"}
TEMPLATE_ESCAPE_PHRASES = (
    "답변 생성이 잠깐 매끄럽지 않았어",
    "입력은 받았고",
    "작업 지시라면",
    "!work에서 진행 여부",
    "!work에서 확인",
    "짧게만 말할게",
)
ADVICE_MARKERS = (
    "1순위", "2순위", "3순위", "우선", "먼저", "다음", "추천", "좋아", "낫", "개선", "업데이트", "줄이", "강화", "분리", "추가",
)
QUESTION_TOKENS = ("?", "？", "뭐", "어떻게", "어케", "왜", "가능", "좋을", "어때", "어떰", "할까", "뭘")
ADVICE_TOKENS = ("뭘 더", "뭐 더", "어떻게 개선", "업데이트", "좋을거", "좋을까", "추천", "우선순위", "어케 개선")
SELF_REPORT_TOKENS = ("개선", "상태", "능력", "뭐 할", "뭘 할", "자가", "자기", "코어", "core")


def _contains_any(text: str, tokens: tuple[str, ...] | set[str]) -> bool:
    return any(token in text for token in tokens)


def _focus_from_context(context: dict[str, Any]) -> str:
    return str(context.get("focus") or "general") if isinstance(context, dict) else "general"


def _compact_summary(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    return cleaned[:180]


def build_answer_contract(user_message: str, interpretation: dict[str, Any] | None = None, self_report_context: dict[str, Any] | None = None, user_goal: dict[str, Any] | None = None) -> dict[str, Any]:
    interpretation = interpretation or {}
    self_report_context = self_report_context or {}
    user_goal = user_goal or {}
    text = user_message.strip()
    lower = text.lower()
    target = str(interpretation.get("target") or "")
    intent = str(interpretation.get("intent") or "")
    focus = _focus_from_context(self_report_context)
    is_question = _contains_any(text, QUESTION_TOKENS)
    asks_advice = _contains_any(lower, ADVICE_TOKENS) or (is_question and _contains_any(lower, ("개선", "업데이트", "더", "좋")))
    asks_self = _contains_any(lower, SELF_REPORT_TOKENS) or focus in {"change", "capability", "status", "failure", "research"}

    kind = "general"
    required_moves: list[str] = ["answer_user_message"]
    min_recommendations = 0
    evidence_required = False
    if user_goal:
        kind = "task_ack"
        required_moves = ["acknowledge_registered_task", "state_next_visibility"]
    elif asks_advice:
        kind = "advice"
        required_moves = ["direct_answer", "give_prioritized_candidates", "mention_reason"]
        min_recommendations = 2
        evidence_required = asks_self
    elif focus in {"change", "capability", "status", "failure", "research"} or target in STATUS_TARGETS:
        kind = "self_report"
        required_moves = ["direct_answer", "use_grounded_evidence", "state_limit"]
        evidence_required = True
    elif target in ADVICE_TARGETS or is_question:
        kind = "direct_question"
        required_moves = ["direct_answer"]

    forbidden_moves = ["template_escape", "work_pointer_only", "internal_field_dump", "unbounded_autonomy_claim"]
    return {
        "kind": kind,
        "user_question_summary": _compact_summary(text),
        "intent": intent,
        "target": target,
        "focus": focus,
        "direct_answer_required": kind in {"advice", "self_report", "direct_question"},
        "evidence_required": evidence_required,
        "min_recommendations": min_recommendations,
        "required_moves": required_moves,
        "forbidden_moves": forbidden_moves,
        "fallback_strategy": "prioritized_advice" if kind == "advice" else "grounded_self_report" if kind == "self_report" else "direct_answer",
        "requires_core_subject": kind in {"self_report", "direct_question"} and asks_self,
        "plain_language_required": kind in {"advice", "self_report", "direct_question"},
    }


def _count_recommendation_markers(text: str) -> int:
    numbered = len(re.findall(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", text))
    lexical = sum(1 for token in ADVICE_MARKERS if token in text)
    return max(numbered, min(lexical, 4))


def validate_answer_contract(text: str, contract: dict[str, Any] | None) -> dict[str, Any]:
    if not contract:
        return {"ok": True, "violations": []}
    violations: list[str] = []
    kind = str(contract.get("kind") or "general")
    if any(phrase in text for phrase in TEMPLATE_ESCAPE_PHRASES):
        violations.append("template_escape")
    if "!work" in text and len(text.strip()) < 180 and kind in {"advice", "direct_question", "self_report"}:
        violations.append("work_pointer_only")
    if contract.get("requires_core_subject") and not ("Core" in text or "코어" in text):
        violations.append("missing_core_subject")
    min_recommendations = int(contract.get("min_recommendations") or 0)
    if min_recommendations and _count_recommendation_markers(text) < min_recommendations:
        violations.append("missing_recommendations")
    if contract.get("direct_answer_required") and len(text.strip()) < 20:
        violations.append("too_short_for_direct_answer")
    return {"ok": not violations, "violations": sorted(set(violations))}