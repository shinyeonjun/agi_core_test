from __future__ import annotations

import re
from typing import Any

from agent.language.schemas import Interpretation, normalize_interpretation

POSITIVE_KEYWORDS = ("좋다", "맞아", "ㅇㅇ", "계속", "이 방향", "오케이", "굿", "좋아")
NEGATIVE_KEYWORDS = ("아니", "그게 아니라", "틀림", "너무 장황", "다시", "이상한데", "별로")

TASK_KEYWORDS = (
    "해줘", "해봐", "하자", "가자", "ㄱㄱ", "만들", "구현", "개발", "디벨롭", "정리",
    "조사", "분석", "보고서", "작성", "추가", "수정", "개선", "고쳐", "처리", "실행",
    "테스트", "검증", "설계", "생성", "빌드", "스캐폴드", "초안",
    "implement", "create", "build", "fix", "develop", "write", "summarize", "test",
)

NON_TASK_PATTERNS = (
    re.compile(r"^\s*(ㅎㅇ|하이|안녕|hello|hi|hey)\s*[.!?ㅋㅎ]*\s*$", re.IGNORECASE),
    re.compile(r"(지금\s*)?(뭐\s*하고|뭐해|상태|살아|정상|도움|명령|사용법)", re.IGNORECASE),
)

STYLE_FEEDBACK_PATTERNS: tuple[tuple[str, re.Pattern[str], dict[str, Any], str], ...] = (
    ("positive_style", re.compile(r"(말투|느낌|톤).*(좋|맞|계속|기억)|이런\s*식으로\s*계속", re.IGNORECASE), {"positive_signal": True}, "positive"),
    ("too_ai_like", re.compile(r"(너무|좀).*(ai|챗봇|기계|정중|딱딱)|ai\s*같", re.IGNORECASE), {"avoid": ["AI-like praise", "overly polite filler"], "tone": "natural_blunt"}, "negative"),
    ("shorter", re.compile(r"(짧게|간단히|줄여|너무\s*길)", re.IGNORECASE), {"detail_level": "shorter"}, "neutral"),
    ("more_detail", re.compile(r"(자세히|디테일|구체적|왜인지)", re.IGNORECASE), {"detail_level": "more_detail"}, "neutral"),
    ("colder", re.compile(r"(말투|톤|어조|답변|응답).*(냉정|직설|팩트|비판)|(냉정|직설|팩트|비판).*(말투|톤|어조|답변|응답)", re.IGNORECASE), {"tone": "calm_blunt", "structure": "findings_first"}, "neutral"),
    ("softer", re.compile(r"(부드럽|친절|덜\s*세게)", re.IGNORECASE), {"tone": "warm_direct"}, "neutral"),
    ("no_emoji", re.compile(r"(이모지|emoji).*(쓰지|빼|싫)", re.IGNORECASE), {"emoji": False}, "negative"),
)


def detect_style_feedback_rule(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    for feedback_type, pattern, extracted, sentiment in STYLE_FEEDBACK_PATTERNS:
        if pattern.search(cleaned):
            return {
                "feedback_type": feedback_type,
                "sentiment": sentiment,
                "extracted_preference": dict(extracted),
            }
    return None


def detect_feedback_rule(user_text: str) -> str:
    lowered = user_text.lower()
    if any(keyword in lowered for keyword in NEGATIVE_KEYWORDS):
        return "negative"
    if any(keyword in lowered for keyword in POSITIVE_KEYWORDS):
        return "positive"
    return "neutral"


def classify_user_goal_kind_rule(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["fastapi", "api", "프로젝트", "앱", "서비스", "봇", "사이트"]):
        return "project_spec"
    if any(token in lowered for token in ["보고서", "정리", "요약", "조사", "분석", "리포트"]):
        return "report"
    if any(token in lowered for token in ["개선", "수정", "고쳐", "리팩터", "디벨롭"]):
        return "improvement_plan"
    if any(token in lowered for token in ["테스트", "검증", "실험"]):
        return "workspace_experiment"
    return "task_note"


def is_task_request_rule(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("!"):
        return False
    if detect_style_feedback_rule(cleaned):
        return False
    if any(pattern.search(cleaned) for pattern in NON_TASK_PATTERNS):
        return False
    lowered = cleaned.lower()
    return any(keyword in lowered for keyword in TASK_KEYWORDS)


def _chat_target(text: str) -> str | None:
    lowered = text.strip().lower()
    normalized = text.replace(" ", "")
    if lowered in {"hi", "hello", "hey", "ㅎㅇ", "하이", "안녕", "안녕하세요", "헬로", "ㅇㅇ"} or lowered.startswith(("ㅎㅇ", "안녕")):
        return "greeting"
    if any(token in text for token in ["상태", "뭐 하고", "뭐 하는", "뭐해", "살아", "정상", "체크", "확인"]) or any(token in normalized for token in ["뭐하고", "뭐하는", "뭐해", "하고있", "하는중"]):
        return "status"
    if any(token in text for token in ["할 수 있는", "뭘 할 수", "뭐 할 수", "가능한", "능력", "할수있는"]):
        return "capabilities"
    if any(token in text for token in ["코어", "구조", "이루어져", "구성", "아키텍처", "어떻게 되어"]):
        return "architecture"
    if any(token in text.lower() for token in ["도움", "명령", "help", "뭐 할", "사용법", "기능"]):
        return "help"
    if text.endswith("?") or text.endswith("？"):
        return "question"
    return None


class FallbackRuleLanguageEngine:
    name = "fallback_rule"

    def interpret_user_message(self, text: str, context: dict[str, Any] | None = None) -> Interpretation:
        cleaned = text.strip()
        if not cleaned:
            return normalize_interpretation({"intent": "unknown", "confidence": 0.0}, engine=self.name)

        style = detect_style_feedback_rule(cleaned)
        if style:
            style_update = dict(style["extracted_preference"])
            style_update["feedback_type"] = style["feedback_type"]
            return normalize_interpretation(
                {
                    "intent": "style_feedback",
                    "sentiment": style["sentiment"],
                    "target": "response_style",
                    "confidence": 0.82,
                    "style_update": style_update,
                    "memory_instruction": True,
                    "execution": {"requires_action": False},
                    "safety_notes": ["style_only_not_policy"],
                },
                engine=self.name,
            )

        if is_task_request_rule(cleaned):
            task_kind = classify_user_goal_kind_rule(cleaned)
            intent = "project_request" if task_kind == "project_spec" else "report_request" if task_kind == "report" else "task_request"
            return normalize_interpretation(
                {
                    "intent": intent,
                    "sentiment": detect_feedback_rule(cleaned),
                    "target": task_kind,
                    "confidence": 0.78,
                    "execution": {"requires_action": True, "suggested_queue_type": task_kind, "risk_hint": "unknown"},
                    "safety_notes": ["policy_engine_must_decide"],
                },
                engine=self.name,
            )

        lowered = cleaned.lower()
        if any(token in cleaned for token in ["할 수 있는", "뭘 할 수", "뭐 할 수", "가능한", "능력", "할수있는"]):
            return normalize_interpretation(
                {
                    "intent": "chat",
                    "sentiment": "neutral",
                    "target": "capabilities",
                    "confidence": 0.82,
                    "execution": {"requires_action": False},
                },
                engine=self.name,
            )
        if any(token in cleaned for token in ["코어", "구조", "이루어져", "구성", "아키텍처", "어떻게 되어"]):
            return normalize_interpretation(
                {
                    "intent": "chat",
                    "sentiment": "neutral",
                    "target": "architecture",
                    "confidence": 0.82,
                    "execution": {"requires_action": False},
                },
                engine=self.name,
            )
        if any(token in lowered for token in ["아이디어", "어떰", "어때", "가능", "쓸만", "평가"]):
            return normalize_interpretation(
                {
                    "intent": "brainstorm",
                    "sentiment": detect_feedback_rule(cleaned),
                    "target": "idea",
                    "confidence": 0.74,
                    "idea": {"summary": cleaned[:240]},
                    "execution": {"requires_action": False, "suggested_next_step": "evaluate_idea"},
                },
                engine=self.name,
            )

        feedback = detect_feedback_rule(cleaned)
        if feedback in {"positive", "negative"}:
            return normalize_interpretation(
                {
                    "intent": "feedback",
                    "sentiment": feedback,
                    "target": "last_turn",
                    "confidence": 0.68,
                    "memory_instruction": feedback == "negative",
                    "execution": {"requires_action": False},
                },
                engine=self.name,
            )

        target = _chat_target(cleaned)
        return normalize_interpretation(
            {
                "intent": "report_request" if target == "status" else "chat",
                "sentiment": "neutral",
                "target": target,
                "confidence": 0.64 if target else 0.52,
                "execution": {"requires_action": False},
            },
            engine=self.name,
        )
