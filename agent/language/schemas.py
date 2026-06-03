from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ALLOWED_INTENTS = {
    "chat",
    "feedback",
    "style_feedback",
    "brainstorm",
    "task_request",
    "report_request",
    "project_request",
    "self_improvement_request",
    "approval",
    "control",
    "memory_instruction",
    "unknown",
}
ALLOWED_SENTIMENTS = {"positive", "negative", "neutral", "mixed", "unknown"}


@dataclass(frozen=True)
class Interpretation:
    intent: str = "unknown"
    sentiment: str = "unknown"
    target: str | None = None
    confidence: float = 0.0
    style_update: dict[str, Any] = field(default_factory=dict)
    memory_instruction: bool = False
    execution: dict[str, Any] = field(default_factory=lambda: {"requires_action": False})
    idea: dict[str, Any] = field(default_factory=dict)
    safety_notes: list[str] = field(default_factory=list)
    engine: str = "unknown"
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "sentiment": self.sentiment,
            "target": self.target,
            "confidence": self.confidence,
            "style_update": self.style_update,
            "memory_instruction": self.memory_instruction,
            "execution": self.execution,
            "idea": self.idea,
            "safety_notes": self.safety_notes,
            "engine": self.engine,
            "fallback_reason": self.fallback_reason,
        }


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if item is not None and item != [] and item != {}}


def _as_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _as_notes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item)[:300] for item in value[:8]]


def normalize_interpretation(value: dict[str, Any] | Interpretation, *, engine: str = "unknown", fallback_reason: str | None = None) -> Interpretation:
    if isinstance(value, Interpretation):
        if value.engine == engine and value.fallback_reason == fallback_reason:
            return value
        data = value.to_dict()
    else:
        data = dict(value or {})

    intent = str(data.get("intent") or "unknown")
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"
    target_text = str(data.get("target") or "")
    if intent == "unknown" and target_text in {"architecture", "capabilities", "status", "help", "greeting", "question"}:
        intent = "chat"
    sentiment = str(data.get("sentiment") or "unknown")
    if sentiment not in ALLOWED_SENTIMENTS:
        sentiment = "unknown"
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    execution = _as_dict(data.get("execution"))
    if "requires_action" not in execution:
        execution["requires_action"] = intent in {"task_request", "project_request", "report_request", "self_improvement_request"}
    return Interpretation(
        intent=intent,
        sentiment=sentiment,
        target=target_text[:160] if data.get("target") is not None else None,
        confidence=confidence,
        style_update=_as_dict(data.get("style_update")),
        memory_instruction=_as_bool(data.get("memory_instruction")),
        execution=execution,
        idea=_as_dict(data.get("idea")),
        safety_notes=_as_notes(data.get("safety_notes")),
        engine=str(data.get("engine") or engine),
        fallback_reason=data.get("fallback_reason") or fallback_reason,
    )
