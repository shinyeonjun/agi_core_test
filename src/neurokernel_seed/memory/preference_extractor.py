from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neurokernel_seed.language.contracts import LanguageContractError, validate_preference_intent


@dataclass(frozen=True)
class PreferenceCandidate:
    key: str
    value: Any
    reply: str
    confidence: float = 0.9
    source: str = "explicit_user_request"
    scope: str = "global"
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "reply": self.reply,
            "confidence": self.confidence,
            "source": self.source,
            "scope": self.scope,
            "evidence": self.evidence,
        }


def extract_preference_candidates(text: str) -> list[PreferenceCandidate]:
    """Raw text is not a preference data source.

    Preference storage must come from a validated LanguageOrgan
    PreferenceIntent. This keeps durable memory from becoming a keyword table.
    """

    return []


def preference_candidates_from_intent(intent: dict[str, Any]) -> list[PreferenceCandidate]:
    checked = validate_preference_intent(intent)
    if checked["kind"] != "preference_update":
        return []
    reply = str(checked["reply"])
    candidates: list[PreferenceCandidate] = []
    for item in checked["candidates"]:
        candidates.append(
            PreferenceCandidate(
                key=str(item["key"]),
                value=item.get("value"),
                reply=reply,
                confidence=float(item["confidence"]),
                source=str(item["source"]),
                scope=str(item["scope"]),
                evidence=str(item["evidence"]),
            )
        )
    return candidates


def safe_preference_candidates_from_intent(intent: dict[str, Any]) -> list[PreferenceCandidate]:
    try:
        return preference_candidates_from_intent(intent)
    except LanguageContractError:
        return []
