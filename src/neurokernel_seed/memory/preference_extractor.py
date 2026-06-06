from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXPLICIT_MARKERS = (
    "앞으로",
    "이제부터",
    "다음부터",
    "항상",
    "계속",
    "기억해",
    "저장해",
    "맞춰줘",
    "해줘",
    "하지마",
    "하지 말",
    "쓰지마",
    "쓰지 말",
)

SENSITIVE_MARKERS = (
    "비번",
    "비밀번호",
    "패스워드",
    "password",
    "token",
    "토큰",
    "api key",
    "apikey",
    "secret",
    "시크릿",
    "주민번호",
    "계좌",
)


@dataclass(frozen=True)
class PreferenceCandidate:
    key: str
    value: Any
    reply: str
    confidence: float = 0.9
    source: str = "explicit_user_request"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "reply": self.reply,
            "confidence": self.confidence,
            "source": self.source,
        }


def extract_preference_candidates(text: str) -> list[PreferenceCandidate]:
    normalized = _normalize(text)
    if not normalized or _has_sensitive_marker(normalized):
        return []
    if not _looks_explicit(normalized):
        return []
    candidates: list[PreferenceCandidate] = []
    candidates.extend(_length_preferences(normalized))
    candidates.extend(_tone_preferences(normalized))
    candidates.extend(_technical_depth_preferences(normalized))
    candidates.extend(_format_preferences(normalized))
    return _dedupe(candidates)


def _length_preferences(text: str) -> list[PreferenceCandidate]:
    result = []
    if any(word in text for word in ["짧게", "간단히", "간결하게", "요약해서", "요점만"]):
        result.append(PreferenceCandidate("response_length", "short", "알겠어. 앞으로 답변은 짧게 할게."))
    if any(word in text for word in ["자세히", "길게", "상세히", "디테일하게"]):
        result.append(PreferenceCandidate("response_length", "long", "알겠어. 앞으로는 더 자세히 설명할게."))
    return result


def _tone_preferences(text: str) -> list[PreferenceCandidate]:
    result = []
    if any(word in text for word in ["편하게", "반말", "친근하게", "캐주얼하게"]):
        result.append(PreferenceCandidate("tone", "casual", "알겠어. 앞으로 좀 더 편하게 말할게."))
    if any(word in text for word in ["존댓말", "정중하게", "공손하게"]):
        result.append(PreferenceCandidate("tone", "polite", "알겠어. 앞으로 더 정중하게 말할게."))
    if any(word in text for word in ["차분하게", "담백하게"]):
        result.append(PreferenceCandidate("tone", "calm", "알겠어. 앞으로 더 담백하게 말할게."))
    return result


def _technical_depth_preferences(text: str) -> list[PreferenceCandidate]:
    result = []
    if any(word in text for word in ["비개발자", "쉽게", "쉬운 말", "전문용어 줄", "코드용어 줄", "내부용어 쓰지"]):
        result.append(PreferenceCandidate("technical_depth", "low", "알겠어. 앞으로는 더 쉽게 풀어서 말할게."))
    if any(word in text for word in ["전문적으로", "깊게", "기술적으로", "개발자 관점"]):
        result.append(PreferenceCandidate("technical_depth", "high", "알겠어. 앞으로는 기술적으로 더 깊게 말할게."))
    if any(word in text for word in ["내부용어 쓰지", "json 말하지", "변수명 말하지", "코드관련된거 못뱉"]):
        result.append(PreferenceCandidate("avoid_internal_terms", True, "알겠어. 내부 용어는 최대한 숨길게."))
    return result


def _format_preferences(text: str) -> list[PreferenceCandidate]:
    result = []
    if any(word in text for word in ["이모지 쓰지", "이모티콘 쓰지", "emoji 쓰지"]):
        result.append(PreferenceCandidate("avoid_emoji", True, "알겠어. 이모지는 쓰지 않을게."))
    if any(word in text for word in ["한국어로", "한글로"]):
        result.append(PreferenceCandidate("language", "ko", "알겠어. 한국어로 말할게."))
    return result


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _looks_explicit(text: str) -> bool:
    return any(marker in text for marker in EXPLICIT_MARKERS)


def _has_sensitive_marker(text: str) -> bool:
    return any(marker in text for marker in SENSITIVE_MARKERS)


def _dedupe(candidates: list[PreferenceCandidate]) -> list[PreferenceCandidate]:
    result: dict[str, PreferenceCandidate] = {}
    for candidate in candidates:
        result[candidate.key] = candidate
    return list(result.values())
