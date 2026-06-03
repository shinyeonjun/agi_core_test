from __future__ import annotations

import re

FORBIDDEN_PHRASES = (
    "auto sudo execution", "sudo auto execution", "rm -rf", "/etc auto edit", "/etc/",
    "SSH key access", "ssh key", "consciousness emerged", "root permission handled", "AGI achieved",
)

INTERNAL_FIELD_PATTERNS = (
    re.compile(r"\bselected_goal_id\b", re.IGNORECASE),
    re.compile(r"\buser_goal_created\b", re.IGNORECASE),
    re.compile(r"\bdecision_json\b", re.IGNORECASE),
    re.compile(r"\bpolicy_summary\b", re.IGNORECASE),
    re.compile(r"\blanguage_interpretation\b", re.IGNORECASE),
    re.compile(r"\brenderer_hint\b", re.IGNORECASE),
    re.compile(r"\bsource_event_id\b", re.IGNORECASE),
    re.compile(r"\bmust_include\b", re.IGNORECASE),
    re.compile(r"\bmust_not_include\b", re.IGNORECASE),
    re.compile(r"\bruntime_self_map\b", re.IGNORECASE),
    re.compile(r"`(?:selected_goal_id|user_goal_created|policy_summary|language_interpretation|runtime_self_map|renderer)`", re.IGNORECASE),
)

COMMAND_REFERENCE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:sudo|apt(?:-get)?\s+install)(?![A-Za-z0-9_])", re.IGNORECASE)
DIRECT_COMMAND_PATTERN = re.compile(r"(?:^|\n|`)\s*(?:sudo\s+)?(?:apt(?:-get)?\s+install|rm\s+-rf)(?![A-Za-z0-9_])", re.IGNORECASE)
UNSAFE_COMMAND_INTENT_PATTERN = re.compile(
    r"(?:sudo|apt(?:-get)?\s+install|rm\s+-rf).{0,40}"
    r"(?:실행해|실행하면|입력해|치면|써서|run|execute|install it)",
    re.IGNORECASE,
)
SAFE_COMMAND_CONTEXT_PATTERN = re.compile(
    r"(?:"
    r"(?:sudo|apt(?:-get)?\s+install|패키지\s*설치|시스템\s*변경).{0,50}"
    r"(?:안\s*해|않아|않는다|못\s*해|차단|승인\s*(?:필요|대기|없이는|없이)|물어|보류)"
    r"|(?:승인\s*(?:필요|대기)).{0,50}"
    r"(?:sudo|apt(?:-get)?\s+install|패키지\s*설치|시스템\s*변경)"
    r"|(?:승인\s*(?:없이는|받기\s*전)|허락\s*없이는).{0,50}"
    r"(?:sudo|apt(?:-get)?\s+install|패키지\s*설치|시스템\s*변경).{0,50}"
    r"(?:안\s*해|않아|않는다|못\s*해|차단|물어|보류)"
    r")",
    re.IGNORECASE,
)
APPROVAL_BYPASS_PATTERN = re.compile(
    r"(?:승인\s*없이|허락\s*없이|without approval).{0,40}(?:해도|가능|실행|변경|바꿀|설치|can|allowed)",
    re.IGNORECASE,
)


def validate_output(text: str, must_include: list[str] | None = None, must_not_include: list[str] | None = None) -> dict[str, object]:
    missing = [item for item in (must_include or []) if item not in text]
    forbidden = [item for item in FORBIDDEN_PHRASES if item in text]
    forbidden.extend([item for item in (must_not_include or []) if item in text])
    forbidden.extend(f"internal_field:{pattern.pattern}" for pattern in INTERNAL_FIELD_PATTERNS if pattern.search(text))
    return {"ok": not missing and not forbidden, "missing": missing, "forbidden": sorted(set(forbidden))}


def validate_codex_output(text: str, decision: dict[str, object]) -> dict[str, object]:
    result = validate_output(text, decision.get("must_include", []), decision.get("must_not_include", []))  # type: ignore[arg-type]
    if _looks_like_unsafe_command_recommendation(text):
        forbidden = set(result.get("forbidden", []))
        forbidden.add("unsafe_command_recommendation")
        result["forbidden"] = sorted(forbidden)
        result["ok"] = False
    confidence = float(decision.get("confidence", decision.get("decision_confidence", 0.0)) or 0.0)
    if confidence < 0.6 and any(word in text for word in ("certain", "must", "always")):
        result["ok"] = False
        result["confidence_warning"] = "overstated_low_confidence"
    return result


def _looks_like_unsafe_command_recommendation(text: str) -> bool:
    if not COMMAND_REFERENCE_PATTERN.search(text):
        return False

    safe_context = bool(SAFE_COMMAND_CONTEXT_PATTERN.search(text))
    if safe_context and not UNSAFE_COMMAND_INTENT_PATTERN.search(text):
        return False
    if APPROVAL_BYPASS_PATTERN.search(text):
        return True
    if UNSAFE_COMMAND_INTENT_PATTERN.search(text):
        return True
    if DIRECT_COMMAND_PATTERN.search(text) and not safe_context:
        return True
    return not safe_context
