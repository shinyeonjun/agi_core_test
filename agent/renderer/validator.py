from __future__ import annotations

import re

FORBIDDEN_PHRASES = (
    "auto sudo execution", "sudo auto execution", "rm -rf", "/etc auto edit", "/etc/",
    "SSH key access", "ssh key", "without approval", "consciousness emerged", "root permission handled", "AGI achieved",
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
    re.compile(r"`(?:selected_goal_id|user_goal_created|policy_summary|language_interpretation|renderer)`", re.IGNORECASE),
)


def validate_output(text: str, must_include: list[str] | None = None, must_not_include: list[str] | None = None) -> dict[str, object]:
    missing = [item for item in (must_include or []) if item not in text]
    forbidden = [item for item in FORBIDDEN_PHRASES if item in text]
    forbidden.extend([item for item in (must_not_include or []) if item in text])
    forbidden.extend(f"internal_field:{pattern.pattern}" for pattern in INTERNAL_FIELD_PATTERNS if pattern.search(text))
    return {"ok": not missing and not forbidden, "missing": missing, "forbidden": sorted(set(forbidden))}


def validate_codex_output(text: str, decision: dict[str, object]) -> dict[str, object]:
    result = validate_output(text, decision.get("must_include", []), decision.get("must_not_include", []))  # type: ignore[arg-type]
    if "sudo" in text.lower() or "apt install" in text.lower():
        forbidden = set(result.get("forbidden", []))
        forbidden.add("unsafe_command_recommendation")
        result["forbidden"] = sorted(forbidden)
        result["ok"] = False
    confidence = float(decision.get("confidence", decision.get("decision_confidence", 0.0)) or 0.0)
    if confidence < 0.6 and any(word in text for word in ("certain", "must", "always")):
        result["ok"] = False
        result["confidence_warning"] = "overstated_low_confidence"
    return result
