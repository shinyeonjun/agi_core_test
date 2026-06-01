from __future__ import annotations

FORBIDDEN_PHRASES = (
    "auto sudo execution", "sudo auto execution", "rm -rf", "/etc auto edit", "/etc/",
    "SSH key access", "ssh key", "without approval", "consciousness emerged", "root permission handled", "AGI achieved",
)


def validate_output(text: str, must_include: list[str] | None = None, must_not_include: list[str] | None = None) -> dict[str, object]:
    missing = [item for item in (must_include or []) if item not in text]
    forbidden = [item for item in FORBIDDEN_PHRASES if item in text]
    forbidden.extend([item for item in (must_not_include or []) if item in text])
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
