from __future__ import annotations

import re


class HumanReplySanitizerError(ValueError):
    pass


INTERNAL_TERMS = {
    "TaskSpec",
    "CoreResult",
    "Core",
    "trace",
    "NeuroKernel",
    "LanguageOrgan",
    "LanguageIntent",
    "JSON",
    "execution_result",
    "allowed_actions",
    "blocked_actions",
    "action_id",
    "task_id",
    "stdout",
    "stderr",
    "duration_ms",
    "error_type",
    "success_criteria",
    "get_memory_usage",
    "get_disk_usage",
    "get_uptime",
    "get_cpu_temp",
    "get_cpu_per_core_usage",
    "list_artifacts",
    "run_safe_benchmark",
}

_TOKEN_BOUNDARY_RE = r"(?<![A-Za-z0-9_]){term}(?![A-Za-z0-9_])"


def contains_internal_language(text: str) -> bool:
    if "```" in text or "{" in text or "}" in text:
        return True
    for term in INTERNAL_TERMS:
        pattern = _TOKEN_BOUNDARY_RE.format(term=re.escape(term))
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if term.lower() == "json" and match.start() > 0 and text[match.start() - 1] == ".":
                continue
            return True
    return False


def clean_human_reply(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        raise HumanReplySanitizerError("human reply is empty")
    if contains_internal_language(cleaned):
        raise HumanReplySanitizerError("human reply contains internal language")
    return cleaned[:1800]
