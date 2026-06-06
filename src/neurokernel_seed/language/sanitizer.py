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
    "list_artifacts",
    "run_safe_benchmark",
}


def contains_internal_language(text: str) -> bool:
    if "```" in text or "{" in text or "}" in text:
        return True
    lowered = text.lower()
    for term in INTERNAL_TERMS:
        if term.lower() in lowered:
            return True
    return bool(re.search(r"\b[a-z]+_[a-z0-9_]+\b", text))


def clean_human_reply(text: str) -> str:
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        raise HumanReplySanitizerError("human reply is empty")
    if contains_internal_language(cleaned):
        raise HumanReplySanitizerError("human reply contains internal language")
    return cleaned[:1800]
