from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from agent.language.fallback_rule import FallbackRuleLanguageEngine
from agent.language.schemas import Interpretation, normalize_interpretation

LANGUAGE_INTERPRETER_PROMPT = """You are the language interpretation layer for Agent Core.
Return JSON only. Do not execute actions. Do not request tools. Do not modify files.
Core will make final policy, approval, goal, memory, and execution decisions.

Allowed intents:
chat, feedback, style_feedback, brainstorm, task_request, report_request, project_request,
approval, control, memory_instruction, unknown.

Required JSON fields:
intent, sentiment, target, confidence, style_update, memory_instruction, execution, idea, safety_notes.

Rules:
- execution.requires_action may describe a request, but never approve execution.
- policy, risk, approval, and full_device_lab decisions belong to Core.
- If uncertain, use intent=unknown and low confidence.
"""


class CodexLanguageEngine:
    name = "codex"

    def __init__(self, *, fallback: FallbackRuleLanguageEngine | None = None, timeout_seconds: int | None = None) -> None:
        self.fallback = fallback or FallbackRuleLanguageEngine()
        self.timeout_seconds = timeout_seconds or int(os.getenv("AGENT_LANGUAGE_CODEX_TIMEOUT", "20"))

    def interpret_user_message(self, text: str, context: dict[str, Any] | None = None) -> Interpretation:
        prompt = "\n".join([
            LANGUAGE_INTERPRETER_PROMPT,
            "",
            "Context JSON:",
            json.dumps(context or {}, ensure_ascii=False),
            "",
            "User message:",
            text,
        ])
        try:
            completed = subprocess.run(
                ["codex", "exec", "--sandbox", "read-only", "--ask-for-approval", "never", prompt],
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except Exception as exc:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason=f"codex_error:{type(exc).__name__}")

        stdout = (completed.stdout or "").strip()
        if completed.returncode != 0 or not stdout:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_empty_or_failed")
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_invalid_json")
        return normalize_interpretation(parsed, engine=self.name)
