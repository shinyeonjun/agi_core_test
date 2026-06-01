from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

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
- Use target=architecture for questions about Core structure, composition, or how it is built.
- Use target=capabilities for questions about what Core can do, limitations, or current ability.
- Use target=status/help/greeting/question when those are the best fit.
- If uncertain, use intent=unknown and low confidence.
"""


class CodexLanguageEngine:
    name = "codex"

    def __init__(self, *, fallback: FallbackRuleLanguageEngine | None = None, timeout_seconds: int | None = None) -> None:
        self.fallback = fallback or FallbackRuleLanguageEngine()
        self.timeout_seconds = timeout_seconds or int(os.getenv("AGENT_LANGUAGE_CODEX_TIMEOUT", "20"))

    def _parse_json_output(self, stdout: str) -> dict[str, Any] | None:
        text = stdout.strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

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
            output_path = Path(tempfile.gettempdir()) / f"agent_core_language_{os.getpid()}_{uuid4().hex}.json"
            if output_path.exists():
                output_path.unlink()
            completed = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--output-last-message",
                    str(output_path),
                    prompt,
                ],
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except Exception as exc:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason=f"codex_error:{type(exc).__name__}")

        stdout = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else (completed.stdout or "").strip()
        if output_path.exists():
            output_path.unlink()
        if completed.returncode != 0 or not stdout:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_empty_or_failed")
        parsed = self._parse_json_output(stdout)
        if parsed is None:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_invalid_json")
        return normalize_interpretation(parsed, engine=self.name)
