from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.codex_config import codex_exec_args, codex_exec_config
from agent.language.fallback_rule import FallbackRuleLanguageEngine
from agent.language.schemas import Interpretation, normalize_interpretation

LANGUAGE_INTERPRETER_PROMPT = """Classify the user message for Agent Core. Return JSON only.
Core, not you, decides policy, approval, goals, memory, and execution.
Never execute actions or request tools.

Targets: architecture, capabilities, status, help, greeting, question, response_style, idea, last_turn, task_note, project_spec, report, code_change, self_improvement.
Use architecture for Core structure questions. Use capabilities for what Core can do or cannot do.
Use code_change for requests to implement, fix, refactor, edit code, add tests, debug a repository, or change Agent Core itself.
Use intent self_improvement_request and target self_improvement when the user asks Core to inspect itself, improve itself, fix its own Core pipeline, or run a self-improvement cycle.
Set execution.requires_action=true only when the user is asking Core to do work later.
For code_change requests, set execution.suggested_queue_type="code_change".
For self_improvement_request, set execution.suggested_queue_type="self_improvement_code".
"""

LANGUAGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "sentiment",
        "target",
        "confidence",
        "style_update",
        "memory_instruction",
        "execution",
        "idea",
        "safety_notes",
    ],
    "properties": {
        "intent": {"type": "string", "enum": ["chat", "feedback", "style_feedback", "brainstorm", "task_request", "report_request", "project_request", "self_improvement_request", "approval", "control", "memory_instruction", "unknown"]},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed", "unknown"]},
        "target": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "style_update": {
            "type": "object",
            "additionalProperties": False,
            "required": ["feedback_type", "positive_signal", "tone", "structure", "detail_level", "emoji", "avoid", "prefer"],
            "properties": {
                "feedback_type": {"type": ["string", "null"]},
                "positive_signal": {"type": ["boolean", "null"]},
                "tone": {"type": ["string", "null"]},
                "structure": {"type": ["string", "null"]},
                "detail_level": {"type": ["string", "null"]},
                "emoji": {"type": ["boolean", "null"]},
                "avoid": {"type": ["array", "null"], "items": {"type": "string"}},
                "prefer": {"type": ["array", "null"], "items": {"type": "string"}},
            },
        },
        "memory_instruction": {"type": "boolean"},
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "required": ["requires_action", "suggested_queue_type", "risk_hint", "description", "request"],
            "properties": {
                "requires_action": {"type": "boolean"},
                "suggested_queue_type": {"type": ["string", "null"]},
                "risk_hint": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "request": {"type": ["string", "null"]},
            },
        },
        "idea": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "domain", "suggested_next_step"],
            "properties": {
                "summary": {"type": ["string", "null"]},
                "domain": {"type": ["array", "null"], "items": {"type": "string"}},
                "suggested_next_step": {"type": ["string", "null"]},
            },
        },
        "safety_notes": {"type": "array", "items": {"type": "string"}},
    },
}


class CodexLanguageEngine:
    name = "codex"

    def __init__(self, *, fallback: FallbackRuleLanguageEngine | None = None, timeout_seconds: int | None = None) -> None:
        self.fallback = fallback or FallbackRuleLanguageEngine()
        self.config = codex_exec_config("LANGUAGE", default_reasoning="low", default_timeout=20)
        self.timeout_seconds = timeout_seconds or self.config.timeout_seconds

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
        compact_context = {key: value for key, value in (context or {}).items() if key in {"surface", "purpose", "channel_role"}}
        prompt = "\n".join([
            LANGUAGE_INTERPRETER_PROMPT,
            f"Context: {json.dumps(compact_context, ensure_ascii=False)}",
            f"User: {text[:1000]}",
        ])
        try:
            output_path = Path(tempfile.gettempdir()) / f"agent_core_language_{os.getpid()}_{uuid4().hex}.json"
            schema_path = Path(tempfile.gettempdir()) / f"agent_core_language_schema_{os.getpid()}_{uuid4().hex}.json"
            if output_path.exists():
                output_path.unlink()
            schema_path.write_text(json.dumps(LANGUAGE_OUTPUT_SCHEMA, ensure_ascii=False), encoding="utf-8")
            args = [
                *codex_exec_args(self.config),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
                prompt,
            ]
            completed = subprocess.run(
                args,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                check=False,
            )
        except Exception as exc:
            if "output_path" in locals() and output_path.exists():
                output_path.unlink()
            if "schema_path" in locals() and schema_path.exists():
                schema_path.unlink()
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason=f"codex_error:{type(exc).__name__}")

        stdout = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else (completed.stdout or "").strip()
        if output_path.exists():
            output_path.unlink()
        if "schema_path" in locals() and schema_path.exists():
            schema_path.unlink()
        if completed.returncode != 0 or not stdout:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_empty_or_failed")
        parsed = self._parse_json_output(stdout)
        if parsed is None:
            fallback = self.fallback.interpret_user_message(text, context or {})
            return normalize_interpretation(fallback, engine=self.fallback.name, fallback_reason="codex_invalid_json")
        return normalize_interpretation(parsed, engine=self.name)
