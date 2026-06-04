from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.codex_config import codex_exec_args, codex_exec_config
from agent.config.defaults import renderer_workspace, now_kst
from agent.core.database import connect, init_db
from agent.renderer.fallback_renderer import render as fallback_render
from agent.renderer.prompts import CODEX_RENDERER_PROMPT
from agent.renderer.validator import validate_codex_output

_ALLOWED_DECISION_KEYS = {
    "version", "kind", "created_at", "user_input", "selected_goal", "selected_goal_id",
    "drive_scores", "policy_summary", "core_judgment", "confidence", "decision_confidence",
    "risk_level", "metrics", "runtime_self_map", "decision_trace", "must_include", "must_not_include", "renderer_hint", "renderer",
    "language_interpretation", "style_profile", "style_directives", "style_feedback",
    "user_directed_goal", "user_goal_created", "capability_map", "self_report_context",
}
_ALLOWED_MEMORY_KEYS = {"id", "title", "memory_type", "importance", "confidence", "score", "tags_json"}
_ALLOWED_SKILL_KEYS = {"id", "name", "trigger", "confidence", "score", "tags_json"}
_DROPPED_KEYS = {
    "content", "metadata", "metadata_json", "payload", "proposed_payload_json",
    "raw_output", "tool_output", "source_event_id", "decision_json", "procedure_json",
}
_SECRET_KEY_RE = re.compile(r"(secret|token|api[_-]?key|authorization|password|private[_-]?key)", re.IGNORECASE)
_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"authorization:\s*bearer\s+[^\s,;]+", re.IGNORECASE),
    re.compile(r"(?m)^\s*[A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD)[A-Z0-9_]*\s*=.*$", re.IGNORECASE),
    re.compile(r"\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^\s,'\"]+", re.IGNORECASE),
    re.compile(r"\.env(?:\.[\w-]+)?", re.IGNORECASE),
]


def _redact_text(value: str, max_length: int = 500) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    if len(result) > max_length:
        result = result[:max_length] + "...[truncated]"
    return result


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value[:20]]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in _DROPPED_KEYS or _SECRET_KEY_RE.search(key_text):
                continue
            clean[key_text] = _sanitize_value(item)
        return clean
    return value


def _sanitize_rows(rows: Any, allowed_keys: set[str]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    clean_rows: list[dict[str, Any]] = []
    for row in rows[:10]:
        if not isinstance(row, dict):
            continue
        clean_rows.append({key: _sanitize_value(row[key]) for key in allowed_keys if key in row})
    return clean_rows


def sanitize_decision_for_renderer(decision: dict[str, Any]) -> dict[str, Any]:
    clean = {key: _sanitize_value(decision[key]) for key in _ALLOWED_DECISION_KEYS if key in decision}
    if "relevant_memories" in decision:
        clean["relevant_memories"] = _sanitize_rows(decision.get("relevant_memories"), _ALLOWED_MEMORY_KEYS)
    if "selected_memories" in decision:
        clean["selected_memories"] = _sanitize_rows(decision.get("selected_memories"), _ALLOWED_MEMORY_KEYS)
    if "relevant_skills" in decision:
        clean["relevant_skills"] = _sanitize_rows(decision.get("relevant_skills"), _ALLOWED_SKILL_KEYS)
    return clean



def _record_renderer_run(decision: dict[str, Any], rendered_text: str | None, success: bool, validation: dict[str, Any], duration_ms: int, error: str | None = None) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO renderer_runs (ts, decision_json, rendered_text, success, validation_result_json, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), json.dumps(decision, ensure_ascii=False), rendered_text, 1 if success else 0, json.dumps(validation, ensure_ascii=False), duration_ms, error),
        )
        conn.commit()
        return int(cur.lastrowid)


def render_with_codex(decision: dict[str, Any], timeout_seconds: int | None = None) -> str:
    start = time.monotonic()
    config = codex_exec_config("RENDERER", default_reasoning="low", default_timeout=20)
    workspace = renderer_workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    output_path = Path(tempfile.gettempdir()) / f"agent_core_chat_{uuid4().hex}.md"
    safe_decision = sanitize_decision_for_renderer(decision)
    prompt = CODEX_RENDERER_PROMPT + "\n\nDecision Object:\n" + json.dumps(safe_decision, ensure_ascii=False, separators=(",", ":"))

    try:
        if output_path.exists():
            output_path.unlink()
        args = [
            *codex_exec_args(config),
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--output-last-message",
            str(output_path),
            prompt,
        ]
        completed = subprocess.run(
            args,
            cwd=workspace,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_seconds or config.timeout_seconds,
            check=False,
        )
        text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else (completed.stdout or "").strip()
        if output_path.exists():
            output_path.unlink()
        if not text:
            raise RuntimeError(completed.stderr.strip() or "codex produced empty output")
        validation = validate_codex_output(text, safe_decision)
        duration_ms = int((time.monotonic() - start) * 1000)
        if not validation["ok"]:
            fallback = fallback_render(safe_decision)
            _record_renderer_run(safe_decision, fallback, False, validation, duration_ms, "validation_failed")
            return fallback
        _record_renderer_run(safe_decision, text, True, validation, duration_ms)
        return text
    except Exception as exc:
        if output_path.exists():
            output_path.unlink()
        fallback = fallback_render(safe_decision)
        duration_ms = int((time.monotonic() - start) * 1000)
        _record_renderer_run(safe_decision, fallback, False, {"ok": False, "fallback": True}, duration_ms, str(exc))
        return fallback
