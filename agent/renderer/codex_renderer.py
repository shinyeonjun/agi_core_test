from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from agent.config.defaults import renderer_workspace, now_kst
from agent.core.database import connect, init_db
from agent.renderer.fallback_renderer import render as fallback_render
from agent.renderer.prompts import CODEX_RENDERER_PROMPT
from agent.renderer.validator import validate_codex_output


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


def render_with_codex(decision: dict[str, Any], timeout_seconds: int = 30) -> str:
    start = time.monotonic()
    workspace = renderer_workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    input_path = workspace / "input.json"
    prompt_path = workspace / "prompt.txt"
    output_path = workspace / "output.md"
    input_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(CODEX_RENDERER_PROMPT + "\n\nDecision Object:\n" + input_path.read_text(encoding="utf-8"), encoding="utf-8")

    try:
        completed = subprocess.run(
            ["codex", "exec", "--sandbox", "read-only", "--ask-for-approval", "never", prompt_path.read_text(encoding="utf-8")],
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        text = (completed.stdout or "").strip()
        if not text:
            raise RuntimeError(completed.stderr.strip() or "codex produced empty output")
        output_path.write_text(text, encoding="utf-8")
        validation = validate_codex_output(text, decision)
        duration_ms = int((time.monotonic() - start) * 1000)
        if not validation["ok"]:
            fallback = fallback_render(decision)
            _record_renderer_run(decision, fallback, False, validation, duration_ms, "validation_failed")
            return fallback
        _record_renderer_run(decision, text, True, validation, duration_ms)
        return text
    except Exception as exc:
        fallback = fallback_render(decision)
        duration_ms = int((time.monotonic() - start) * 1000)
        _record_renderer_run(decision, fallback, False, {"ok": False, "fallback": True}, duration_ms, str(exc))
        return fallback
