from __future__ import annotations

import os
from typing import Any

from agent.renderer.codex_renderer import render_with_codex
from agent.renderer.fallback_renderer import render as fallback_render


def selected_renderer_name() -> str:
    return os.getenv("AGENT_CHAT_RENDERER", "codex").strip().lower() or "codex"


def render_response(decision: dict[str, Any]) -> str:
    mode = selected_renderer_name()
    if mode in {"codex", "codex_cli", "chat_codex"}:
        return render_with_codex(decision)
    return fallback_render(decision)
