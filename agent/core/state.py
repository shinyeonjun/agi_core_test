from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from typing import Any

from agent.config.defaults import ensure_runtime_dirs, now_kst, state_path

DEFAULT_STATE: dict[str, Any] = {
    "version": "0.1",
    "mode": "idle",
    "current_focus": "agent_core_v0_1",
    "last_user_interaction_at": None,
    "last_idle_tick_at": None,
    "autonomous_level": 2,
    "uncertainty_level": 0.25,
    "risk_level": 0.20,
    "memory_clutter": 0.30,
    "open_goal_pressure": 0.0,
    "system_health_pressure": 0.20,
    "user_alignment_pressure": 0.15,
    "renderer": {
        "type": "fallback",
        "available": True,
        "last_success_at": None,
        "failure_count": 0,
    },
    "limits": {
        "max_idle_actions_per_hour": 3,
        "max_codex_calls_per_hour": 0,
        "max_tool_runtime_seconds": 30,
    },
}


def _merge_defaults(state: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(defaults)
    for key, value in state.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_defaults(value, merged[key])
        else:
            merged[key] = value
    return merged


def load_state() -> dict[str, Any]:
    ensure_runtime_dirs()
    path = state_path()
    if not path.exists():
        state = deepcopy(DEFAULT_STATE)
        save_state(state)
        return state
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        broken = path.with_suffix(".broken.json")
        path.replace(broken)
        state = deepcopy(DEFAULT_STATE)
        state["mode"] = "degraded"
        state["degraded_reason"] = f"state_json_corrupted:{broken.name}"
        save_state(state)
        return state
    return _merge_defaults(state, DEFAULT_STATE)


def save_state(state: dict[str, Any]) -> None:
    ensure_runtime_dirs()
    path = state_path()
    fd, tmp_name = tempfile.mkstemp(prefix="state.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def set_mode(mode: str) -> dict[str, Any]:
    state = load_state()
    state["mode"] = mode
    save_state(state)
    return state


def mark_user_interaction() -> dict[str, Any]:
    state = load_state()
    state["mode"] = "interactive"
    state["last_user_interaction_at"] = now_kst()
    save_state(state)
    return state


def mark_idle_tick() -> dict[str, Any]:
    state = load_state()
    state["mode"] = "idle"
    state["last_idle_tick_at"] = now_kst()
    save_state(state)
    return state
