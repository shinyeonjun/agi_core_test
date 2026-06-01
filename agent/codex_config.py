from __future__ import annotations

import os
from dataclasses import dataclass

VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


@dataclass(frozen=True)
class CodexExecConfig:
    component: str
    model: str | None
    reasoning_effort: str | None
    timeout_seconds: int


def _env_name(component: str, suffix: str) -> str:
    return f"AGENT_CODEX_{component.upper()}_{suffix}"


def _env_value(component: str, suffix: str) -> str | None:
    component_value = os.getenv(_env_name(component, suffix))
    if component_value is not None and component_value.strip():
        return component_value.strip()
    global_value = os.getenv(f"AGENT_CODEX_{suffix}")
    if global_value is not None and global_value.strip():
        return global_value.strip()
    return None


def _timeout(component: str, default: int) -> int:
    raw = _env_value(component, "TIMEOUT")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(5, min(120, value))


def _reasoning_effort(component: str, default: str | None) -> str | None:
    raw = _env_value(component, "REASONING") or default
    if raw is None:
        return None
    value = raw.strip().lower()
    return value if value in VALID_REASONING_EFFORTS else default


def codex_exec_config(component: str, *, default_reasoning: str | None, default_timeout: int) -> CodexExecConfig:
    return CodexExecConfig(
        component=component,
        model=_env_value(component, "MODEL"),
        reasoning_effort=_reasoning_effort(component, default_reasoning),
        timeout_seconds=_timeout(component, default_timeout),
    )


def codex_exec_args(config: CodexExecConfig) -> list[str]:
    args = ["codex", "exec"]
    if config.model:
        args.extend(["--model", config.model])
    if config.reasoning_effort:
        args.extend(["-c", f"model_reasoning_effort={config.reasoning_effort}"])
    return args
