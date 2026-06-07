from __future__ import annotations

from typing import Literal

TaskState = Literal[
    "created",
    "validated",
    "waiting_approval",
    "ready",
    "running",
    "deciding",
    "executing",
    "evaluating",
    "completed",
    "failed",
    "cancelled",
]


ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "created": {"validated", "failed", "cancelled"},
    "validated": {"ready", "waiting_approval", "failed", "cancelled"},
    "waiting_approval": {"ready", "failed", "cancelled"},
    "ready": {"running", "failed", "cancelled"},
    "running": {"deciding", "executing", "failed", "cancelled"},
    "deciding": {"waiting_approval", "executing", "failed", "cancelled"},
    "executing": {"evaluating", "failed", "cancelled"},
    "evaluating": {"completed", "failed", "deciding", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


class StateTransitionError(ValueError):
    pass


def assert_transition(current: str, next_state: str) -> None:
    if current not in ALLOWED_TRANSITIONS:
        raise StateTransitionError(f"unknown current state: {current}")
    if next_state not in ALLOWED_TRANSITIONS[current]:
        raise StateTransitionError(f"invalid state transition: {current} -> {next_state}")
