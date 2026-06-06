from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NumberVector = tuple[float, ...]
ParamType = type | tuple[type, ...]


@dataclass(frozen=True)
class ActionSpec:
    name: str
    required_params: tuple[str, ...] = ()
    param_types: dict[str, ParamType] = field(default_factory=dict)
    allowed_values: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    reveals_information: bool = False
    requires_known_slot: bool = False
    terminal_only: bool = False
    consumes_progress_step: bool = True
    role: str = "execution"


@dataclass(frozen=True)
class Action:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "agent"
    confidence: float = 1.0


@dataclass(frozen=True)
class WorldState:
    env_name: str
    state_id: str
    vector: NumberVector
    facts: dict[str, Any]
    terminal: bool = False


@dataclass(frozen=True)
class Observation:
    text: str
    vector: NumberVector
    facts: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    next_state_vector: NumberVector
    reward: float = 0.0
    terminal: bool = False
    confidence: float = 0.0
    reason: str = ""
    success_probability: float | None = None
    progress_delta: float | None = None
    information_gain: float | None = None


@dataclass(frozen=True)
class TransitionResult:
    prev_state: WorldState
    action: Action
    observation: Observation
    next_state: WorldState
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeEvent:
    episode_id: str
    env_name: str
    step_index: int
    transition: TransitionResult
    prediction: Prediction | None = None


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    env_name: str
    agent_name: str
    seed: int
    steps: int
    total_reward: float
    success: bool
    done: bool
    prediction_error: float | None = None
