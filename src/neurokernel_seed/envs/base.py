from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from neurokernel_seed.core.schema import Action, ActionSpec, TransitionResult, WorldState


class MicroWorld(ABC):
    name: str
    split: str
    max_steps: int
    action_specs: tuple[ActionSpec, ...]

    @abstractmethod
    def reset(self, seed: int = 0) -> WorldState:
        ...

    @abstractmethod
    def step(self, action: Action) -> TransitionResult:
        ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> WorldState:
        ...

    @abstractmethod
    def candidate_actions(self, state: WorldState) -> list[Action]:
        ...

    @abstractmethod
    def expert_action(self, state: WorldState) -> Action:
        ...

    @abstractmethod
    def success(self, state: WorldState) -> bool:
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        state = getattr(self, "_state", None)
        return {
            "name": self.name,
            "split": self.split,
            "max_steps": self.max_steps,
            "vector_size": len(state.vector) if state else None,
            "action_specs": [_action_spec_metadata(spec) for spec in self.action_specs],
        }


def _action_spec_metadata(spec: ActionSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "required_params": list(spec.required_params),
        "param_types": {key: _type_name(value) for key, value in spec.param_types.items()},
        "allowed_values": {key: list(value) for key, value in spec.allowed_values.items()},
        "reveals_information": spec.reveals_information,
        "requires_known_slot": spec.requires_known_slot,
        "terminal_only": spec.terminal_only,
        "consumes_progress_step": spec.consumes_progress_step,
    }


def _type_name(value: type | tuple[type, ...]) -> str:
    values = value if isinstance(value, tuple) else (value,)
    return "|".join(item.__name__ for item in values)
