from __future__ import annotations

from .schema import Action, ActionSpec


def find_action_spec(action: Action, specs: tuple[ActionSpec, ...]) -> ActionSpec | None:
    for spec in specs:
        if action.name == spec.name:
            return spec
    return None


def is_action_allowed(action: Action, specs: tuple[ActionSpec, ...]) -> bool:
    spec = find_action_spec(action, specs)
    if spec is None:
        return False
    return all(key in action.params for key in spec.required_params)


def allowed_action_names(specs: tuple[ActionSpec, ...]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)
