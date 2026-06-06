from __future__ import annotations

import math
from typing import Any

from .actions import find_action_spec
from .schema import Action, ActionSpec, Observation, Prediction, TransitionResult, WorldState


class ContractError(ValueError):
    pass


def validate_vector(vector: tuple[float, ...], *, name: str) -> None:
    if not vector:
        raise ContractError(f"{name} must not be empty")
    for value in vector:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError(f"{name} contains non-finite value: {value!r}")


def validate_action(action: Action, specs: tuple[ActionSpec, ...]) -> None:
    if not action.name:
        raise ContractError("action.name must not be empty")
    if not 0.0 <= float(action.confidence) <= 1.0:
        raise ContractError("action.confidence must be between 0 and 1")
    spec = find_action_spec(action, specs)
    if spec is None:
        raise ContractError(f"action is not allowed: {action.name}")
    missing = [key for key in spec.required_params if key not in action.params]
    if missing:
        raise ContractError(f"action {action.name} missing required params: {missing}")
    known_params = set(spec.required_params) | set(spec.param_types) | set(spec.allowed_values)
    unknown = sorted(set(action.params) - known_params)
    if unknown:
        raise ContractError(f"action {action.name} has unknown params: {unknown}")
    for key, expected_type in spec.param_types.items():
        if key in action.params and not _matches_type(action.params[key], expected_type):
            raise ContractError(f"action {action.name}.{key} must be {_type_name(expected_type)}")
    for key, allowed in spec.allowed_values.items():
        if key in action.params and action.params[key] not in allowed:
            raise ContractError(f"action {action.name}.{key} must be one of {list(allowed)}")


def validate_execution_sequence(sequence: tuple[str, ...], specs: tuple[ActionSpec, ...]) -> None:
    """검증용: setup/info/reset 액션이 실행 sequence에 섞이는 것을 막는다."""
    by_name = {spec.name: spec for spec in specs}
    for index, action_name in enumerate(sequence):
        spec = by_name.get(action_name)
        if spec is None:
            raise ContractError(f"execution sequence contains unknown action: {action_name}")
        if spec.role in {"setup", "information", "reset"}:
            raise ContractError(f"execution sequence must not contain {spec.role} action: {action_name}")
        if spec.terminal_only and index != len(sequence) - 1:
            raise ContractError(f"terminal action must appear only at the final sequence slot: {action_name}")
        if spec.role == "terminal" and index != len(sequence) - 1:
            raise ContractError(f"terminal role must appear only at the final sequence slot: {action_name}")


def validate_state(state: WorldState) -> None:
    if not state.env_name or not state.state_id:
        raise ContractError("state must include env_name and state_id")
    validate_vector(state.vector, name="state.vector")


def validate_observation(observation: Observation, expected_len: int) -> None:
    validate_vector(observation.vector, name="observation.vector")
    if len(observation.vector) != expected_len:
        raise ContractError("observation vector length mismatch")


def validate_prediction(prediction: Prediction, expected_len: int) -> None:
    validate_vector(prediction.next_state_vector, name="prediction.next_state_vector")
    if len(prediction.next_state_vector) != expected_len:
        raise ContractError("prediction vector length mismatch")
    if not 0.0 <= float(prediction.confidence) <= 1.0:
        raise ContractError("prediction.confidence must be between 0 and 1")
    if prediction.success_probability is not None and not 0.0 <= float(prediction.success_probability) <= 1.0:
        raise ContractError("prediction.success_probability must be between 0 and 1")
    if prediction.information_gain is not None and not 0.0 <= float(prediction.information_gain) <= 1.0:
        raise ContractError("prediction.information_gain must be between 0 and 1")


def validate_transition(result: TransitionResult, specs: tuple[ActionSpec, ...]) -> None:
    validate_state(result.prev_state)
    validate_action(result.action, specs)
    validate_state(result.next_state)
    validate_observation(result.observation, len(result.next_state.vector))
    if len(result.prev_state.vector) != len(result.next_state.vector):
        raise ContractError("state vector length changed across transition")
    if result.done != result.next_state.terminal:
        raise ContractError("done flag must match next_state.terminal")


def _matches_type(value: Any, expected_type: type | tuple[type, ...]) -> bool:
    expected = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    for item in expected:
        if item is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return True
        elif item is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
        elif isinstance(value, item):
            return True
    return False


def _type_name(expected_type: type | tuple[type, ...]) -> str:
    expected = expected_type if isinstance(expected_type, tuple) else (expected_type,)
    return "|".join(item.__name__ for item in expected)
