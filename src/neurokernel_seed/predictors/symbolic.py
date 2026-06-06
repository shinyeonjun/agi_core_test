from __future__ import annotations

from neurokernel_seed.core.schema import Action, Prediction, WorldState
from .base import BasePredictor


class SymbolicPredictor(BasePredictor):
    name = "symbolic"

    def predict(self, state: WorldState, action: Action) -> Prediction:
        env = state.env_name.split(".")[0]
        if env == "lock":
            return self._lock(state, action)
        if env == "tool":
            return self._tool(state, action)
        if env == "maze":
            return self._maze(state, action)
        raise ValueError(f"unsupported symbolic predictor env: {state.env_name}")

    def _lock(self, state: WorldState, action: Action) -> Prediction:
        code = tuple(state.facts["code"])
        max_steps = int(state.facts["max_steps"])
        progress = int(state.facts["progress"])
        steps = int(state.facts["steps"]) + 1
        trap_armed = bool(state.facts.get("trap_armed", False))
        if action.name == "inspect":
            reward = 0.1 if trap_armed else (0.05 if not bool(state.facts.get("current_slot_known", True)) else -0.01)
            information_gain = 1.0 if trap_armed or reward > 0 else 0.0
            terminal = steps >= max_steps
            next_vector = _pad_vector((state.vector[0], steps / max_steps, 1.0 if terminal else 0.0, 0.0 if trap_armed else float(state.vector[-1])), len(state.vector))
            return Prediction(next_vector, reward, terminal, 0.8, "symbolic lock inspect", progress_delta=0.0, information_gain=information_gain)
        if trap_armed and action.name == "press" and int(action.params.get("digit", -1)) == code[progress]:
            progress = 0
            reward = -0.35
        elif action.name == "press" and int(action.params.get("digit", -1)) == code[progress]:
            progress += 1
            reward = 1.0 if progress >= len(code) else 0.2
        elif action.name == "reset":
            progress = 0
            reward = -0.01
        else:
            progress = 0
            reward = -0.2
        terminal = progress >= len(code) or steps >= max_steps
        progress_delta = progress / len(code) - float(state.vector[0])
        next_vector = _pad_vector((progress / len(code), steps / max_steps, 1.0 if terminal else 0.0, 1.0 if trap_armed else 0.0), len(state.vector))
        return Prediction(next_vector, reward, terminal, 0.9, "symbolic lock transition", progress_delta=progress_delta)

    def _tool(self, state: WorldState, action: Action) -> Prediction:
        sequence = tuple(state.facts["sequence"])
        max_steps = int(state.facts["max_steps"])
        stage = int(state.facts["stage"])
        steps = int(state.facts["steps"]) + 1
        expected = sequence[min(stage, len(sequence) - 1)]
        precondition_ready = bool(state.facts.get("precondition_ready", True))
        if "precondition_ready" in state.facts and action.name == "search" and not precondition_ready:
            next_vector = _pad_vector((stage / len(sequence), steps / max_steps, 1.0 if steps >= max_steps else 0.0, 1.0), len(state.vector))
            return Prediction(next_vector, 0.12, steps >= max_steps, 0.85, "symbolic tool precondition setup", progress_delta=0.0, information_gain=1.0)
        if "precondition_ready" in state.facts and action.name == expected and not precondition_ready:
            next_vector = _pad_vector((stage / len(sequence), steps / max_steps, 1.0 if steps >= max_steps else 0.0, 0.0), len(state.vector))
            return Prediction(next_vector, -0.3, steps >= max_steps, 0.85, "symbolic tool missing precondition", progress_delta=0.0)
        if not bool(state.facts.get("current_slot_known", True)) and action.name in {"search", "read"}:
            reward = 0.05
            terminal = steps >= max_steps
            return Prediction(state.vector[:1] + (steps / max_steps, 1.0 if terminal else 0.0), reward, terminal, 0.8, "symbolic tool reveal", information_gain=1.0)
        if action.name == expected:
            stage += 1
            reward = 1.0 if stage >= len(sequence) else 0.25
        else:
            stage = max(0, stage - 1)
            reward = -0.1
        terminal = stage >= len(sequence) or steps >= max_steps
        next_vector = _pad_vector((stage / len(sequence), steps / max_steps, 1.0 if terminal else 0.0, 1.0 if precondition_ready else 0.0), len(state.vector))
        return Prediction(next_vector, reward, terminal, 0.85, "symbolic tool transition")

    def _maze(self, state: WorldState, action: Action) -> Prediction:
        saw_hint = bool(state.facts["saw_hint"])
        escaped = bool(state.facts["escaped"])
        max_steps = int(state.facts["max_steps"])
        steps = int(state.facts["steps"]) + 1
        target = str(state.facts["target_color"])
        has_hazard_state = "hazard_active" in state.facts
        hazard_active = bool(state.facts.get("hazard_active", False))
        if action.name == "inspect":
            saw_hint = True
            reward = 0.12 if hazard_active else 0.1
            hazard_active = False
        elif action.name == "move" and saw_hint and action.params.get("door") == target and not hazard_active:
            escaped = True
            reward = 1.0
        elif action.name == "move" and saw_hint and action.params.get("door") == target and hazard_active:
            reward = -0.35
        else:
            reward = -0.25
        terminal = escaped or steps >= max_steps
        status_value = 1.0 if hazard_active else 0.0
        if not has_hazard_state:
            status_value = 1.0 if terminal else 0.0
        next_vector = (1.0 if saw_hint else 0.0, 1.0 if escaped else 0.0, steps / max_steps, status_value)
        return Prediction(next_vector, reward, terminal, 0.85, "symbolic maze transition", information_gain=1.0 if action.name == "inspect" and (not bool(state.facts.get("saw_hint")) or bool(state.facts.get("hazard_active", False))) else 0.0)


def _pad_vector(values: tuple[float, ...], size: int) -> tuple[float, ...]:
    if len(values) >= size:
        return tuple(values[:size])
    return tuple(values) + (0.0,) * (size - len(values))
