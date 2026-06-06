from __future__ import annotations

from neurokernel_seed.core.schema import Action, ActionSpec, Observation, TransitionResult, WorldState
from neurokernel_seed.core.validation import validate_action
from .base import MicroWorld


class MemoryMaze(MicroWorld):
    doors = ("red", "blue", "green")

    def __init__(
        self,
        name: str = "maze.train",
        split: str = "train",
        target_color: str = "blue",
        max_steps: int = 7,
        doors: tuple[str, ...] | None = None,
        visibility_mode: str = "partial_hidden",
    ):
        self.name = name
        self.split = split
        self.doors = tuple(doors or self.doors)
        self.target_color = target_color
        self.max_steps = max_steps
        self.visibility_mode = visibility_mode
        self.action_specs = (
            ActionSpec("inspect", reveals_information=True, consumes_progress_step=False),
            ActionSpec("move", ("door",), {"door": str}, {"door": self.doors}, requires_known_slot=True, terminal_only=True),
        )
        self._state = self._make_state(False, False, 0, False)

    @property
    def metadata(self) -> dict:
        data = super().metadata
        data["doors"] = list(self.doors)
        data["target_color"] = self.target_color
        data["visibility_mode"] = self.visibility_mode
        return data

    def reset(self, seed: int = 0) -> WorldState:
        self._state = self._make_state(self.visibility_mode == "visible", False, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        saw_hint = bool(prev.facts["saw_hint"])
        escaped = bool(prev.facts["escaped"])
        steps = int(prev.facts["steps"]) + 1
        reward = -0.02
        if action.name == "inspect":
            information_gain = 0.0 if saw_hint else 1.0
            saw_hint = True
            reward = 0.1
        elif action.name == "move":
            information_gain = 0.0
            door = str(action.params["door"])
            if saw_hint and door == self.target_color:
                escaped = True
                reward = 1.0
            else:
                reward = -0.25
        else:
            information_gain = 0.0
        done = escaped or steps >= self.max_steps
        self._state = self._make_state(saw_hint, escaped, steps, done)
        obs = Observation(f"hint={self.target_color if saw_hint else 'hidden'}, escaped={escaped}", self._state.vector, dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": escaped, "information_gain": information_gain})

    def snapshot(self) -> dict:
        return {
            "env_name": self.name,
            "split": self.split,
            "target_color": self.target_color,
            "doors": list(self.doors),
            "max_steps": self.max_steps,
            "visibility_mode": self.visibility_mode,
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
        }

    def restore(self, snapshot: dict) -> WorldState:
        state = snapshot["state"]
        self.target_color = str(snapshot.get("target_color", state.get("target_color", self.target_color)))
        self.doors = tuple(str(value) for value in snapshot.get("doors", state.get("doors", self.doors)))
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.visibility_mode = str(snapshot.get("visibility_mode", state.get("visibility_mode", self.visibility_mode)))
        self.action_specs = (
            ActionSpec("inspect", reveals_information=True, consumes_progress_step=False),
            ActionSpec("move", ("door",), {"door": str}, {"door": self.doors}, requires_known_slot=True, terminal_only=True),
        )
        self._state = self._make_state(bool(state["saw_hint"]), bool(state["escaped"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        return [Action("inspect", source="candidate", confidence=0.7)] + [Action("move", {"door": color}, source="candidate", confidence=0.7) for color in self.doors]

    def expert_action(self, state: WorldState) -> Action:
        if not bool(state.facts["saw_hint"]):
            return Action("inspect", source="expert")
        return Action("move", {"door": self.target_color}, source="expert")

    def success(self, state: WorldState) -> bool:
        return bool(state.facts["escaped"])

    def _make_state(self, saw_hint: bool, escaped: bool, steps: int, terminal: bool) -> WorldState:
        known_mask = (1,) if saw_hint else (0,)
        unknown_mask = (0,) if saw_hint else (1,)
        remaining_steps = self.max_steps - steps
        return WorldState(
            self.name,
            f"hint{int(saw_hint)}:esc{int(escaped)}:s{steps}",
            (1.0 if saw_hint else 0.0, 1.0 if escaped else 0.0, steps / self.max_steps, 1.0 if terminal else 0.0),
            {
                "saw_hint": saw_hint,
                "escaped": escaped,
                "target_color": self.target_color,
                "visible_target_color": self.target_color if saw_hint else "",
                "doors": self.doors,
                "known_mask": known_mask,
                "unknown_mask": unknown_mask,
                "revealed_slots_count": 1 if saw_hint else 0,
                "unknown_slots_count": 0 if saw_hint else 1,
                "current_slot_known": saw_hint,
                "last_observation": {"type": "inspect" if saw_hint else "none", "slot": 0 if saw_hint else -1, "value": self.target_color if saw_hint else ""},
                "last_observation_type": "inspect" if saw_hint else "none",
                "last_observation_slot": 0 if saw_hint else -1,
                "last_observation_value": self.target_color if saw_hint else "",
                "current_index": 0,
                "sequence_length": 1,
                "remaining_steps": remaining_steps,
                "remaining_sequence_slots": 0 if saw_hint else 1,
                "progress_fraction": 1.0 if escaped else (0.3 if saw_hint else 0.0),
                "is_last_step_index": saw_hint,
                "can_finish": saw_hint,
                "can_summarize": False,
                "visibility_mode": "visible" if saw_hint else self.visibility_mode,
                "steps": steps,
                "max_steps": self.max_steps,
                "split": self.split,
            },
            terminal,
        )
