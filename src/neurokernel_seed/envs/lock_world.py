from __future__ import annotations

from neurokernel_seed.core.schema import Action, ActionSpec, Observation, TransitionResult, WorldState
from neurokernel_seed.core.validation import validate_action
from .base import MicroWorld


class LockWorld(MicroWorld):
    def __init__(
        self,
        name: str = "lock.train",
        split: str = "train",
        code: tuple[int, ...] = (1, 2, 3),
        max_steps: int = 6,
        visibility_mode: str = "visible",
    ):
        self.name = name
        self.split = split
        self.code = code
        self.max_steps = max_steps
        self.visibility_mode = visibility_mode
        self._known_mask = self._initial_known_mask()
        self._last_observation = {"type": "none", "slot": -1, "value": 0}
        self.action_specs = (
            ActionSpec("inspect", reveals_information=True, consumes_progress_step=False),
            ActionSpec("press", ("digit",), {"digit": int}, requires_known_slot=True),
            ActionSpec("reset", consumes_progress_step=False),
        )
        self._state = self._make_state(0, 0, False)

    @property
    def metadata(self) -> dict:
        data = super().metadata
        data["code_length"] = len(self.code)
        data["allowed_digits"] = sorted(set(self.code) | set(range(1, 5)))
        data["visibility_mode"] = self.visibility_mode
        return data

    def reset(self, seed: int = 0) -> WorldState:
        self._known_mask = self._initial_known_mask()
        self._last_observation = {"type": "none", "slot": -1, "value": 0}
        self._state = self._make_state(0, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        progress = int(prev.facts["progress"])
        steps = int(prev.facts["steps"]) + 1
        reward = -0.01
        information_gain = 0.0
        if action.name == "inspect":
            slot = min(progress, len(self.code) - 1)
            if not self._known_mask[slot]:
                self._known_mask[slot] = True
                information_gain = 1.0
                reward = 0.05
            self._last_observation = {"type": "inspect", "slot": slot, "value": self.code[slot]}
        elif action.name == "reset":
            progress = 0
        elif action.name == "press":
            digit = int(action.params["digit"])
            if digit == self.code[progress]:
                progress += 1
                reward = 0.2
            else:
                progress = 0
                reward = -0.2
        done = progress >= len(self.code) or steps >= self.max_steps
        success = progress >= len(self.code)
        if success:
            reward = 1.0
        self._state = self._make_state(progress, steps, done)
        obs = Observation(text=f"progress={progress}, success={success}", vector=self._state.vector, facts=dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": success, "information_gain": information_gain})

    def snapshot(self) -> dict:
        return {
            "env_name": self.name,
            "split": self.split,
            "code": list(self.code),
            "max_steps": self.max_steps,
            "visibility_mode": self.visibility_mode,
            "known_mask": list(self._known_mask),
            "last_observation": dict(self._last_observation),
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
        }

    def restore(self, snapshot: dict) -> WorldState:
        state = snapshot["state"]
        self.code = tuple(int(value) for value in snapshot.get("code", state.get("code", self.code)))
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.visibility_mode = str(snapshot.get("visibility_mode", state.get("visibility_mode", self.visibility_mode)))
        known_mask = snapshot.get("known_mask", state.get("known_mask"))
        self._known_mask = [bool(value) for value in known_mask] if known_mask is not None else self._initial_known_mask()
        self._last_observation = dict(snapshot.get("last_observation", state.get("last_observation", {"type": "none", "slot": -1, "value": 0})))
        self._state = self._make_state(int(state["progress"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        actions = [Action("press", {"digit": d}, source="candidate", confidence=0.8) for d in range(1, 5)] + [Action("reset", source="candidate", confidence=0.3)]
        if not bool(state.facts.get("current_slot_known", True)):
            return [Action("inspect", source="candidate", confidence=0.9)] + actions
        return actions

    def expert_action(self, state: WorldState) -> Action:
        progress = int(state.facts["progress"])
        if progress >= len(self.code):
            return Action("reset", source="expert")
        if not bool(state.facts.get("current_slot_known", True)):
            return Action("inspect", source="expert")
        return Action("press", {"digit": self.code[progress]}, source="expert")

    def success(self, state: WorldState) -> bool:
        return int(state.facts["progress"]) >= len(self.code)

    def _initial_known_mask(self) -> list[bool]:
        if self.visibility_mode == "visible":
            return [True] * len(self.code)
        if self.visibility_mode == "hidden":
            return [False] * len(self.code)
        if not self.code:
            return []
        midpoint = max(1, len(self.code) // 2)
        return [index < midpoint for index in range(len(self.code))]

    def _make_state(self, progress: int, steps: int, terminal: bool) -> WorldState:
        known_mask = list(self._known_mask)
        if len(known_mask) != len(self.code):
            known_mask = self._initial_known_mask()
            self._known_mask = known_mask
        visible_code = tuple(self.code[index] if known_mask[index] else 0 for index in range(len(self.code)))
        unknown_mask = [not known for known in known_mask]
        current_slot_known = progress >= len(known_mask) or bool(known_mask[progress])
        remaining_steps = self.max_steps - steps
        facts = {
            "progress": progress,
            "steps": steps,
            "code": self.code,
            "visible_code": visible_code,
            "known_mask": tuple(1 if value else 0 for value in known_mask),
            "unknown_mask": tuple(1 if value else 0 for value in unknown_mask),
            "revealed_slots_count": sum(1 for value in known_mask if value),
            "unknown_slots_count": sum(1 for value in unknown_mask if value),
            "current_slot_known": current_slot_known,
            "last_observation": dict(self._last_observation),
            "last_observation_type": str(self._last_observation.get("type", "none")),
            "last_observation_slot": int(self._last_observation.get("slot", -1)),
            "last_observation_value": int(self._last_observation.get("value", 0)),
            "current_index": progress,
            "sequence_length": len(self.code),
            "remaining_steps": remaining_steps,
            "remaining_sequence_slots": max(0, len(self.code) - progress),
            "progress_fraction": progress / max(1, len(self.code)),
            "is_last_step_index": progress == len(self.code) - 1,
            "can_finish": current_slot_known and progress == len(self.code) - 1,
            "can_summarize": False,
            "visibility_mode": self.visibility_mode,
            "max_steps": self.max_steps,
            "split": self.split,
        }
        return WorldState(
            env_name=self.name,
            state_id=f"p{progress}:s{steps}:k{''.join(str(int(value)) for value in known_mask)}",
            vector=(progress / len(self.code), steps / self.max_steps, 1.0 if terminal else 0.0),
            facts=facts,
            terminal=terminal,
        )
