from __future__ import annotations

from neurokernel_seed.core.schema import Action, ActionSpec, Observation, TransitionResult, WorldState
from neurokernel_seed.core.validation import validate_action
from .base import MicroWorld


class ToolWorld(MicroWorld):
    sequence = ("search", "read", "summarize")

    def __init__(
        self,
        name: str = "tool.train",
        split: str = "train",
        max_steps: int = 5,
        sequence: tuple[str, ...] | None = None,
        visibility_mode: str = "visible",
    ):
        self.name = name
        self.split = split
        self.max_steps = max_steps
        self.sequence = tuple(sequence or self.sequence)
        self._validate_sequence_contract()
        self.visibility_mode = visibility_mode
        self._known_mask = self._initial_known_mask()
        self._last_observation = {"type": "none", "slot": -1, "value": ""}
        self.action_specs = self._make_action_specs()
        self._state = self._make_state(0, 0, False)

    @property
    def metadata(self) -> dict:
        data = super().metadata
        data["sequence"] = list(self.sequence)
        data["visibility_mode"] = self.visibility_mode
        return data

    def reset(self, seed: int = 0) -> WorldState:
        self._known_mask = self._initial_known_mask()
        self._last_observation = {"type": "none", "slot": -1, "value": ""}
        self._state = self._make_state(0, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        stage = int(prev.facts["stage"])
        steps = int(prev.facts["steps"]) + 1
        expected = self.sequence[min(stage, len(self.sequence) - 1)]
        reward = -0.1
        information_gain = 0.0
        current_known = bool(prev.facts.get("current_slot_known", True))
        if not current_known and action.name in {"search", "read"}:
            if stage < len(self._known_mask) and not self._known_mask[stage]:
                self._known_mask[stage] = True
                information_gain = 1.0
                reward = 0.05
            self._last_observation = {"type": action.name, "slot": stage, "value": expected}
        elif action.name == expected:
            stage += 1
            reward = 0.25
        else:
            stage = max(0, stage - 1)
        done = stage >= len(self.sequence) or steps >= self.max_steps
        success = stage >= len(self.sequence)
        if success:
            reward = 1.0
        self._state = self._make_state(stage, steps, done)
        obs = Observation(f"stage={stage}, expected_done={success}", self._state.vector, dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": success, "information_gain": information_gain})

    def snapshot(self) -> dict:
        return {
            "env_name": self.name,
            "split": self.split,
            "sequence": list(self.sequence),
            "max_steps": self.max_steps,
            "visibility_mode": self.visibility_mode,
            "known_mask": list(self._known_mask),
            "last_observation": dict(self._last_observation),
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
        }

    def restore(self, snapshot: dict) -> WorldState:
        state = snapshot["state"]
        self.sequence = tuple(str(value) for value in snapshot.get("sequence", state.get("sequence", self.sequence)))
        self._validate_sequence_contract()
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.visibility_mode = str(snapshot.get("visibility_mode", state.get("visibility_mode", self.visibility_mode)))
        known_mask = snapshot.get("known_mask", state.get("known_mask"))
        self._known_mask = [bool(value) for value in known_mask] if known_mask is not None else self._initial_known_mask()
        self._last_observation = dict(snapshot.get("last_observation", state.get("last_observation", {"type": "none", "slot": -1, "value": ""})))
        self.action_specs = self._make_action_specs()
        self._state = self._make_state(int(state["stage"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        return [Action(name, source="candidate", confidence=0.7) for name in dict.fromkeys(self.sequence)]

    def expert_action(self, state: WorldState) -> Action:
        stage = int(state.facts["stage"])
        if not bool(state.facts.get("current_slot_known", True)):
            expected = self.sequence[min(stage, len(self.sequence) - 1)]
            if expected in {"search", "read"}:
                return Action(expected, source="expert")
            return Action("search", source="expert")
        return Action(self.sequence[min(stage, len(self.sequence) - 1)], source="expert")

    def success(self, state: WorldState) -> bool:
        return int(state.facts["stage"]) >= len(self.sequence)

    def _initial_known_mask(self) -> list[bool]:
        if self.visibility_mode == "visible":
            return [True] * len(self.sequence)
        if self.visibility_mode == "hidden":
            return [False] * len(self.sequence)
        if not self.sequence:
            return []
        midpoint = max(1, len(self.sequence) // 2)
        return [index < midpoint for index in range(len(self.sequence))]

    def _validate_sequence_contract(self) -> None:
        if "summarize" not in self.sequence:
            return
        last_index = len(self.sequence) - 1
        bad_positions = [index for index, name in enumerate(self.sequence) if name == "summarize" and index != last_index]
        if bad_positions:
            raise ValueError("ToolWorld uses summarize as a terminal action; it must appear only at the final sequence slot.")

    def _make_action_specs(self) -> tuple[ActionSpec, ...]:
        specs: list[ActionSpec] = []
        for name in dict.fromkeys(self.sequence):
            specs.append(
                ActionSpec(
                    name,
                    reveals_information=name in {"search", "read"},
                    requires_known_slot=name == "summarize",
                    terminal_only=name == "summarize",
                )
            )
        return tuple(specs)

    def _make_state(self, stage: int, steps: int, terminal: bool) -> WorldState:
        known_mask = list(self._known_mask)
        if len(known_mask) != len(self.sequence):
            known_mask = self._initial_known_mask()
            self._known_mask = known_mask
        visible_sequence = tuple(self.sequence[index] if known_mask[index] else "" for index in range(len(self.sequence)))
        unknown_mask = [not known for known in known_mask]
        current_slot_known = stage >= len(known_mask) or bool(known_mask[stage])
        remaining_steps = self.max_steps - steps
        current_required_action = self.sequence[min(stage, len(self.sequence) - 1)] if current_slot_known and stage < len(self.sequence) else "unknown"
        return WorldState(
            self.name,
            f"stage{stage}:s{steps}:k{''.join(str(int(value)) for value in known_mask)}",
            (stage / len(self.sequence), steps / self.max_steps, 1.0 if terminal else 0.0),
            {
                "stage": stage,
                "steps": steps,
                "sequence": self.sequence,
                "visible_sequence": visible_sequence,
                "known_mask": tuple(1 if value else 0 for value in known_mask),
                "unknown_mask": tuple(1 if value else 0 for value in unknown_mask),
                "revealed_slots_count": sum(1 for value in known_mask if value),
                "unknown_slots_count": sum(1 for value in unknown_mask if value),
                "current_slot_known": current_slot_known,
                "last_observation": dict(self._last_observation),
                "last_observation_type": str(self._last_observation.get("type", "none")),
                "last_observation_slot": int(self._last_observation.get("slot", -1)),
                "last_observation_value": str(self._last_observation.get("value", "")),
                "current_index": stage,
                "sequence_length": len(self.sequence),
                "remaining_steps": remaining_steps,
                "remaining_sequence_slots": max(0, len(self.sequence) - stage),
                "progress_fraction": stage / max(1, len(self.sequence)),
                "is_last_step_index": stage == len(self.sequence) - 1,
                "current_required_action": current_required_action,
                "can_finish": current_slot_known and stage == len(self.sequence) - 1,
                "can_summarize": current_slot_known and stage == len(self.sequence) - 1 and current_required_action == "summarize",
                "visibility_mode": self.visibility_mode,
                "max_steps": self.max_steps,
                "split": self.split,
            },
            terminal,
        )
