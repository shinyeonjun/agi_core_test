from __future__ import annotations

from typing import Any

from neurokernel_seed.core.schema import Action, ActionSpec, Observation, TransitionResult, WorldState
from neurokernel_seed.core.validation import validate_action, validate_execution_sequence
from .base import MicroWorld


class LockTrapWorld(MicroWorld):
    def __init__(
        self,
        name: str = "lock.probe.trap",
        split: str = "probe",
        code: tuple[int, ...] = (2, 4, 1),
        max_steps: int = 8,
        trap_armed: bool = True,
    ):
        self.name = name
        self.split = split
        self.code = code
        self.max_steps = max_steps
        self.trap_armed = trap_armed
        self.action_specs = (
            ActionSpec("inspect", reveals_information=True, consumes_progress_step=False, role="setup"),
            ActionSpec("press", ("digit",), {"digit": int}, requires_known_slot=True),
            ActionSpec("reset", consumes_progress_step=False, role="reset"),
        )
        self._state = self._make_state(0, 0, False)

    def reset(self, seed: int = 0) -> WorldState:
        self.trap_armed = True
        self._state = self._make_state(0, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        progress = int(prev.facts["progress"])
        steps = int(prev.facts["steps"]) + 1
        trap_armed = bool(prev.facts["trap_armed"])
        reward = -0.01
        information_gain = 0.0
        if action.name == "inspect":
            if trap_armed:
                trap_armed = False
                reward = 0.1
                information_gain = 1.0
        elif action.name == "reset":
            progress = 0
            trap_armed = True
        elif action.name == "press":
            digit = int(action.params["digit"])
            if trap_armed and digit == self.code[progress]:
                progress = 0
                reward = -0.35
            elif digit == self.code[progress]:
                progress += 1
                reward = 0.25
            else:
                progress = 0
                reward = -0.2
        done = progress >= len(self.code) or steps >= self.max_steps
        success = progress >= len(self.code)
        if success:
            reward = 1.0
        self.trap_armed = trap_armed
        self._state = self._make_state(progress, steps, done)
        obs = Observation(f"progress={progress}, trap_armed={trap_armed}", self._state.vector, dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": success, "information_gain": information_gain})

    def snapshot(self) -> dict[str, Any]:
        return {
            "env_name": self.name,
            "split": self.split,
            "code": list(self.code),
            "max_steps": self.max_steps,
            "trap_armed": self.trap_armed,
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
            "model_needed_kind": "lock_trap",
        }

    def restore(self, snapshot: dict[str, Any]) -> WorldState:
        state = dict(snapshot["state"])
        self.code = tuple(int(value) for value in snapshot.get("code", state.get("code", self.code)))
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.trap_armed = bool(snapshot.get("trap_armed", state.get("trap_armed", self.trap_armed)))
        self._state = self._make_state(int(state["progress"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        return [Action("inspect", source="candidate", confidence=0.9)] + [Action("press", {"digit": d}, source="candidate", confidence=0.8) for d in range(1, 5)] + [Action("reset", source="candidate", confidence=0.3)]

    def expert_action(self, state: WorldState) -> Action:
        progress = int(state.facts["progress"])
        if progress >= len(self.code):
            return Action("reset", source="expert")
        if bool(state.facts["trap_armed"]):
            return Action("inspect", source="expert")
        return Action("press", {"digit": self.code[progress]}, source="expert")

    def success(self, state: WorldState) -> bool:
        return int(state.facts["progress"]) >= len(self.code)

    def _make_state(self, progress: int, steps: int, terminal: bool) -> WorldState:
        current_slot_known = progress < len(self.code)
        can_finish = current_slot_known and progress == len(self.code) - 1 and not self.trap_armed
        setup_state = "pre_setup" if self.trap_armed else "post_setup_done"
        facts = {
            "progress": progress,
            "steps": steps,
            "code": self.code,
            "visible_code": self.code,
            "known_mask": tuple(1 for _ in self.code),
            "unknown_mask": tuple(0 for _ in self.code),
            "revealed_slots_count": len(self.code),
            "unknown_slots_count": 0,
            "current_slot_known": current_slot_known,
            "last_observation": {"type": "inspect" if not self.trap_armed else "none", "slot": progress if not self.trap_armed else -1, "value": "trap_disarmed" if not self.trap_armed else 0},
            "last_observation_type": "inspect" if not self.trap_armed else "none",
            "last_observation_slot": progress if not self.trap_armed else -1,
            "last_observation_value": "trap_disarmed" if not self.trap_armed else 0,
            "current_index": progress,
            "sequence_length": len(self.code),
            "remaining_steps": self.max_steps - steps,
            "remaining_sequence_slots": max(0, len(self.code) - progress),
            "progress_fraction": progress / max(1, len(self.code)),
            "is_last_step_index": progress == len(self.code) - 1,
            "can_finish": can_finish,
            "can_summarize": False,
            "visibility_mode": "visible",
            "max_steps": self.max_steps,
            "split": self.split,
            "trap_armed": self.trap_armed,
            "trap_disarmed": not self.trap_armed,
            "model_needed_kind": "lock_trap",
            "setup_state": setup_state,
            "setup_known": True,
            "setup_effective": not self.trap_armed,
            "post_setup_state": not self.trap_armed,
            "compatible_but_bad_action": self.trap_armed,
        }
        return WorldState(
            self.name,
            f"p{progress}:s{steps}:trap{int(self.trap_armed)}",
            (progress / max(1, len(self.code)), steps / self.max_steps, 1.0 if terminal else 0.0, 1.0 if self.trap_armed else 0.0),
            facts,
            terminal,
        )


class MazeHazardWorld(MicroWorld):
    doors = ("red", "blue", "green")

    def __init__(
        self,
        name: str = "maze.probe.hazard",
        split: str = "probe",
        target_color: str = "green",
        max_steps: int = 7,
        doors: tuple[str, ...] | None = None,
        hazard_active: bool = True,
        hazard_present: bool = True,
        initial_saw_hint: bool = True,
    ):
        self.name = name
        self.split = split
        self.target_color = target_color
        self.max_steps = max_steps
        self.doors = tuple(doors or self.doors)
        self.hazard_present = hazard_present
        self.initial_hazard_active = bool(hazard_active and hazard_present)
        self.hazard_active = self.initial_hazard_active
        self.initial_saw_hint = initial_saw_hint
        self.action_specs = (
            ActionSpec("inspect", reveals_information=True, consumes_progress_step=False, role="setup"),
            ActionSpec("move", ("door",), {"door": str}, {"door": self.doors}, requires_known_slot=True, terminal_only=True, role="terminal"),
        )
        self._state = self._make_state(self.initial_saw_hint, False, 0, False)

    def reset(self, seed: int = 0) -> WorldState:
        self.hazard_active = self.initial_hazard_active
        self._state = self._make_state(self.initial_saw_hint, False, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        steps = int(prev.facts["steps"]) + 1
        saw_hint = bool(prev.facts["saw_hint"])
        escaped = bool(prev.facts["escaped"])
        hazard_active = bool(prev.facts["hazard_active"])
        reward = -0.02
        information_gain = 0.0
        if action.name == "inspect":
            if self.hazard_present and hazard_active:
                hazard_active = False
                reward = 0.12
                information_gain = 1.0
            elif not self.hazard_present:
                reward = 0.02
            saw_hint = True
        elif action.name == "move":
            door = str(action.params["door"])
            if door == self.target_color and not hazard_active:
                escaped = True
                reward = 1.0
            elif door == self.target_color and hazard_active:
                reward = -0.35
            else:
                reward = -0.25
        done = escaped or steps >= self.max_steps
        self.hazard_active = hazard_active
        self._state = self._make_state(saw_hint, escaped, steps, done)
        obs = Observation(f"target={self.target_color}, hazard_active={hazard_active}, escaped={escaped}", self._state.vector, dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": escaped, "information_gain": information_gain})

    def snapshot(self) -> dict[str, Any]:
        return {
            "env_name": self.name,
            "split": self.split,
            "target_color": self.target_color,
            "doors": list(self.doors),
            "max_steps": self.max_steps,
            "hazard_present": self.hazard_present,
            "hazard_active": self.hazard_active,
            "initial_saw_hint": self.initial_saw_hint,
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
            "model_needed_kind": "maze_hazard",
        }

    def restore(self, snapshot: dict[str, Any]) -> WorldState:
        state = dict(snapshot["state"])
        self.target_color = str(snapshot.get("target_color", state.get("target_color", self.target_color)))
        self.doors = tuple(str(value) for value in snapshot.get("doors", state.get("doors", self.doors)))
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.hazard_present = bool(snapshot.get("hazard_present", state.get("hazard_present", self.hazard_present)))
        self.hazard_active = bool(snapshot.get("hazard_active", state.get("hazard_active", self.hazard_active)))
        self.initial_saw_hint = bool(snapshot.get("initial_saw_hint", state.get("initial_saw_hint", self.initial_saw_hint)))
        self._state = self._make_state(bool(state["saw_hint"]), bool(state["escaped"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        return [Action("inspect", source="candidate", confidence=0.9)] + [Action("move", {"door": color}, source="candidate", confidence=0.7) for color in self.doors]

    def expert_action(self, state: WorldState) -> Action:
        if bool(state.facts["hazard_active"]):
            return Action("inspect", source="expert")
        return Action("move", {"door": self.target_color}, source="expert")

    def success(self, state: WorldState) -> bool:
        return bool(state.facts["escaped"])

    def _make_state(self, saw_hint: bool, escaped: bool, steps: int, terminal: bool) -> WorldState:
        if not self.hazard_present:
            setup_state = "safe_no_hazard"
        else:
            setup_state = "pre_setup" if self.hazard_active else "post_setup_done"
        target_known = bool(saw_hint)
        structural_can_finish = target_known and not escaped
        facts = {
            "saw_hint": saw_hint,
            "escaped": escaped,
            "target_color": self.target_color,
            "visible_target_color": self.target_color if saw_hint else "",
            "doors": self.doors,
            "known_mask": (1,),
            "unknown_mask": (0,),
            "revealed_slots_count": 1,
            "unknown_slots_count": 0,
            "current_slot_known": True,
            "last_observation": {"type": "inspect" if not self.hazard_active else "none", "slot": 0 if not self.hazard_active else -1, "value": "hazard_cleared" if not self.hazard_active else ""},
            "last_observation_type": "inspect" if not self.hazard_active else "none",
            "last_observation_slot": 0 if not self.hazard_active else -1,
            "last_observation_value": "hazard_cleared" if not self.hazard_active else "",
            "current_index": 0,
            "sequence_length": 1,
            "remaining_steps": self.max_steps - steps,
            "remaining_sequence_slots": 1,
            "progress_fraction": 1.0 if escaped else 0.3,
            "is_last_step_index": True,
            "can_finish": structural_can_finish,
            "can_summarize": False,
            "visibility_mode": "visible",
            "steps": steps,
            "max_steps": self.max_steps,
            "split": self.split,
            "initial_saw_hint": self.initial_saw_hint,
            "hazard_present": self.hazard_present,
            "hazard_active": self.hazard_active,
            "hazard_cleared": self.hazard_present and not self.hazard_active,
            "structural_can_finish": structural_can_finish,
            "model_needed_kind": "maze_hazard",
            "setup_state": setup_state,
            "setup_known": True,
            "setup_effective": not self.hazard_active,
            "post_setup_state": not self.hazard_active,
            "compatible_but_bad_action": self.hazard_active,
        }
        return WorldState(
            self.name,
            f"hint{int(saw_hint)}:haz{int(self.hazard_active)}:esc{int(escaped)}:s{steps}",
            (1.0 if saw_hint else 0.0, 1.0 if escaped else 0.0, steps / self.max_steps, 1.0 if self.hazard_active else 0.0),
            facts,
            terminal,
        )


class ToolPreconditionWorld(MicroWorld):
    sequence = ("read", "summarize")

    def __init__(
        self,
        name: str = "tool.probe.precondition",
        split: str = "probe",
        max_steps: int = 6,
        sequence: tuple[str, ...] | None = None,
        precondition_ready: bool = False,
    ):
        self.name = name
        self.split = split
        self.max_steps = max_steps
        self.sequence = tuple(sequence or self.sequence)
        self.precondition_ready = precondition_ready
        self.action_specs = (
            ActionSpec("search", reveals_information=True, consumes_progress_step=False, role="setup"),
            ActionSpec("read", reveals_information=True, requires_known_slot=True),
            ActionSpec("summarize", requires_known_slot=True, terminal_only=True, role="terminal"),
        )
        validate_execution_sequence(self.sequence, self.action_specs)
        self._state = self._make_state(0, 0, False)

    def reset(self, seed: int = 0) -> WorldState:
        self.precondition_ready = False
        self._state = self._make_state(0, 0, False)
        return self._state

    def step(self, action: Action) -> TransitionResult:
        validate_action(action, self.action_specs)
        prev = self._state
        stage = int(prev.facts["stage"])
        steps = int(prev.facts["steps"]) + 1
        ready = bool(prev.facts["precondition_ready"])
        expected = self.sequence[min(stage, len(self.sequence) - 1)]
        reward = -0.1
        information_gain = 0.0
        if action.name == "search":
            if not ready:
                ready = True
                reward = 0.12
                information_gain = 1.0
            else:
                reward = -0.02
        elif action.name == expected:
            if not ready:
                stage = max(0, stage - 1)
                reward = -0.3
            else:
                stage += 1
                reward = 1.0 if stage >= len(self.sequence) else 0.25
        else:
            stage = max(0, stage - 1)
        done = stage >= len(self.sequence) or steps >= self.max_steps
        success = stage >= len(self.sequence)
        self.precondition_ready = ready
        self._state = self._make_state(stage, steps, done)
        obs = Observation(f"stage={stage}, precondition_ready={ready}, success={success}", self._state.vector, dict(self._state.facts))
        return TransitionResult(prev, action, obs, self._state, reward, done, {"success": success, "information_gain": information_gain})

    def snapshot(self) -> dict[str, Any]:
        return {
            "env_name": self.name,
            "split": self.split,
            "sequence": list(self.sequence),
            "max_steps": self.max_steps,
            "precondition_ready": self.precondition_ready,
            "state": dict(self._state.facts),
            "terminal": self._state.terminal,
            "model_needed_kind": "tool_precondition",
        }

    def restore(self, snapshot: dict[str, Any]) -> WorldState:
        state = dict(snapshot["state"])
        self.sequence = tuple(str(value) for value in snapshot.get("sequence", state.get("sequence", self.sequence)))
        self.max_steps = int(snapshot.get("max_steps", state.get("max_steps", self.max_steps)))
        self.precondition_ready = bool(snapshot.get("precondition_ready", state.get("precondition_ready", self.precondition_ready)))
        self._state = self._make_state(int(state["stage"]), int(state["steps"]), bool(snapshot.get("terminal", False)))
        return self._state

    def candidate_actions(self, state: WorldState) -> list[Action]:
        return [Action("search", source="candidate", confidence=0.8), Action("read", source="candidate", confidence=0.8), Action("summarize", source="candidate", confidence=0.7)]

    def expert_action(self, state: WorldState) -> Action:
        stage = int(state.facts["stage"])
        if not bool(state.facts["precondition_ready"]):
            return Action("search", source="expert")
        return Action(self.sequence[min(stage, len(self.sequence) - 1)], source="expert")

    def success(self, state: WorldState) -> bool:
        return int(state.facts["stage"]) >= len(self.sequence)

    def _make_state(self, stage: int, steps: int, terminal: bool) -> WorldState:
        current_required = self.sequence[min(stage, len(self.sequence) - 1)] if stage < len(self.sequence) else "summarize"
        setup_state = "post_setup_done" if self.precondition_ready else "pre_setup"
        facts = {
            "stage": stage,
            "steps": steps,
            "sequence": self.sequence,
            "visible_sequence": self.sequence,
            "known_mask": tuple(1 for _ in self.sequence),
            "unknown_mask": tuple(0 for _ in self.sequence),
            "revealed_slots_count": len(self.sequence),
            "unknown_slots_count": 0,
            "current_slot_known": True,
            "last_observation": {"type": "search" if self.precondition_ready else "none", "slot": 0 if self.precondition_ready else -1, "value": "precondition_ready" if self.precondition_ready else ""},
            "last_observation_type": "search" if self.precondition_ready else "none",
            "last_observation_slot": 0 if self.precondition_ready else -1,
            "last_observation_value": "precondition_ready" if self.precondition_ready else "",
            "current_index": stage,
            "sequence_length": len(self.sequence),
            "remaining_steps": self.max_steps - steps,
            "remaining_sequence_slots": max(0, len(self.sequence) - stage),
            "progress_fraction": stage / max(1, len(self.sequence)),
            "is_last_step_index": stage == len(self.sequence) - 1,
            "current_required_action": current_required,
            "can_finish": self.precondition_ready and stage == len(self.sequence) - 1,
            "can_summarize": self.precondition_ready and current_required == "summarize",
            "visibility_mode": "visible",
            "max_steps": self.max_steps,
            "split": self.split,
            "precondition_ready": self.precondition_ready,
            "model_needed_kind": "tool_precondition",
            "setup_state": setup_state,
            "setup_known": True,
            "setup_effective": self.precondition_ready,
            "post_setup_state": self.precondition_ready,
            "compatible_but_bad_action": not self.precondition_ready,
        }
        return WorldState(
            self.name,
            f"stage{stage}:s{steps}:ready{int(self.precondition_ready)}",
            (stage / max(1, len(self.sequence)), steps / self.max_steps, 1.0 if terminal else 0.0, 1.0 if self.precondition_ready else 0.0),
            facts,
            terminal,
        )
