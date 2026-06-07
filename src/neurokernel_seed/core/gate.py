from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .actions import find_action_spec, is_action_allowed
from .schema import Action, ActionSpec, Prediction, WorldState
from .validation import validate_action, validate_prediction

GateMode = Literal["model_only", "prior_only", "hybrid", "hybrid_veto"]


class Predictor(Protocol):
    name: str

    def predict(self, state: WorldState, action: Action) -> Prediction:
        ...


@dataclass(frozen=True)
class GateDecision:
    action: Action
    prediction: Prediction
    score: float
    reason: str


@dataclass(frozen=True)
class GateConfig:
    mode: GateMode = "hybrid"
    enforce_compatibility: bool = True
    hard_block_score: float = -1_000_000.0
    match_bonus: float = 1.0
    mismatch_penalty: float = 1.2
    information_need_bonus: float = 1.0
    unknown_execution_penalty: float = 2.0
    early_finish_penalty: float = 2.0
    reset_progress_penalty: float = 0.8
    bad_reward_threshold: float = -0.05
    bad_progress_threshold: float = -0.05
    strong_bad_reward_threshold: float = -0.15
    strong_bad_progress_threshold: float = -0.10
    model_bad_outcome_penalty: float = 1.5
    terminal_success_threshold: float = 0.5
    bad_terminal_penalty: float = 1.5


@dataclass(frozen=True)
class CompatibilityInfo:
    required_action_key: str | None
    action_key: str
    action_matches_required: bool
    executes_current_slot: bool
    reveals_information: bool
    requires_known_slot: bool
    terminal_only: bool
    reset_like: bool
    current_required_action_known: bool
    current_slot_known: bool
    progress_fraction: float


class ActionGate:
    """Scores legal candidates using predicted progress, reward, success, and confidence."""

    def __init__(self, action_specs: tuple[ActionSpec, ...], predictor: Predictor, config: GateConfig | None = None, *, mode: GateMode | None = None):
        self.action_specs = action_specs
        self.predictor = predictor
        base = config or GateConfig()
        self.config = base if mode is None else GateConfig(
            mode=mode,
            enforce_compatibility=base.enforce_compatibility,
            hard_block_score=base.hard_block_score,
            match_bonus=base.match_bonus,
            mismatch_penalty=base.mismatch_penalty,
            information_need_bonus=base.information_need_bonus,
            unknown_execution_penalty=base.unknown_execution_penalty,
            early_finish_penalty=base.early_finish_penalty,
            reset_progress_penalty=base.reset_progress_penalty,
            bad_reward_threshold=base.bad_reward_threshold,
            bad_progress_threshold=base.bad_progress_threshold,
            strong_bad_reward_threshold=base.strong_bad_reward_threshold,
            strong_bad_progress_threshold=base.strong_bad_progress_threshold,
            model_bad_outcome_penalty=base.model_bad_outcome_penalty,
            terminal_success_threshold=base.terminal_success_threshold,
            bad_terminal_penalty=base.bad_terminal_penalty,
        )

    def choose(self, state: WorldState, candidates: list[Action]) -> GateDecision:
        legal = [candidate for candidate in candidates if is_action_allowed(candidate, self.action_specs)]
        if not legal:
            raise ValueError("no legal action candidates")
        scoring_state = self._visible_state_for_gate(state)
        best: GateDecision | None = None
        for action in legal:
            validate_action(action, self.action_specs)
            prediction = self.predictor.predict(state, action)
            validate_prediction(prediction, len(state.vector))
            score = self.score(scoring_state, action, prediction)
            decision = GateDecision(action=action, prediction=prediction, score=score, reason=prediction.reason)
            if best is None or decision.score > best.score:
                best = decision
        assert best is not None
        return best

    def score(self, state: WorldState, action: Action, prediction: Prediction) -> float:
        return float(self.score_breakdown(state, action, prediction)["final_score"])

    def score_breakdown(self, state: WorldState, action: Action, prediction: Prediction) -> dict[str, Any]:
        model_score = self._score(state, prediction)
        prior_components = self.compatibility_prior_components(state, action)
        prior_score = float(prior_components["total"])
        model_veto = self.model_veto(state, action, prediction)
        compatibility_blocked = self._compatibility_blocked(state, action)
        veto_escape_allowed = self._veto_escape_allowed(state, action)
        hard_blocked = False
        if self.config.mode == "model_only":
            final_score = model_score
        elif self.config.mode == "prior_only":
            final_score = prior_score
        elif self.config.mode == "hybrid_veto":
            hard_blocked = self.config.enforce_compatibility and compatibility_blocked and not veto_escape_allowed
            if hard_blocked:
                final_score = self.config.hard_block_score + prior_score
            else:
                final_score = model_score + prior_score + model_veto
        else:
            hard_blocked = self.config.enforce_compatibility and compatibility_blocked
            if hard_blocked:
                final_score = self.config.hard_block_score + prior_score
            else:
                final_score = model_score + prior_score
        return {
            "model_score": float(model_score),
            "compatibility_prior": float(prior_score),
            "match_guard": float(prior_components["match_guard"]),
            "information_guard": float(prior_components["information_guard"]),
            "visibility_guard": float(prior_components["visibility_guard"]),
            "finish_guard": float(prior_components["finish_guard"]),
            "reset_guard": float(prior_components["reset_guard"]),
            "model_veto": float(model_veto),
            "compatibility_blocked": bool(compatibility_blocked),
            "veto_escape_allowed": bool(veto_escape_allowed),
            "hard_blocked": bool(hard_blocked),
            "final_score": float(final_score),
            "prior_audit": self.compatibility_prior_audit(state, action),
        }

    @staticmethod
    def _score(state: WorldState, prediction: Prediction) -> float:
        current_progress = float(state.vector[0])
        predicted_progress = float(prediction.next_state_vector[0])
        progress_gain = prediction.progress_delta if prediction.progress_delta is not None else predicted_progress - current_progress
        if prediction.success_probability is None:
            success_score = 1.0 if prediction.terminal and prediction.reward > 0 else 0.0
        else:
            success_score = float(prediction.success_probability)
        information_gain = float(prediction.information_gain or 0.0)
        return float(prediction.reward) + 2.0 * float(progress_gain) + information_gain + 0.5 * success_score

    def _visible_state_for_gate(self, state: WorldState) -> WorldState:
        visible_state = getattr(self.predictor, "visible_state", None)
        if callable(visible_state):
            return visible_state(state)
        return state

    def compatibility_prior(self, state: WorldState, action: Action) -> float:
        return float(self.compatibility_prior_components(state, action)["total"])

    def compatibility_prior_components(self, state: WorldState, action: Action) -> dict[str, float]:
        info = self.compatibility_info(state, action)
        match_guard = 0.0
        information_guard = 0.0
        visibility_guard = 0.0
        finish_guard = 0.0
        reset_guard = 0.0
        if info.current_required_action_known:
            if info.action_matches_required:
                match_guard += self.config.match_bonus
            elif info.executes_current_slot:
                match_guard -= self.config.mismatch_penalty
        else:
            if info.reveals_information:
                information_guard += self.config.information_need_bonus
            elif info.requires_known_slot or info.executes_current_slot:
                visibility_guard -= self.config.unknown_execution_penalty
        if info.terminal_only and not bool(state.facts.get("can_finish", state.facts.get("is_last_step_index", False))):
            finish_guard -= self.config.early_finish_penalty
        if info.reset_like and info.progress_fraction > 0.0:
            reset_guard -= self.config.reset_progress_penalty
        total = match_guard + information_guard + visibility_guard + finish_guard + reset_guard
        return {
            "match_guard": float(match_guard),
            "information_guard": float(information_guard),
            "visibility_guard": float(visibility_guard),
            "finish_guard": float(finish_guard),
            "reset_guard": float(reset_guard),
            "total": float(total),
        }

    def compatibility_prior_audit(self, state: WorldState, action: Action) -> dict[str, Any]:
        info = self.compatibility_info(state, action)
        family = state.env_name.split(".", 1)[0]
        used_fields = {
            "action.name",
            "action.params",
            "action_spec.reveals_information",
            "action_spec.requires_known_slot",
            "action_spec.terminal_only",
            "current_slot_known",
            "progress_fraction",
        }
        if info.current_required_action_known:
            used_fields.add("current_required_action_key")
        if info.terminal_only:
            used_fields.update({"can_finish", "is_last_step_index"})
        if info.reset_like:
            used_fields.add("reset_like")
        if family == "lock":
            used_fields.update({"visible_code", "code", "current_index"})
        elif family == "tool":
            used_fields.update({"current_required_action", "visible_sequence", "sequence", "current_index"})
        elif family == "maze":
            used_fields.update({"visible_target_color", "target_color", "saw_hint"})
        sensitive_fields = {
            "hazard_active",
            "hazard_cleared",
            "trap_armed",
            "trap_disarmed",
            "precondition_ready",
            "precondition_missing",
            "door_cost",
            "blocked",
        }
        derived_hint_fields = {
            "can_finish",
            "is_last_step_index",
            "post_setup_state",
            "setup_state",
            "setup_effective",
            "compatible_but_bad_action",
            "current_required_action",
            "visible_target_color",
            "target_color",
            "saw_hint",
        }
        fact_keys = set(state.facts)
        return {
            "used_fields": sorted(used_fields),
            "sensitive_fields_present": sorted(fact_keys & sensitive_fields),
            "sensitive_fields_used_directly": sorted(used_fields & sensitive_fields),
            "derived_hint_fields_present": sorted(fact_keys & derived_hint_fields),
            "derived_hint_fields_used": sorted(used_fields & derived_hint_fields),
            "family": family,
        }

    def compatibility_info(self, state: WorldState, action: Action) -> CompatibilityInfo:
        spec = find_action_spec(action, self.action_specs)
        required = current_required_action_key(state)
        action_key = _action_key(action)
        executes = _executes_current_slot(state, action, spec)
        reset_like = action.name == "reset"
        return CompatibilityInfo(
            required_action_key=required,
            action_key=action_key,
            action_matches_required=required is not None and action_key == required,
            executes_current_slot=executes,
            reveals_information=bool(spec.reveals_information) if spec else False,
            requires_known_slot=bool(spec.requires_known_slot) if spec else False,
            terminal_only=bool(spec.terminal_only) if spec else False,
            reset_like=reset_like,
            current_required_action_known=required is not None,
            current_slot_known=bool(state.facts.get("current_slot_known", True)),
            progress_fraction=float(state.facts.get("progress_fraction", state.vector[0] if state.vector else 0.0) or 0.0),
        )

    def model_veto(self, state: WorldState, action: Action, prediction: Prediction) -> float:
        info = self.compatibility_info(state, action)
        veto = 0.0
        bad_reward = float(prediction.reward) < self.config.bad_reward_threshold
        bad_progress = prediction.progress_delta is not None and float(prediction.progress_delta) < self.config.bad_progress_threshold
        strong_bad_reward = float(prediction.reward) < self.config.strong_bad_reward_threshold
        strong_bad_progress = prediction.progress_delta is not None and float(prediction.progress_delta) < self.config.strong_bad_progress_threshold
        strong_bad_outcome = strong_bad_reward and strong_bad_progress
        post_setup_required = info.action_matches_required and info.executes_current_slot and _post_setup_required_action(state)
        if info.action_matches_required and info.executes_current_slot and (bad_reward or bad_progress):
            if post_setup_required:
                if strong_bad_outcome:
                    veto -= self.config.model_bad_outcome_penalty
            else:
                veto -= self.config.model_bad_outcome_penalty
        success_probability = prediction.success_probability
        if info.terminal_only and success_probability is not None and float(success_probability) < self.config.terminal_success_threshold:
            if post_setup_required:
                if strong_bad_outcome:
                    veto -= self.config.bad_terminal_penalty
            else:
                veto -= self.config.bad_terminal_penalty
        return veto

    def _compatibility_blocked(self, state: WorldState, action: Action) -> bool:
        info = self.compatibility_info(state, action)
        can_finish = bool(state.facts.get("can_finish", state.facts.get("is_last_step_index", False)))
        if info.terminal_only:
            return not can_finish
        if info.current_required_action_known:
            return not info.action_matches_required
        if info.requires_known_slot or info.executes_current_slot:
            return not info.reveals_information
        return False

    def _veto_escape_allowed(self, state: WorldState, action: Action) -> bool:
        info = self.compatibility_info(state, action)
        if info.current_required_action_known and info.reveals_information and not info.executes_current_slot:
            return True
        if info.current_required_action_known and info.reveals_information and bool(state.facts.get("compatible_but_bad_action", False)):
            return True
        return False


def current_required_action_key(state: WorldState) -> str | None:
    facts = state.facts
    if not bool(facts.get("current_slot_known", True)):
        return None
    family = state.env_name.split(".", 1)[0]
    current_index = int(facts.get("current_index", facts.get("progress", facts.get("stage", 0))) or 0)
    if family == "lock":
        visible_code = list(facts.get("visible_code") or facts.get("code") or [])
        if 0 <= current_index < len(visible_code):
            digit = visible_code[current_index]
            if digit not in (0, "", None, "unknown"):
                return f"press(digit={int(digit)})"
    if family == "tool":
        required = str(facts.get("current_required_action") or "")
        if required and required != "unknown":
            return required
        visible_sequence = list(facts.get("visible_sequence") or facts.get("sequence") or [])
        if 0 <= current_index < len(visible_sequence):
            action_name = visible_sequence[current_index]
            if action_name not in ("", None, "unknown"):
                return str(action_name)
    if family == "maze":
        visible_color = facts.get("visible_target_color")
        if visible_color in ("", None, "unknown"):
            visible_color = facts.get("target_color") if bool(facts.get("saw_hint")) else None
        if visible_color not in ("", None, "unknown"):
            return f'move(door="{visible_color}")'
    return None


def _executes_current_slot(state: WorldState, action: Action, spec: ActionSpec | None) -> bool:
    family = state.env_name.split(".", 1)[0]
    if family == "lock":
        return action.name == "press"
    if family == "maze":
        return action.name == "move"
    if family == "tool":
        return action.name in {"search", "read", "summarize"}
    return bool(spec and spec.requires_known_slot)


def _post_setup_required_action(state: WorldState) -> bool:
    facts = state.facts
    if bool(facts.get("hazard_active", False)):
        return False
    return bool(facts.get("post_setup_state", False)) or str(facts.get("setup_state", "")) in {"post_setup_done", "safe_no_hazard"}


def _action_key(action: Action) -> str:
    if not action.params:
        return action.name
    args = ",".join(f"{key}={_format_param(value)}" for key, value in sorted(action.params.items()))
    return f"{action.name}({args})"


def _format_param(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)
