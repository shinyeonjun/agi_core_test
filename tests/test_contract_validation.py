import pytest

from neurokernel_seed.core.gate import ActionGate, current_required_action_key
from neurokernel_seed.core.schema import Action, Prediction
from neurokernel_seed.envs.lock_world import LockWorld
from neurokernel_seed.envs.model_needed import LockTrapWorld, MazeHazardWorld
from neurokernel_seed.core.validation import ContractError, validate_action, validate_prediction
from neurokernel_seed.envs.registry import make_env
from neurokernel_seed.predictors.noop import NoOpPredictor
from neurokernel_seed.predictors.symbolic import SymbolicPredictor


def test_validate_rejects_unknown_action():
    env = make_env("tool.test")
    with pytest.raises(ContractError):
        validate_action(Action("delete_everything"), env.action_specs)


def test_validate_rejects_invalid_action_param_type_before_step():
    env = make_env("lock.test")
    with pytest.raises(ContractError):
        validate_action(Action("press", {"digit": "x"}), env.action_specs)


def test_validate_rejects_unknown_action_param():
    env = make_env("maze.test")
    with pytest.raises(ContractError):
        validate_action(Action("move", {"door": "green", "extra": 1}), env.action_specs)


def test_validate_rejects_disallowed_param_value():
    env = make_env("maze.test")
    with pytest.raises(ContractError):
        validate_action(Action("move", {"door": "black"}), env.action_specs)


def test_prediction_vector_length_contract():
    with pytest.raises(ContractError):
        validate_prediction(Prediction((1.0,)), expected_len=3)


def test_prediction_success_probability_contract():
    with pytest.raises(ContractError):
        validate_prediction(Prediction((1.0, 0.0, 0.0), success_probability=1.5), expected_len=3)


def test_gate_filters_illegal_candidates():
    env = make_env("lock.test")
    state = env.reset(0)
    gate = ActionGate(env.action_specs, NoOpPredictor())
    decision = gate.choose(state, [Action("hack"), Action("press", {"digit": 1})])
    assert decision.action.name == "press"


def test_gate_visibility_guard_prefers_reveal_before_unknown_execution():
    env = LockWorld("lock.partial.unit", "test", (2, 4, 1), 7, visibility_mode="partial_hidden")
    state = env.reset(0)
    state = env.step(env.expert_action(state)).next_state
    assert state.facts["current_slot_known"] is False

    gate = ActionGate(env.action_specs, NoOpPredictor())
    decision = gate.choose(state, [Action("press", {"digit": 4}), Action("inspect")])

    assert decision.action.name == "inspect"


def test_current_required_action_key_uses_visible_lock_slot():
    env = make_env("lock.test")
    state = env.reset(0)

    assert current_required_action_key(state) == "press(digit=2)"


def test_hybrid_gate_prefers_visible_required_lock_action():
    env = make_env("lock.test")
    state = env.reset(0)

    gate = ActionGate(env.action_specs, NoOpPredictor(), mode="hybrid")
    decision = gate.choose(state, env.candidate_actions(state))

    assert decision.action.name == "press"
    assert decision.action.params["digit"] == 2


def test_prior_only_gate_prefers_reveal_when_current_slot_unknown():
    env = LockWorld("lock.hidden.unit", "test", (2, 4, 1), 7, visibility_mode="hidden")
    state = env.reset(0)

    gate = ActionGate(env.action_specs, NoOpPredictor(), mode="prior_only")
    decision = gate.choose(state, env.candidate_actions(state))

    assert decision.action.name == "inspect"


def test_hybrid_veto_can_choose_setup_action_over_compatible_trapped_press():
    env = LockTrapWorld("lock.probe.trap.unit", "probe", (2, 4, 1), 8, trap_armed=True)
    state = env.reset(0)

    prior_decision = ActionGate(env.action_specs, SymbolicPredictor(), mode="prior_only").choose(state, env.candidate_actions(state))
    veto_decision = ActionGate(env.action_specs, SymbolicPredictor(), mode="hybrid_veto").choose(state, env.candidate_actions(state))

    assert prior_decision.action.name == "press"
    assert prior_decision.action.params["digit"] == 2
    assert veto_decision.action.name == "inspect"


def test_hybrid_veto_can_choose_setup_action_over_compatible_hazard_move():
    env = MazeHazardWorld("maze.probe.hazard.active.unit", "probe", "red", 7, ("green", "blue", "red"), hazard_active=True)
    state = env.reset(0)

    prior_decision = ActionGate(env.action_specs, NoOpPredictor(), mode="prior_only").choose(state, env.candidate_actions(state))
    veto_decision = ActionGate(env.action_specs, SymbolicPredictor(), mode="hybrid_veto").choose(state, env.candidate_actions(state))

    assert prior_decision.action.name == "move"
    assert prior_decision.action.params["door"] == "red"
    assert veto_decision.action.name == "inspect"


def test_post_setup_required_action_uses_state_aware_veto_threshold():
    env = MazeHazardWorld("maze.probe.hazard.unit", "probe", "red", 7, ("green", "blue", "red"), hazard_active=False)
    state = env.reset(0)
    state = env.restore({**env.snapshot(), "hazard_active": False, "state": {**env.snapshot()["state"], "hazard_active": False, "hazard_cleared": True, "post_setup_state": True, "setup_state": "post_setup_done"}})
    gate = ActionGate(env.action_specs, NoOpPredictor(), mode="hybrid_veto")
    action = Action("move", {"door": "red"})

    weak_bad_prediction = Prediction(state.vector, -0.20, True, 0.9, "weak bad", progress_delta=-0.02, success_probability=0.0)
    strong_bad_prediction = Prediction(state.vector, -0.20, True, 0.9, "strong bad", progress_delta=-0.20, success_probability=0.0)

    assert gate.model_veto(state, action, weak_bad_prediction) == 0.0
    assert gate.model_veto(state, action, strong_bad_prediction) < 0.0


def test_prior_audit_reports_direct_and_derived_hints_for_maze():
    env = MazeHazardWorld("maze.probe.hazard.audit", "probe", "red", 7, ("green", "blue", "red"), hazard_active=True)
    state = env.reset(0)
    gate = ActionGate(env.action_specs, NoOpPredictor(), mode="prior_only")

    audit = gate.score_breakdown(state, Action("move", {"door": "red"}), Prediction(state.vector))["prior_audit"]

    assert "hazard_active" in audit["sensitive_fields_present"]
    assert "hazard_active" not in audit["sensitive_fields_used_directly"]
    assert "target_color" in audit["derived_hint_fields_used"]
    assert "can_finish" in audit["derived_hint_fields_present"]
