import pytest

from neurokernel_seed.core.schema import Action
from neurokernel_seed.core.validation import ContractError
from neurokernel_seed.envs.lock_world import LockWorld
from neurokernel_seed.envs.model_needed import ToolPreconditionWorld
from neurokernel_seed.envs.tool_world import ToolWorld
from neurokernel_seed.envs.registry import list_envs, make_env
from neurokernel_seed.replay.dataset import canonical_action_key


def test_registry_has_train_test_worlds():
    assert list_envs("train") == ["lock.train", "maze.train", "tool.train"]
    assert list_envs("test") == ["lock.test", "maze.test", "tool.test"]


def test_worlds_reset_and_expert_success():
    for name in list_envs("test"):
        env = make_env(name)
        state = env.reset(0)
        for _ in range(env.max_steps):
            if state.terminal:
                break
            result = env.step(env.expert_action(state))
            state = result.next_state
        assert env.success(state), name


def test_illegal_lock_action_does_not_succeed():
    env = make_env("lock.test")
    env.reset(0)
    result = env.step(Action("press", {"digit": 9}))
    assert not result.info["success"]
    assert result.reward < 0


def test_snapshot_restore_is_deterministic_for_candidate_action():
    env = make_env("lock.test")
    state = env.reset(0)
    action = env.candidate_actions(state)[0]
    snapshot = env.snapshot()
    result1 = env.step(action)
    restored_state = env.restore(snapshot)
    result2 = env.step(action)
    assert restored_state.state_id == state.state_id
    assert result1.next_state.vector == result2.next_state.vector
    assert result1.reward == result2.reward
    assert result1.done == result2.done
    assert canonical_action_key(action).startswith("press")


def test_partial_hidden_lock_inspect_reveals_current_slot():
    env = LockWorld("lock.partial.unit", "test", (2, 4, 1), 7, visibility_mode="partial_hidden")
    state = env.reset(0)
    state = env.step(env.expert_action(state)).next_state
    assert state.facts["current_slot_known"] is False

    result = env.step(Action("inspect"))

    assert result.info["information_gain"] == 1.0
    assert result.next_state.facts["current_slot_known"] is True
    assert result.next_state.facts["known_mask"] == (1, 1, 0)


def test_partial_hidden_tool_reveal_happens_before_execution():
    env = ToolWorld("tool.partial.unit", "test", 8, ("search", "read", "search", "summarize"), visibility_mode="partial_hidden")
    state = env.reset(0)
    state = env.step(env.expert_action(state)).next_state
    state = env.step(env.expert_action(state)).next_state
    assert state.facts["current_slot_known"] is False

    result = env.step(env.expert_action(state))

    assert result.info["information_gain"] == 1.0
    assert result.next_state.facts["stage"] == state.facts["stage"]
    assert result.next_state.facts["current_slot_known"] is True


def test_tool_world_rejects_nonterminal_summarize_sequences():
    with pytest.raises(ValueError):
        ToolWorld("tool.invalid.unit", "test", 6, ("summarize", "search", "read"))


def test_tool_precondition_rejects_setup_action_in_execution_sequence():
    with pytest.raises(ContractError, match="setup action"):
        ToolPreconditionWorld("tool.precondition.invalid.search", "probe", 8, ("search", "read", "summarize"), False)


def test_tool_precondition_rejects_nonterminal_summarize_sequence():
    with pytest.raises(ContractError, match="terminal"):
        ToolPreconditionWorld("tool.precondition.invalid.terminal", "probe", 8, ("read", "summarize", "read"), False)


def test_hidden_worlds_start_with_no_known_sequence_slots():
    lock = LockWorld("lock.hidden.unit", "test", (2, 4, 1), 8, visibility_mode="hidden")
    tool = ToolWorld("tool.hidden.unit", "test", 8, ("search", "read", "summarize"), visibility_mode="hidden")

    assert lock.reset(0).facts["known_mask"] == (0, 0, 0)
    assert tool.reset(0).facts["known_mask"] == (0, 0, 0)
