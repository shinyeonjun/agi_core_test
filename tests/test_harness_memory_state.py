import pytest

from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.state_machine import StateTransitionError, assert_transition
from neurokernel_seed.harness.task_spec import task_spec_from_dict


def test_memory_records_task_lifecycle(tmp_path):
    spec = task_spec_from_dict({"goal": "check", "target": "orangepi5", "allowed_actions": ["get_uptime"]})
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        memory.create_task(spec)
        memory.transition_task(spec.task_id, "validated", {})
        memory.transition_task(spec.task_id, "ready", {})
        memory.transition_task(spec.task_id, "running", {})
        task = memory.get_task(spec.task_id)
        events = memory.events_for_task(spec.task_id)
    assert task["status"] == "running"
    assert [event["event_type"] for event in events][:3] == ["created", "validated", "ready"]


def test_state_machine_rejects_approval_skip():
    with pytest.raises(StateTransitionError):
        assert_transition("waiting_approval", "executing")


def test_user_preferences_round_trip_and_delete(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        saved = memory.set_preference("discord:1", "response_length", "short")
        prefs = memory.list_preferences("discord:1")
        deleted = memory.delete_preference("discord:1", "response_length")
        remaining = memory.list_preferences("discord:1")
    assert saved["value_json"] == "short"
    assert prefs[0]["key"] == "response_length"
    assert deleted == 1
    assert remaining == []


def test_conversation_memory_redacts_and_orders_recent_messages(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        memory.add_conversation_message(user_id="discord:1", channel_id="chan", role="user", content="token=abc123 메모리 봐줘")
        memory.add_conversation_message(user_id="discord:1", channel_id="chan", role="assistant", content="확인했어")
        messages = memory.recent_conversation_messages("discord:1", channel_id="chan", limit=10)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert "[REDACTED]" in messages[0]["content_redacted"]


def test_task_references_are_recent_first(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        memory.add_task_reference(user_id="discord:1", task_id="task_a", short_label="메모리 확인", status="completed")
        memory.add_task_reference(user_id="discord:1", task_id="task_b", short_label="모델 확인", status="completed")
        refs = memory.recent_task_references("discord:1")
    assert [ref["task_id"] for ref in refs] == ["task_b", "task_a"]


def test_interaction_outcome_and_message_task_link_are_recorded(tmp_path):
    db = tmp_path / "harness.db"
    spec = task_spec_from_dict({"goal": "메모리 확인", "target": "orangepi5", "allowed_actions": ["get_memory_usage"]})
    with HarnessMemory(db) as memory:
        memory.create_task(spec)
        message = memory.add_conversation_message(user_id="discord:1", channel_id="chan", message_id="m1", role="user", content="메모리 알려줘")
        updated = memory.link_conversation_message_to_task(user_id="discord:1", channel_id="chan", message_id="m1", role="user", task_id=spec.task_id)
        outcome = memory.add_interaction_outcome(
            user_id="discord:1",
            channel_id="chan",
            user_message_id="m1",
            task_id=spec.task_id,
            request_text="메모리 알려줘",
            response_text="메모리 확인 완료",
            required_outputs=["memory_usage"],
            answered_outputs=["memory_usage"],
            missing_outputs=[],
            answer_quality="complete",
            task_status="completed",
            action_id="get_memory_usage",
            success=True,
        )
        linked = memory.recent_conversation_messages("discord:1", channel_id="chan", limit=1)[0]
        grouped = memory.interaction_outcomes_for_tasks([spec.task_id])

    assert message["linked_task_id"] is None
    assert updated == 1
    assert linked["linked_task_id"] == spec.task_id
    assert outcome["required_outputs_json"] == ["memory_usage"]
    assert grouped[spec.task_id][0]["answer_quality"] == "complete"


def test_agent_event_log_is_append_only_and_idempotent(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        first = memory.append_agent_event(
            event_type="supervisor.observed",
            source="test",
            payload={"ok": True},
            idempotency_key="same-observation",
        )
        second = memory.append_agent_event(
            event_type="supervisor.observed",
            source="test",
            payload={"ok": False},
            idempotency_key="same-observation",
        )
        events = memory.recent_agent_events()

    assert first["event_id"] == second["event_id"]
    assert len(events) == 1
    assert events[0]["payload_json"] == {"ok": True}
