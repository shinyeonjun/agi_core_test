from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.discord_bot import _acquire_single_instance_lock, _release_single_instance_lock
from agent.bridge.formatter import split_for_discord, strip_bot_mention
from agent.bridge.router import DiscordEvent, _approval_summary, _goal_summary, _memory_summary, route_discord_event
from agent.bridge.formatter import format_chat_reply


def config() -> DiscordAuthConfig:
    return DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"10"}, user_cooldown_seconds=0)


def cooldown_config() -> DiscordAuthConfig:
    return DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"10"}, user_cooldown_seconds=30)


def test_classify_allowed_dm():
    assert classify_context(user_id="1", channel_id="dm", is_dm=True, was_mention=False, author_is_bot=False, config=config()) == "conversation"


def test_classify_disallowed_user():
    assert classify_context(user_id="2", channel_id="10", is_dm=False, was_mention=False, author_is_bot=False, config=config()) == "denied"


def test_classify_allowed_channel():
    assert classify_context(user_id="1", channel_id="10", is_dm=False, was_mention=False, author_is_bot=False, config=config()) == "conversation"


def test_classify_other_channel_requires_mention():
    assert classify_context(user_id="1", channel_id="99", is_dm=False, was_mention=False, author_is_bot=False, config=config()) == "ignored"
    assert classify_context(user_id="1", channel_id="99", is_dm=False, was_mention=True, author_is_bot=False, config=config()) == "mention_conversation"


def test_strip_bot_mention():
    assert strip_bot_mention("<@123> hello") == "hello"
    assert strip_bot_mention("<@!123> hello") == "hello"


def test_split_for_discord_limit():
    chunks = split_for_discord("a" * 4000, limit=1800)
    assert chunks
    assert all(len(chunk) <= 1800 for chunk in chunks)


def test_route_command_state_returns_chunks():
    event = DiscordEvent(None, "10", "1", "m1", False, False, "!state")
    chunks = route_discord_event(event, config())
    assert chunks
    assert "Core \uc0c1\ud0dc" in chunks[0]
    assert "renderer" not in chunks[0]


def test_bot_message_ignored():
    event = DiscordEvent(None, "10", "1", "m2", False, False, "hello", author_is_bot=True)
    assert route_discord_event(event, config()) == []


def test_discord_bot_single_instance_lock_blocks_duplicate():
    lock = _acquire_single_instance_lock()
    assert lock is not None
    try:
        assert _acquire_single_instance_lock() is None
    finally:
        _release_single_instance_lock(lock)


def test_discord_bot_single_instance_lock_replaces_stale_pid():
    lock = _acquire_single_instance_lock()
    assert lock is not None
    _release_single_instance_lock(lock)
    lock.write_text("99999999", encoding="utf-8")

    replacement = _acquire_single_instance_lock()
    assert replacement is not None
    try:
        assert replacement.read_text(encoding="utf-8").strip()
    finally:
        _release_single_instance_lock(replacement)


def test_chat_cooldown_suppresses_template_reply():
    first = DiscordEvent(None, "10", "1", "cooldown-1", False, False, "첫 메시지")
    second = DiscordEvent(None, "10", "1", "cooldown-2", False, False, "두 번째 메시지")
    assert route_discord_event(first, cooldown_config())
    assert route_discord_event(second, cooldown_config()) == []



def test_self_check_bridge_cli_shape():
    from agent.cli.agentctl import main
    assert main(["self-check", "bridge"]) == 0


def test_command_memory_summary_hides_content():
    from agent.memory.store import add_memory

    add_memory("secret marker memory", "DISCORD_BOT_TOKEN=abc123", tags=["secret_marker"])
    output = _memory_summary("secret marker")
    assert "secret marker memory" in output
    assert "DISCORD_BOT_TOKEN" not in output
    assert "content" not in output


def test_command_approval_summary_hides_payload():
    output = _approval_summary()
    assert "proposed_payload_json" not in output
    assert "proposal" not in output


def test_command_goal_summary_hides_description_and_metadata():
    from agent.core.goals import create_goal

    create_goal(
        "secret goal summary marker",
        "description contains DISCORD_BOT_TOKEN=abc123",
        goal_type="test",
        metadata={"token": "abc123"},
        dedupe=False,
    )
    output = _goal_summary()
    assert "secret goal summary marker" not in output
    assert "description" not in output
    assert "metadata_json" not in output
    assert "DISCORD_BOT_TOKEN" not in output
    assert "abc123" not in output



def test_route_chat_hides_fallback_renderer():
    event = DiscordEvent(None, "10", "1", "m3", False, False, "\u314e\u3147")
    output = "\n".join(route_discord_event(event, config()))
    assert "fallback renderer" not in output
    assert "goal:" not in output
    assert "Core가 지금 입력은 기록해뒀어" in output
    assert "잠깐만" not in output
    assert "천천히" not in output


def test_format_chat_reply_prefers_core_renderer_text():
    output = format_chat_reply(
        "그 너 코어 어떻게 이루어져있어?",
        {
            "text": "Core는 language, memory, goal, policy, renderer, scheduler로 나뉘어 있어.",
            "decision": {
                "policy_summary": {"risk_level": "low", "requires_approval": False, "denied": False},
                "language_interpretation": {"intent": "chat", "target": "architecture"},
            },
        },
    )

    assert "language, memory" in output


def test_format_chat_reply_rejects_internal_renderer_text():
    output = format_chat_reply(
        "그럼 목표로 된거임?",
        {
            "text": "selected_goal_id: 184, user_goal_created: false라서 새 목표는 아니야.",
            "decision": {
                "policy_summary": {"risk_level": "low", "requires_approval": False, "denied": False},
                "language_interpretation": {"intent": "chat", "target": "question"},
            },
        },
    )

    assert "selected_goal_id" not in output
    assert "user_goal_created" not in output
    assert "Core가 지금 입력은 기록해뒀어" in output
