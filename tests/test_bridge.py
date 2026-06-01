from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.formatter import split_for_discord, strip_bot_mention
from agent.bridge.router import DiscordEvent, _approval_summary, _goal_summary, _memory_summary, route_discord_event


def config() -> DiscordAuthConfig:
    return DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"10"}, user_cooldown_seconds=0)


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
    assert "secret goal summary marker" in output
    assert "description" not in output
    assert "metadata_json" not in output
    assert "DISCORD_BOT_TOKEN" not in output
    assert "abc123" not in output
