from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.formatter import split_for_discord, strip_bot_mention
from agent.bridge.router import DiscordEvent, route_discord_event


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
    assert "mode" in chunks[0]


def test_bot_message_ignored():
    event = DiscordEvent(None, "10", "1", "m2", False, False, "hello", author_is_bot=True)
    assert route_discord_event(event, config()) == []



def test_self_check_bridge_cli_shape():
    from agent.cli.agentctl import main
    assert main(["self-check", "bridge"]) == 0
