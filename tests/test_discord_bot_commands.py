import pytest

from neurokernel_seed.discord_bot.bot import DiscordBotConfig, _command_line_from_content, _discord_chunks, _is_auto_executable_task, _message_allowed, _parse_user_ids
from neurokernel_seed.discord_bot.commands import build_benchmark_task, build_preset_task, parse_task_json


TEST_CHANNEL_ID = 123456789012345678


def _bot_config(**overrides):
    payload = {"token": "token", "channel_id": TEST_CHANNEL_ID, "core_url": "http://127.0.0.1:8765", "prefix": ""}
    payload.update(overrides)
    return DiscordBotConfig(**payload)


def test_build_preset_task_uses_readonly_action():
    task = build_preset_task("disk")
    assert task["allowed_actions"] == ["get_disk_usage"]
    assert task["requires_approval"] is False
    assert task["mode"] == "readonly"


def test_build_preset_task_rejects_unknown_preset():
    with pytest.raises(ValueError):
        build_preset_task("delete_everything")


def test_build_benchmark_task_is_low_risk():
    task = build_benchmark_task(episodes=2)
    assert task["allowed_actions"] == ["run_safe_benchmark"]
    assert task["context"]["params"]["episodes"] == 2
    assert task["risk_level"] == "low"


def test_parse_task_json_requires_object():
    assert parse_task_json('{"goal":"x"}') == {"goal": "x"}
    with pytest.raises(ValueError):
        parse_task_json("[1, 2, 3]")


def test_message_allowed_only_configured_channel():
    class Channel:
        id = TEST_CHANNEL_ID

    class Author:
        id = 123

    class Message:
        channel = Channel()
        guild = object()
        author = Author()

    config = _bot_config()
    assert _message_allowed(Message(), config)


def test_message_denies_other_channels_and_dms_by_default():
    class OtherChannel:
        id = 1

    class GuildMessage:
        channel = OtherChannel()
        guild = object()
        author = type("Author", (), {"id": 123})()

    class DmMessage:
        channel = OtherChannel()
        guild = None
        author = type("Author", (), {"id": 123})()

    config = _bot_config()
    assert not _message_allowed(GuildMessage(), config)
    assert not _message_allowed(DmMessage(), config)


def test_discord_chunks_keeps_short_messages_intact():
    assert _discord_chunks("hello") == ["hello"]
    assert len(_discord_chunks("x" * 4000, limit=1000)) == 4


def test_allowed_user_filter_blocks_other_users():
    class Channel:
        id = TEST_CHANNEL_ID

    class Message:
        channel = Channel()
        guild = object()
        author = type("Author", (), {"id": 222})()

    config = _bot_config(allowed_user_ids=(111,))
    assert not _message_allowed(Message(), config)


def test_allowed_user_filter_allows_configured_user():
    class Channel:
        id = TEST_CHANNEL_ID

    class Message:
        channel = Channel()
        guild = object()
        author = type("Author", (), {"id": 111})()

    config = _bot_config(allowed_user_ids=(111,))
    assert _message_allowed(Message(), config)


def test_parse_user_ids_accepts_single_or_many_values():
    assert _parse_user_ids("1") == (1,)
    assert _parse_user_ids("1, 2;3") == (1, 2, 3)


def test_empty_prefix_routes_unknown_text_to_conversation():
    config = _bot_config(prefix="", reply_without_prefix=True)
    assert _command_line_from_content("ㅎㅎㅎㅎ", config) == "auto ㅎㅎㅎㅎ"
    assert _command_line_from_content("암마", config) == "auto 암마"


def test_empty_prefix_keeps_known_commands_available():
    config = _bot_config(prefix="", reply_without_prefix=True)
    assert _command_line_from_content("help", config) == "help"
    assert _command_line_from_content("memory recent", config) == "memory recent"


def test_nonempty_prefix_keeps_prefixed_command_mode():
    config = _bot_config(prefix="!nk", reply_without_prefix=False)
    assert _command_line_from_content("ㅎㅎㅎㅎ", config) is None
    assert _command_line_from_content("!nk help", config) == "help"


def test_auto_executable_allows_low_risk_readonly_lookup():
    assert _is_auto_executable_task(
        {
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
            "allowed_actions": ["get_memory_usage"],
        }
    )


def test_auto_executable_blocks_reboot_even_if_task_exists():
    assert not _is_auto_executable_task(
        {
            "risk_level": "high",
            "requires_approval": True,
            "mode": "readonly",
            "allowed_actions": ["reboot"],
        }
    )


def test_auto_executable_blocks_benchmark_for_conversational_mode():
    assert not _is_auto_executable_task(
        {
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
            "allowed_actions": ["run_safe_benchmark"],
        }
    )
