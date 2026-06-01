from __future__ import annotations

from dataclasses import dataclass

from agent.config.defaults import env_bool, env_csv, env_int


@dataclass(frozen=True)
class DiscordAuthConfig:
    allowed_user_ids: set[str]
    allowed_channel_ids: set[str]
    chat_channel_id: str | None = None
    approval_channel_id: str | None = None
    summary_channel_id: str | None = None
    update_channel_id: str | None = None
    always_chat_in_dm: bool = True
    require_mention_outside_allowed_channels: bool = True
    max_response_chars: int = 1800
    user_cooldown_seconds: int = 3

    @classmethod
    def from_env(cls) -> "DiscordAuthConfig":
        chat_channel_id = _optional_env("DISCORD_CHAT_CHANNEL_ID")
        approval_channel_id = _optional_env("DISCORD_APPROVAL_CHANNEL_ID")
        summary_channel_id = _optional_env("DISCORD_SUMMARY_CHANNEL_ID")
        update_channel_id = _optional_env("DISCORD_UPDATE_CHANNEL_ID")
        allowed = set(env_csv("DISCORD_ALLOWED_CHANNEL_IDS"))
        for channel in [chat_channel_id, approval_channel_id]:
            if channel:
                allowed.add(channel)
        return cls(
            allowed_user_ids=env_csv("DISCORD_ALLOWED_USER_IDS"),
            allowed_channel_ids=allowed,
            chat_channel_id=chat_channel_id,
            approval_channel_id=approval_channel_id,
            summary_channel_id=summary_channel_id,
            update_channel_id=update_channel_id,
            always_chat_in_dm=env_bool("DISCORD_ALWAYS_CHAT_IN_DM", True),
            require_mention_outside_allowed_channels=env_bool("DISCORD_REQUIRE_MENTION_OUTSIDE_ALLOWED_CHANNELS", True),
            max_response_chars=env_int("DISCORD_MAX_RESPONSE_CHARS", 1800),
            user_cooldown_seconds=env_int("DISCORD_USER_COOLDOWN_SECONDS", 3),
        )


def _optional_env(name: str) -> str | None:
    import os

    value = os.environ.get(name, "").strip()
    return value or None


def is_allowed_user(user_id: str, config: DiscordAuthConfig) -> bool:
    return bool(config.allowed_user_ids) and str(user_id) in config.allowed_user_ids


def classify_context(
    *,
    user_id: str,
    channel_id: str,
    is_dm: bool,
    was_mention: bool,
    author_is_bot: bool,
    config: DiscordAuthConfig,
) -> str:
    if author_is_bot:
        return "ignored"
    if not is_allowed_user(user_id, config):
        return "denied"
    if is_dm and config.always_chat_in_dm:
        return "conversation"
    if str(channel_id) in config.allowed_channel_ids:
        return "conversation"
    if was_mention and config.require_mention_outside_allowed_channels:
        return "mention_conversation"
    if was_mention and not config.require_mention_outside_allowed_channels:
        return "mention_conversation"
    return "ignored"
