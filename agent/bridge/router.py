from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.commands import handle_command
from agent.bridge.formatter import format_chat_reply, redact_discord_content, split_for_discord, strip_bot_mention
from agent.core.cooldown import is_ready, mark
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.pipeline import run_talk
from agent.core.wake_signals import emit_wake_signal

ChannelRole = Literal["chat", "approval", "summary", "update", "other"]

@dataclass(frozen=True)
class DiscordEvent:
    guild_id: str | None
    channel_id: str
    user_id: str
    message_id: str
    is_dm: bool
    was_mention: bool
    content: str
    author_is_bot: bool = False


def channel_role(channel_id: str, config: DiscordAuthConfig) -> ChannelRole:
    channel = str(channel_id)
    if config.chat_channel_id and channel == config.chat_channel_id:
        return "chat"
    if config.approval_channel_id and channel == config.approval_channel_id:
        return "approval"
    if config.summary_channel_id and channel == config.summary_channel_id:
        return "summary"
    if config.update_channel_id and channel == config.update_channel_id:
        return "update"
    return "other"


def _discord_message_seen(event: DiscordEvent) -> bool:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM discord_events
            WHERE channel_id = ? AND user_id = ? AND message_id = ?
            LIMIT 1
            """,
            (event.channel_id, event.user_id, event.message_id),
        ).fetchone()
    return row is not None


def log_discord_input(event: DiscordEvent, text: str, role: str | None = None) -> int:
    init_db()
    metadata = {
        "guild_id": event.guild_id,
        "channel_id": event.channel_id,
        "user_id": event.user_id,
        "message_id": event.message_id,
        "is_dm": event.is_dm,
        "was_mention": event.was_mention,
        "channel_role": role,
    }
    core_event_id = log_event("discord", "discord_message", redact_discord_content(text), metadata, 0.8)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO discord_events (
                created_at, guild_id, channel_id, user_id, message_id,
                is_dm, was_mention, content_redacted, event_id
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.guild_id,
                event.channel_id,
                event.user_id,
                event.message_id,
                1 if event.is_dm else 0,
                1 if event.was_mention else 0,
                redact_discord_content(text),
                core_event_id,
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            return 0
    return core_event_id


def route_discord_event(event: DiscordEvent, config: DiscordAuthConfig) -> list[str]:
    role = channel_role(event.channel_id, config)
    mode = classify_context(
        user_id=event.user_id,
        channel_id=event.channel_id,
        is_dm=event.is_dm,
        was_mention=event.was_mention,
        author_is_bot=event.author_is_bot,
        config=config,
    )
    if mode in {"ignored", "denied"}:
        return []
    text = strip_bot_mention(event.content) if mode == "mention_conversation" else event.content.strip()
    if not text:
        return []
    if _discord_message_seen(event):
        log_event("discord", "discord_duplicate_message_ignored", text, {"message_id": event.message_id, "channel_role": role}, 0.35)
        return []
    core_event_id = log_discord_input(event, text, role)
    if core_event_id <= 0:
        log_event("discord", "discord_duplicate_message_ignored", text, {"message_id": event.message_id, "channel_role": role}, 0.35)
        return []
    emit_wake_signal(
        "discord_message",
        "discord",
        priority=0.96 if role == "chat" else 0.72,
        payload={"event_id": core_event_id, "message_id": event.message_id, "channel_id": event.channel_id, "channel_role": role, "is_dm": event.is_dm},
        dedupe_key=f"discord_message:{event.message_id}",
    )
    if role in {"summary", "update"}:
        log_event("discord", "discord_webhook_only_channel_input", text, {"message_id": event.message_id, "channel_role": role}, 0.5)
        return ["이 채널은 Core가 알림만 보내는 곳이야. 대화는 #대화, 승인은 #승인에서 해줘."]
    if role == "approval" and not text.startswith("!"):
        log_event("discord", "discord_approval_chat_blocked", text, {"message_id": event.message_id}, 0.55)
        return ["여기는 승인 전용 채널이야. `!approvals`, `!approve <id>`, `!reject <id>`만 사용할 수 있어."]
    command_output = handle_command(text, role=role)
    if command_output is not None:
        log_event("discord", "discord_command_output", command_output, {"message_id": event.message_id, "channel_role": role}, 0.6)
        return split_for_discord(command_output, config.max_response_chars)
    ready, wait = is_ready(f"discord_answer:{event.user_id}", config.user_cooldown_seconds)
    if not ready:
        log_event("discord", "discord_chat_cooldown_suppressed", "", {"message_id": event.message_id, "channel_role": role, "wait_seconds": wait}, 0.35)
        return []
    mark(f"discord_answer:{event.user_id}", config.user_cooldown_seconds, {"channel_id": event.channel_id, "channel_role": role})
    if role == "approval":
        return ["승인 채널에서는 일반 대화를 처리하지 않아. `!approvals`로 대기 목록을 확인해줘."]
    result = run_talk(text, source="discord", source_event_id=core_event_id, metadata={"message_id": event.message_id, "channel_role": role})
    reply = format_chat_reply(text, result)
    log_event("discord", "discord_chat_reply", reply, {"message_id": event.message_id, "channel_role": role}, 0.55)
    return split_for_discord(reply, config.max_response_chars)
