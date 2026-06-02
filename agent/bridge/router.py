from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.formatter import format_approval_card, format_chat_reply, redact_discord_content, split_for_discord, strip_bot_mention
from agent.core.approvals import ApprovalStore
from agent.core.cooldown import is_ready, mark
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goal_generator import meaningful_open_goals
from agent.core.pipeline import run_talk
from agent.core.state import load_state
from agent.lab.planner import run_user_task
from agent.memory.store import search_memories
from agent.scheduler.tick import run_tick

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


def log_discord_input(event: DiscordEvent, text: str, role: str | None = None) -> int:
    init_db()
    metadata = {"guild_id": event.guild_id, "channel_id": event.channel_id, "user_id": event.user_id, "message_id": event.message_id, "is_dm": event.is_dm, "was_mention": event.was_mention, "channel_role": role}
    core_event_id = log_event("discord", "discord_message", redact_discord_content(text), metadata, 0.8)
    with connect() as conn:
        conn.execute("""INSERT INTO discord_events (created_at, guild_id, channel_id, user_id, message_id, is_dm, was_mention, content_redacted, event_id) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)""", (event.guild_id, event.channel_id, event.user_id, event.message_id, 1 if event.is_dm else 0, 1 if event.was_mention else 0, redact_discord_content(text), core_event_id))
        conn.commit()
    return core_event_id


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _state_summary() -> str:
    state = load_state()
    keys = ["version", "mode", "current_focus", "autonomous_level", "risk_level", "memory_clutter", "last_user_interaction_at", "last_idle_tick_at"]
    view = {key: state.get(key) for key in keys}
    return "\n".join([
        "**Core \uc0c1\ud0dc**",
        f"\ubaa8\ub4dc: {view.get('mode')}",
        f"\ucd08\uc810: {view.get('current_focus')}",
        f"\uc790\uc728 \ub808\ubca8: {view.get('autonomous_level')}",
        f"\uc704\ud5d8\ub3c4: {view.get('risk_level')}",
        f"\ub9c8\uc9c0\ub9c9 \ub300\ud654: {view.get('last_user_interaction_at')}",
        f"\ub9c8\uc9c0\ub9c9 tick: {view.get('last_idle_tick_at')}",
    ])


def _memory_summary(query: str) -> str:
    rows = search_memories(query or "core", limit=8)
    lines = ["**\uad00\ub828 \uae30\uc5b5 \uc694\uc57d**"]
    if not rows:
        return "**\uad00\ub828 \uae30\uc5b5 \uc694\uc57d**\n\uc544\uc9c1 \ucc3e\uc740 \uae30\uc5b5\uc774 \uc5c6\uc5b4."
    for row in rows:
        lines.append(f"- #{row.get('id')} {redact_discord_content(str(row.get('title')))} ({row.get('memory_type')}, score={row.get('score')})")
    return "\n".join(lines)


def _approval_summary() -> str:
    rows = ApprovalStore().list_pending()
    if not rows:
        return "**\uc2b9\uc778 \ub300\uae30**\n\uc9c0\uae08 \uc2b9\uc778\ud560 \ud56d\ubaa9\uc740 \uc5c6\uc5b4."
    return "\n\n".join(format_approval_card(row) for row in rows[:10])


def _goal_summary() -> str:
    rows = meaningful_open_goals(limit=10)
    if not rows:
        return "**\ubaa9\ud45c \uc694\uc57d**\n\ub4f1\ub85d\ub41c \ubaa9\ud45c\uac00 \uac70\uc758 \uc5c6\uc5b4."
    lines = ["**\ubaa9\ud45c \uc694\uc57d**"]
    for row in rows:
        lines.append(f"- #{row.get('id')} {redact_discord_content(str(row.get('title')))} / {row.get('status')} / priority={row.get('priority')}")
    return "\n".join(lines)


def handle_command(text: str, *, role: ChannelRole = "chat") -> str | None:
    if not text.startswith("!"):
        return None
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    if role == "approval" and command not in {"!approvals", "!approve", "!reject", "!detail"}:
        return "\uc5ec\uae30\ub294 \uc2b9\uc778 \ucc44\ub110\uc774\uc57c. `!approvals`, `!approve <id>`, `!reject <id>`\ub9cc \uc0ac\uc6a9\ud560 \uc218 \uc788\uc5b4."
    if command == "!state":
        return _state_summary()
    if command == "!goals":
        return _goal_summary()
    if command == "!tick":
        return run_tick()["message"]
    if command == "!memories":
        return _memory_summary(arg)
    if command == "!approvals":
        return _approval_summary()
    if command in {"!detail", "!approval"} and arg.strip().isdigit():
        rows = [row for row in ApprovalStore().list(status=None, limit=100) if int(row.get("id", -1)) == int(arg.strip())]
        return format_approval_card(rows[0]) if rows else "\ud574\ub2f9 \uc2b9\uc778 \ud56d\ubaa9\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc5b4."
    if command == "!approve" and arg.strip().isdigit():
        ok = ApprovalStore().approve(int(arg.strip()))
        return f"\uc2b9\uc778 \uc644\ub8cc: #{arg.strip()}\n연결된 작업이 있으면 사용자 작업 큐로 돌려뒀어." if ok else "\uc2b9\uc778\ud560 \ud56d\ubaa9\uc774 \uc5c6\uac70\ub098 \uc774\ubbf8 \ucc98\ub9ac\ub410\uc5b4."
    if command == "!reject" and arg.strip().isdigit():
        ok = ApprovalStore().reject(int(arg.strip()))
        return f"\uac70\uc808 \uc644\ub8cc: #{arg.strip()}\n연결된 작업이 있으면 차단 상태로 정리했어." if ok else "\uac70\uc808\ud560 \ud56d\ubaa9\uc774 \uc5c6\uac70\ub098 \uc774\ubbf8 \ucc98\ub9ac\ub410\uc5b4."
    return "\uc54c \uc218 \uc5c6\ub294 \uba85\ub839\uc774\uc57c. \uc0ac\uc6a9 \uac00\ub2a5: `!state`, `!goals`, `!tick`, `!memories`, `!approvals`, `!approve <id>`, `!reject <id>`."


def route_discord_event(event: DiscordEvent, config: DiscordAuthConfig) -> list[str]:
    role = channel_role(event.channel_id, config)
    mode = classify_context(user_id=event.user_id, channel_id=event.channel_id, is_dm=event.is_dm, was_mention=event.was_mention, author_is_bot=event.author_is_bot, config=config)
    if mode in {"ignored", "denied"}:
        return []
    text = strip_bot_mention(event.content) if mode == "mention_conversation" else event.content.strip()
    if not text:
        return []
    core_event_id = log_discord_input(event, text, role)
    if role in {"summary", "update"}:
        log_event("discord", "discord_webhook_only_channel_input", text, {"message_id": event.message_id, "channel_role": role}, 0.5)
        return ["\uc774 \ucc44\ub110\uc740 Core\uac00 \uc6f9\ud6c5\uc73c\ub85c \uc54c\ub9bc\ub9cc \ubcf4\ub0b4\ub294 \uacf3\uc774\uc57c. \ub300\ud654\ub294 #\ub300\ud654, \uc2b9\uc778\uc740 #\uc2b9\uc778\uc5d0\uc11c \ud574\uc918."]
    if role == "approval" and not text.startswith("!"):
        log_event("discord", "discord_approval_chat_blocked", text, {"message_id": event.message_id}, 0.55)
        return ["\uc5ec\uae30\ub294 \uc2b9\uc778 \uc804\uc6a9 \ucc44\ub110\uc774\uc57c. `!approvals`, `!approve <id>`, `!reject <id>`\ub9cc \uc0ac\uc6a9\ud560 \uc218 \uc788\uc5b4."]
    ready, wait = is_ready(f"discord_answer:{event.user_id}", config.user_cooldown_seconds)
    if not ready:
        return ["\uc7a0\uae50\ub9cc \ucc9c\ucc9c\ud788 \ubcf4\ub0b4\uc918. \ubc29\uae08 \uba54\uc2dc\uc9c0\ub97c \ucc98\ub9ac \uc911\uc774\uc57c."]
    mark(f"discord_answer:{event.user_id}", config.user_cooldown_seconds, {"channel_id": event.channel_id, "channel_role": role})
    command_output = handle_command(text, role=role)
    if command_output is not None:
        log_event("discord", "discord_command_output", command_output, {"message_id": event.message_id, "channel_role": role}, 0.6)
        return split_for_discord(command_output, config.max_response_chars)
    if role == "approval":
        return ["\uc2b9\uc778 \ucc44\ub110\uc5d0\uc11c\ub294 \uc77c\ubc18 \ub300\ud654\ub97c \ucc98\ub9ac\ud558\uc9c0 \uc54a\uc544. `!approvals`\ub85c \ub300\uae30 \ubaa9\ub85d\uc744 \ud655\uc778\ud574\uc918."]
    result = run_talk(text, source="discord", source_event_id=core_event_id, metadata={"message_id": event.message_id, "channel_role": role})
    user_goal = (result.get("decision") or {}).get("user_directed_goal") or {}
    if user_goal.get("task_id") and user_goal.get("status") == "active":
        result["task_result"] = run_user_task(int(user_goal["task_id"]))
    reply = format_chat_reply(text, result)
    log_event("discord", "discord_chat_reply", reply, {"message_id": event.message_id, "channel_role": role}, 0.55)
    return split_for_discord(reply, config.max_response_chars)
