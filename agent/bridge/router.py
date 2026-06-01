from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.formatter import redact_discord_content, split_for_discord, strip_bot_mention
from agent.core.approvals import ApprovalStore
from agent.core.cooldown import is_ready, mark
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goals import list_goals
from agent.core.pipeline import run_talk
from agent.core.state import load_state
from agent.memory.store import search_memories
from agent.scheduler.tick import run_tick


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


def log_discord_input(event: DiscordEvent, text: str) -> int:
    init_db()
    core_event_id = log_event("discord", "discord_message", redact_discord_content(text), {"guild_id": event.guild_id, "channel_id": event.channel_id, "user_id": event.user_id, "message_id": event.message_id, "is_dm": event.is_dm, "was_mention": event.was_mention}, 0.8)
    with connect() as conn:
        conn.execute("""INSERT INTO discord_events (created_at, guild_id, channel_id, user_id, message_id, is_dm, was_mention, content_redacted, event_id) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)""", (event.guild_id, event.channel_id, event.user_id, event.message_id, 1 if event.is_dm else 0, 1 if event.was_mention else 0, redact_discord_content(text), core_event_id))
        conn.commit()
    return core_event_id


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _state_summary() -> str:
    state = load_state()
    keys = ["version", "mode", "current_focus", "autonomous_level", "risk_level", "memory_clutter", "last_user_interaction_at", "last_idle_tick_at"]
    return _json_text({key: state.get(key) for key in keys})


def _memory_summary(query: str) -> str:
    rows = search_memories(query or "core", limit=8)
    return _json_text([
        {"id": row.get("id"), "title": row.get("title"), "memory_type": row.get("memory_type"), "importance": row.get("importance"), "score": row.get("score")}
        for row in rows
    ])


def _approval_summary() -> str:
    rows = ApprovalStore().list_pending()
    return _json_text([
        {"id": row.get("id"), "action_type": row.get("action_type"), "risk_level": row.get("risk_level"), "status": row.get("status"), "description": row.get("description")}
        for row in rows
    ])


def _goal_summary() -> str:
    rows = list_goals(limit=10)
    return _json_text([
        {
            "id": row.get("id"),
            "title": row.get("title"),
            "goal_type": row.get("goal_type"),
            "status": row.get("status"),
            "priority": row.get("priority"),
            "risk_level": row.get("risk_level"),
            "requires_approval": row.get("requires_approval"),
        }
        for row in rows
    ])


def handle_command(text: str) -> str | None:
    if not text.startswith("!"):
        return None
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
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
    if command == "!approve" and arg.strip().isdigit():
        ok = ApprovalStore().approve(int(arg.strip()))
        return "approved" if ok else "not found or not pending"
    if command == "!reject" and arg.strip().isdigit():
        ok = ApprovalStore().reject(int(arg.strip()))
        return "rejected" if ok else "not found or not pending"
    return "Unknown command. Use !state, !goals, !tick, !memories, !approvals, !approve <id>, !reject <id>."


def route_discord_event(event: DiscordEvent, config: DiscordAuthConfig) -> list[str]:
    mode = classify_context(user_id=event.user_id, channel_id=event.channel_id, is_dm=event.is_dm, was_mention=event.was_mention, author_is_bot=event.author_is_bot, config=config)
    if mode in {"ignored", "denied"}:
        return []
    text = strip_bot_mention(event.content) if mode == "mention_conversation" else event.content.strip()
    if not text:
        return []
    ready, wait = is_ready(f"discord_answer:{event.user_id}", config.user_cooldown_seconds)
    if not ready:
        return ["Please slow down a little."]
    mark(f"discord_answer:{event.user_id}", config.user_cooldown_seconds, {"channel_id": event.channel_id})
    core_event_id = log_discord_input(event, text)
    command_output = handle_command(text)
    if command_output is not None:
        log_event("discord", "discord_command_output", command_output, {"message_id": event.message_id}, 0.6)
        return split_for_discord(command_output, config.max_response_chars)
    result = run_talk(text, source="discord", source_event_id=core_event_id, metadata={"message_id": event.message_id})
    return split_for_discord(result["text"], config.max_response_chars)
