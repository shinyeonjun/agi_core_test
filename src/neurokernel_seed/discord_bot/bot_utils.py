from __future__ import annotations

import asyncio
import json
from typing import Any

from .core_client import CoreClientError


async def call_blocking(func: Any, *args: Any) -> Any:
    try:
        return await asyncio.to_thread(func, *args)
    except CoreClientError:
        raise


def discord_user_id(message: Any) -> str:
    author = getattr(message, "author", None)
    return f"discord:{int(getattr(author, 'id', 0) or 0)}"


def discord_channel_id(message: Any) -> str:
    channel = getattr(message, "channel", None)
    return str(int(getattr(channel, "id", 0) or 0))


def interaction_user_id(interaction: Any) -> str:
    user = getattr(interaction, "user", None)
    return f"discord:{int(getattr(user, 'id', 0) or 0)}"


def parse_pref_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def format_pref_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def message_allowed(message: Any, config: Any) -> bool:
    author = getattr(message, "author", None)
    author_id = int(getattr(author, "id", 0) or 0)
    if config.allowed_user_ids and author_id not in config.allowed_user_ids:
        return False
    channel = getattr(message, "channel", None)
    channel_id = getattr(channel, "id", None)
    guild = getattr(message, "guild", None)
    if guild is None:
        return config.allow_dms
    return int(channel_id or 0) == config.channel_id


def parse_user_ids(raw: str) -> tuple[int, ...]:
    result = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        result.append(int(chunk))
    return tuple(result)


def is_auto_executable_task(task: Any) -> bool:
    if not isinstance(task, dict):
        return False
    if task.get("risk_level") not in {"none", "low"}:
        return False
    if task.get("requires_approval"):
        return False
    if task.get("mode") != "readonly":
        return False
    allowed = task.get("allowed_actions")
    if not isinstance(allowed, list) or not allowed:
        return False
    safe_actions = {
        "get_uptime",
        "get_disk_usage",
        "get_memory_usage",
        "get_cpu_temp",
        "get_cpu_per_core_usage",
        "get_service_status",
        "tail_logs",
        "list_artifacts",
        "get_recent_trace",
    }
    return all(action in safe_actions for action in allowed)


async def reply(message: Any, text: str, *, view: Any | None = None) -> None:
    chunks = discord_chunks(text)
    for index, chunk in enumerate(chunks):
        await message.channel.send(chunk, view=view if index == 0 else None)


def discord_chunks(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        chunks.append(remaining[:limit])
        remaining = remaining[limit:]
    return chunks


def parse_int(raw: str, *, default: int, minimum: int, maximum: int) -> int:
    if not raw:
        return default
    value = int(raw.split()[0])
    return max(minimum, min(maximum, value))


def split_id_reason(raw: str) -> tuple[str, str | None]:
    task_id, _, reason = raw.partition(" ")
    if not task_id:
        raise ValueError("task_id is required")
    return task_id, reason.strip() or None
