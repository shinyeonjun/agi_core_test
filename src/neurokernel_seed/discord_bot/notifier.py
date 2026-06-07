from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from .bot_utils import call_blocking as _call
from .bot_utils import discord_chunks as _discord_chunks
from .core_client import CoreClient
from .models import DiscordBotConfig
from .work_status import format_work_notification


async def work_notification_loop(
    client: Any,
    core: CoreClient,
    config: DiscordBotConfig,
    activation_view_factory: Any | None,
    retry_view_factory: Any | None,
    promote_view_factory: Any | None,
) -> None:
    seen: dict[str, tuple[str, str]] = {}
    first_poll = True
    while True:
        try:
            payload = await _call(core.get, "/work-items?limit=30")
            items = payload.get("work_items") if isinstance(payload, dict) else []
            if isinstance(items, list):
                await notify_work_changes(client, core, config, activation_view_factory, retry_view_factory, promote_view_factory, items, seen=seen, first_poll=first_poll)
            first_poll = False
        except Exception as exc:
            print(f"[discord-work-notifier] poll failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(max(1.0, float(config.work_notify_interval_seconds)))


async def notify_work_changes(
    client: Any,
    core: CoreClient,
    config: DiscordBotConfig,
    activation_view_factory: Any | None,
    retry_view_factory: Any | None,
    promote_view_factory: Any | None,
    items: list[Any],
    *,
    seen: dict[str, tuple[str, str]],
    first_poll: bool,
) -> None:
    for item in items:
        if not isinstance(item, dict):
            continue
        work_id = str(item.get("work_id") or "").strip()
        if not work_id:
            continue
        status = str(item.get("status") or "")
        updated_at = str(item.get("updated_at") or "")
        signature = (status, updated_at)
        previous = seen.get(work_id)
        seen[work_id] = signature
        if first_poll or previous == signature or not is_notifiable_work_status(status):
            continue
        detail = await _call(core.get, f"/work-items/{quote(work_id)}")
        if not isinstance(detail, dict):
            continue
        text, view_kind = format_work_notification(detail)
        channel = await resolve_notification_channel(client, item, config)
        if channel is None:
            print(f"[discord-work-notifier] channel not found for work_id={work_id}", flush=True)
            continue
        view = None
        if view_kind == "activation" and activation_view_factory:
            view = activation_view_factory(work_id)
        elif view_kind == "retry" and retry_view_factory:
            view = retry_view_factory(work_id)
        elif view_kind == "promote" and promote_view_factory:
            view = promote_view_factory(work_id)
        for index, chunk in enumerate(_discord_chunks(text)):
            await channel.send(chunk, view=view if index == 0 else None)


def is_notifiable_work_status(status: str) -> bool:
    return status in {"planned", "waiting_approval", "reviewing", "blocked", "failed", "completed"}


async def resolve_notification_channel(client: Any, item: dict[str, Any], config: DiscordBotConfig) -> Any | None:
    raw_channel_id = item.get("channel_id") or config.channel_id
    try:
        channel_id = int(raw_channel_id)
    except (TypeError, ValueError):
        channel_id = config.channel_id
    channel = client.get_channel(channel_id) if hasattr(client, "get_channel") else None
    if channel is not None:
        return channel
    if hasattr(client, "fetch_channel"):
        return await client.fetch_channel(channel_id)
    return None
