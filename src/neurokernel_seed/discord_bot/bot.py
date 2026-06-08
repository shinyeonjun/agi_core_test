from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

from .bot_utils import call_blocking as _call
from .bot_utils import discord_channel_id as _discord_channel_id
from .bot_utils import discord_chunks as _discord_chunks
from .bot_utils import discord_user_id as _discord_user_id
from .bot_utils import is_auto_executable_task as _is_auto_executable_task
from .bot_utils import message_allowed as _message_allowed
from .bot_utils import parse_user_ids as _parse_user_ids
from .bot_utils import reply as _reply
from .command_handlers import activation_verify_note as _activation_verify_note
from .command_handlers import create_and_run as _create_and_run
from .command_handlers import handle_command as _handle_command
from .command_handlers import handle_work as _handle_work
from .command_handlers import humanize as _humanize
from .command_handlers import language_to_core as _language_to_core
from .command_handlers import maybe_create_capability_proposal as _maybe_create_capability_proposal
from .command_handlers import maybe_route_work as _maybe_route_work
from .command_handlers import required_text as _required_text
from .command_handlers import run_preset as _run_preset
from .config import config_from_env
from .config import optional_bool_env as _optional_bool_env
from .config import required_bool_env as _required_bool_env
from .config import required_env as _required_env
from .config import required_env_defined as _required_env_defined
from .core_client import CoreClient
from .memory_handlers import handle_memory as _handle_memory
from .memory_handlers import handle_prefs as _handle_prefs
from .memory_handlers import link_message_to_task as _link_message_to_task
from .memory_handlers import maybe_save_conversational_preferences as _maybe_save_conversational_preferences
from .memory_handlers import record_interaction_outcome as _record_interaction_outcome
from .memory_handlers import record_message as _record_message
from .memory_handlers import remember_task_reference as _remember_task_reference
from .models import BotResponse, DiscordBotConfig
from .notifier import is_notifiable_work_status as _is_notifiable_work_status
from .notifier import notify_work_changes as _notify_work_changes
from .notifier import resolve_notification_channel as _resolve_notification_channel
from .notifier import work_notification_loop as _work_notification_loop
from .routing import COMMAND_NAMES
from .routing import command_line_from_content as _command_line_from_content
from .views import build_view_factories


def run_discord_bot(config: DiscordBotConfig) -> None:
    try:
        import discord
    except ImportError as exc:
        raise RuntimeError("discord.py is required. Install with: pip install discord.py") from exc

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    core = CoreClient(config.core_url)
    view_factories = build_view_factories(discord=discord, core=core, allowed_user_ids=config.allowed_user_ids, activation_verify_note=_activation_verify_note)
    notify_task: asyncio.Task[Any] | None = None
    persistent_views_registered = False

    @client.event
    async def on_ready() -> None:
        nonlocal notify_task, persistent_views_registered
        print(f"Discord bot logged in as {client.user} | channel={config.channel_id} | core={config.core_url}", flush=True)
        if not persistent_views_registered:
            await _register_persistent_work_views(client, core, view_factories)
            persistent_views_registered = True
        if config.work_notify_enabled and notify_task is None:
            notify_task = asyncio.create_task(
                _work_notification_loop(
                    client,
                    core,
                    config,
                    view_factories.proposal,
                    view_factories.work,
                    view_factories.activation,
                    view_factories.retry,
                    view_factories.promote,
                )
            )

    @client.event
    async def on_message(message: Any) -> None:
        if message.author.bot:
            return
        if not _message_allowed(message, config):
            return
        content = message.content.strip()
        command_line = _command_line_from_content(content, config)
        if command_line is None:
            return
        if not command_line:
            from .commands import help_text

            await _reply(message, help_text(config.prefix))
            return
        user_id = _discord_user_id(message)
        channel_id = _discord_channel_id(message)
        await _record_message(core, message, role="user", content=content)
        try:
            response = await _handle_command(
                command_line,
                core,
                config,
                user_id=user_id,
                channel_id=channel_id,
                proposal_view_factory=view_factories.proposal,
                work_view_factory=view_factories.work,
                activation_view_factory=view_factories.activation,
                retry_view_factory=view_factories.retry,
                promote_view_factory=view_factories.promote,
            )
        except Exception as exc:  # Discord handlers should never crash the bot.
            response = f"실행 실패: `{type(exc).__name__}: {exc}`"
        response_text = response.text if isinstance(response, BotResponse) else str(response)
        task_id = response.task_id if isinstance(response, BotResponse) else None
        if task_id:
            await _link_message_to_task(core, message, role="user", task_id=task_id)
        await _record_message(core, message, role="assistant", content=response_text, linked_task_id=task_id)
        if task_id:
            await _record_interaction_outcome(
                core,
                message,
                task_id=task_id,
                request_text=content,
                response_text=response_text,
                metadata=response.metadata if isinstance(response, BotResponse) else None,
            )
        await _reply(message, response_text, view=response.view if isinstance(response, BotResponse) else None)

    client.run(config.token)


async def _register_persistent_work_views(client: Any, core: CoreClient, view_factories: Any) -> None:
    try:
        payload = await _call(core.get, "/work-items?limit=100")
    except Exception as exc:
        print(f"[discord-views] restore failed: {type(exc).__name__}: {exc}", flush=True)
        return
    items = payload.get("work_items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return
    registered: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        work_id = str(item.get("work_id") or "").strip()
        if not work_id:
            continue
        for kind, view in await _persistent_views_for_work_item(core, view_factories, item):
            key = (kind, work_id)
            if key in registered:
                continue
            client.add_view(view)
            registered.add(key)
    if registered:
        print(f"[discord-views] restored persistent views={len(registered)}", flush=True)


async def _persistent_views_for_work_item(core: CoreClient, view_factories: Any, item: dict[str, Any]) -> list[tuple[str, Any]]:
    work_id = str(item.get("work_id") or "").strip()
    status = str(item.get("status") or "")
    work_type = str(item.get("type") or "")
    views: list[tuple[str, Any]] = []
    if status == "proposed":
        views.append(("work", view_factories.work(work_id)))
        proposal_id = await _proposal_id_for_work(core, work_id)
        if proposal_id:
            views.append(("proposal", view_factories.proposal(proposal_id)))
    if status == "waiting_approval":
        views.append(("activation", view_factories.activation(work_id)))
    if status in {"reviewing", "blocked", "failed"}:
        views.append(("retry", view_factories.retry(work_id)))
    if status == "planned" and work_type == "external_work":
        views.append(("promote", view_factories.promote(work_id)))
    return views


async def _proposal_id_for_work(core: CoreClient, work_id: str) -> str | None:
    try:
        detail = await _call(core.get, f"/work-items/{quote(work_id)}")
    except Exception:
        return None
    proposal = detail.get("capability_proposal") if isinstance(detail, dict) and isinstance(detail.get("capability_proposal"), dict) else {}
    proposal_id = str(proposal.get("proposal_id") or "").strip()
    return proposal_id or None
