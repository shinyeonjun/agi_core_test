from __future__ import annotations

import asyncio
import os
from urllib.parse import quote
from dataclasses import dataclass
from typing import Any

from .bot_utils import call_blocking as _call
from .bot_utils import discord_channel_id as _discord_channel_id
from .bot_utils import discord_chunks as _discord_chunks
from .bot_utils import discord_user_id as _discord_user_id
from .bot_utils import format_pref_value as _format_pref_value
from .bot_utils import is_auto_executable_task as _is_auto_executable_task
from .bot_utils import message_allowed as _message_allowed
from .bot_utils import parse_int as _parse_int
from .bot_utils import parse_pref_value as _parse_pref_value
from .bot_utils import parse_user_ids as _parse_user_ids
from .bot_utils import reply as _reply
from .bot_utils import split_id_reason as _split_id_reason
from .commands import build_benchmark_task, build_preset_task, format_code_block, help_text, parse_task_json, summarize_status
from .core_client import CoreClient
from .formatters import format_capability_proposal_response as _format_capability_proposal_response
from .formatters import format_work_item_response as _format_work_item_response
from .views import build_view_factories
from .work_status import build_work_status_payload as _build_work_status_payload
from .work_status import first_promotable_work_id as _first_promotable_work_id
from .work_status import format_work_notification as _format_work_notification


@dataclass(frozen=True)
class DiscordBotConfig:
    token: str
    channel_id: int
    core_url: str
    prefix: str
    allow_dms: bool = False
    allowed_user_ids: tuple[int, ...] = ()
    reply_without_prefix: bool = False
    auto_do_low_risk: bool = False
    work_notify_enabled: bool = True
    work_notify_interval_seconds: float = 10.0


@dataclass(frozen=True)
class BotResponse:
    text: str
    view: Any | None = None


COMMAND_NAMES = {
    "help",
    "도움말",
    "?",
    "ping",
    "status",
    "actions",
    "prefs",
    "memory",
    "work",
    "run",
    "benchmark",
    "ask",
    "auto",
    "plan",
    "do",
    "task",
    "run-task",
    "runtask",
    "approve",
    "reject",
    "trace",
}


def config_from_env(
    *,
    token_env: str = "DISCORD_BOT_TOKEN",
    channel_id: int | None = None,
    core_url: str | None = None,
    prefix: str | None = None,
    allow_dms: bool | None = None,
    allowed_user_ids: tuple[int, ...] | None = None,
    reply_without_prefix: bool | None = None,
    auto_do_low_risk: bool | None = None,
    work_notify_enabled: bool | None = None,
    work_notify_interval_seconds: float | None = None,
) -> DiscordBotConfig:
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"{token_env} is required")
    resolved_channel_id = channel_id
    if resolved_channel_id is None:
        resolved_channel_id = int(_required_env("DISCORD_CHANNEL_ID"))
    resolved_allow_dms = allow_dms
    if resolved_allow_dms is None:
        resolved_allow_dms = _required_bool_env("DISCORD_ALLOW_DMS")
    resolved_allowed_user_ids = allowed_user_ids
    if resolved_allowed_user_ids is None:
        resolved_allowed_user_ids = _parse_user_ids(_required_env("DISCORD_ALLOWED_USER_IDS"))
        if not resolved_allowed_user_ids:
            raise RuntimeError("DISCORD_ALLOWED_USER_IDS must contain at least one user id")
    resolved_reply_without_prefix = reply_without_prefix
    if resolved_reply_without_prefix is None:
        resolved_reply_without_prefix = _required_bool_env("NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX")
    resolved_auto_do_low_risk = auto_do_low_risk
    if resolved_auto_do_low_risk is None:
        resolved_auto_do_low_risk = _required_bool_env("NEUROKERNEL_BOT_AUTO_DO_LOW_RISK")
    resolved_work_notify_enabled = work_notify_enabled
    if resolved_work_notify_enabled is None:
        resolved_work_notify_enabled = _optional_bool_env("NEUROKERNEL_BOT_WORK_NOTIFY_ENABLED", default=True)
    resolved_work_notify_interval = work_notify_interval_seconds
    if resolved_work_notify_interval is None:
        resolved_work_notify_interval = float(os.environ.get("NEUROKERNEL_BOT_WORK_NOTIFY_INTERVAL", "10"))
    return DiscordBotConfig(
        token=token,
        channel_id=resolved_channel_id,
        core_url=core_url if core_url is not None else _required_env("NEUROKERNEL_CORE_URL"),
        prefix=prefix if prefix is not None else _required_env_defined("NEUROKERNEL_BOT_PREFIX"),
        allow_dms=resolved_allow_dms,
        allowed_user_ids=resolved_allowed_user_ids,
        reply_without_prefix=resolved_reply_without_prefix,
        auto_do_low_risk=resolved_auto_do_low_risk,
        work_notify_enabled=resolved_work_notify_enabled,
        work_notify_interval_seconds=resolved_work_notify_interval,
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _required_env_defined(name: str) -> str:
    if name not in os.environ:
        raise RuntimeError(f"{name} is required")
    return os.environ[name]


def _required_bool_env(name: str) -> bool:
    value = _required_env(name).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def _optional_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


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

    @client.event
    async def on_ready() -> None:
        nonlocal notify_task
        print(f"Discord bot logged in as {client.user} | channel={config.channel_id} | core={config.core_url}", flush=True)
        if config.work_notify_enabled and notify_task is None:
            notify_task = asyncio.create_task(_work_notification_loop(client, core, config, view_factories.activation, view_factories.retry, view_factories.promote))

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
            await _reply(message, help_text(config.prefix))
            return
        user_id = _discord_user_id(message)
        channel_id = _discord_channel_id(message)
        await _record_message(core, message, role="user", content=content)
        try:
            response = await _handle_command(command_line, core, config, user_id=user_id, channel_id=channel_id, proposal_view_factory=view_factories.proposal, work_view_factory=view_factories.work, activation_view_factory=view_factories.activation, retry_view_factory=view_factories.retry, promote_view_factory=view_factories.promote)
        except Exception as exc:  # Discord handlers should never crash the bot.
            response = f"실행 실패: `{type(exc).__name__}: {exc}`"
        response_text = response.text if isinstance(response, BotResponse) else str(response)
        await _record_message(core, message, role="assistant", content=response_text)
        await _reply(message, response_text, view=response.view if isinstance(response, BotResponse) else None)

    client.run(config.token)


async def _work_notification_loop(
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
                await _notify_work_changes(client, core, config, activation_view_factory, retry_view_factory, promote_view_factory, items, seen=seen, first_poll=first_poll)
            first_poll = False
        except Exception as exc:
            print(f"[discord-work-notifier] poll failed: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(max(1.0, float(config.work_notify_interval_seconds)))


async def _notify_work_changes(
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
        if first_poll or previous == signature or not _is_notifiable_work_status(status):
            continue
        detail = await _call(core.get, f"/work-items/{quote(work_id)}")
        if not isinstance(detail, dict):
            continue
        text, view_kind = _format_work_notification(detail)
        channel = await _resolve_notification_channel(client, item, config)
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


def _is_notifiable_work_status(status: str) -> bool:
    return status in {"planned", "waiting_approval", "reviewing", "blocked", "failed", "completed"}


async def _resolve_notification_channel(client: Any, item: dict[str, Any], config: DiscordBotConfig) -> Any | None:
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


def _command_line_from_content(content: str, config: DiscordBotConfig) -> str | None:
    content = content.strip()
    if not content:
        return ""
    if config.prefix:
        if not content.startswith(config.prefix):
            if not config.reply_without_prefix:
                return None
            return f"auto {content}"
        return content[len(config.prefix) :].strip()
    if not config.reply_without_prefix:
        return None
    first, _, _ = content.partition(" ")
    if first.lower().strip() in COMMAND_NAMES:
        return content
    return f"auto {content}"


async def _handle_command(
    command_line: str,
    core: CoreClient,
    config: DiscordBotConfig,
    *,
    user_id: str | None = None,
    channel_id: str | None = None,
    proposal_view_factory: Any | None = None,
    work_view_factory: Any | None = None,
    activation_view_factory: Any | None = None,
    retry_view_factory: Any | None = None,
    promote_view_factory: Any | None = None,
) -> str | BotResponse:
    command, _, rest = command_line.partition(" ")
    command = command.lower().strip()
    rest = rest.strip()
    if command in {"help", "도움말", "?"}:
        return help_text(config.prefix)
    if command == "ping":
        payload = await _call(core.get, "/health")
        return await _humanize(core, payload)
    if command == "status":
        payload = await _call(core.get, "/status")
        if isinstance(payload, dict):
            return await _humanize(core, payload)
        return format_code_block(payload)
    if command == "actions":
        payload = await _call(core.get, "/actions")
        if isinstance(payload, list):
            names = [f"- `{item.get('action_id')}` / risk={item.get('risk_level')} / approval={item.get('requires_approval')}" for item in payload[:20]]
            return "허용된 액션 목록\n" + "\n".join(names)
        return format_code_block(payload)
    if command == "prefs":
        return await _handle_prefs(rest, core, user_id=user_id)
    if command == "memory":
        return await _handle_memory(rest, core, user_id=user_id, channel_id=channel_id)
    if command == "work":
        return await _handle_work(rest, core, user_id=user_id, channel_id=channel_id, activation_view_factory=activation_view_factory, retry_view_factory=retry_view_factory, promote_view_factory=promote_view_factory)
    if command == "run":
        return await _run_preset(rest, core, user_id=user_id, channel_id=channel_id)
    if command == "benchmark":
        episodes = _parse_int(rest, default=3, minimum=1, maximum=50)
        task = build_benchmark_task(episodes=episodes)
        return await _create_and_run(task, core, user_id=user_id, channel_id=channel_id)
    if command == "ask":
        if not rest:
            return "무엇을 물어볼지 뒤에 적어줘."
        payload = await _language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
        return _required_text(payload.get("reply"), "language reply")
    if command == "auto":
        if not rest:
            return ""
        preference_reply = await _maybe_save_conversational_preferences(rest, core, user_id=user_id)
        if preference_reply:
            return preference_reply
        payload = await _language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
        route_response = await _maybe_route_work(
            rest,
            payload,
            core,
            user_id=user_id,
            channel_id=channel_id,
            proposal_view_factory=proposal_view_factory,
            work_view_factory=work_view_factory,
            activation_view_factory=activation_view_factory,
            retry_view_factory=retry_view_factory,
            promote_view_factory=promote_view_factory,
        )
        if route_response:
            return route_response
        task = payload.get("task_spec") if isinstance(payload, dict) else None
        if config.auto_do_low_risk and _is_auto_executable_task(task):
            return await _create_and_run(task, core, user_id=user_id, channel_id=channel_id)
        if isinstance(task, dict) and not config.auto_do_low_risk:
            reply = _required_text(payload.get("reply"), "language reply")
            return f"{reply}\n실행하려면 `do {rest}`라고 말해줘."
        return _required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")
    if command == "plan":
        if not rest:
            return "어떤 일을 계획할지 뒤에 적어줘."
        payload = await _language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
        task = payload.get("task_spec") if isinstance(payload, dict) else None
        if not isinstance(task, dict):
            return _required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")
        created = await _call(core.post, "/tasks", task)
        task_id = created.get("task", {}).get("task_id") if isinstance(created, dict) else None
        dry_run = await _call(core.post, f"/tasks/{task_id}/dry-run", {}) if task_id else {}
        chosen = dry_run.get("chosen", {}).get("safety", {}).get("decision") if isinstance(dry_run, dict) else None
        if chosen in {"allow", "dry_run_only"}:
            reply = _required_text(payload.get("reply"), "language reply")
            return f"{reply}\n안전 검사까지 통과했어. 실행하려면 `do {rest}`라고 말해줘."
        reply = _required_text(payload.get("reply"), "language reply")
        return f"{reply}\n다만 바로 실행하기엔 확인이 더 필요해."
    if command == "do":
        if not rest:
            return "무엇을 실행할지 뒤에 적어줘."
        payload = await _language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
        task = payload.get("task_spec") if isinstance(payload, dict) else None
        if not isinstance(task, dict):
            return _required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")
        return await _create_and_run(task, core, user_id=user_id, channel_id=channel_id)
    if command == "task":
        task = parse_task_json(rest)
        created = await _call(core.post, "/tasks", task)
        if not isinstance(created, dict):
            return format_code_block(created)
        task_id = created.get("task", {}).get("task_id")
        dry_run = await _call(core.post, f"/tasks/{task_id}/dry-run", {})
        return "작업을 만들고 미리 검사했어.\n" + format_code_block(dry_run)
    if command in {"run-task", "runtask"}:
        task_id = rest.split()[0] if rest else ""
        if not task_id:
            return "task_id가 필요해."
        payload = await _call(core.post, f"/tasks/{task_id}/run", {})
        return await _humanize(core, payload, user_id=user_id, channel_id=channel_id)
    if command == "approve":
        task_id, reason = _split_id_reason(rest)
        payload = await _call(core.post, f"/tasks/{task_id}/approve", {"approved_by": "discord", "reason": reason})
        return "승인 처리했어."
    if command == "reject":
        task_id, reason = _split_id_reason(rest)
        payload = await _call(core.post, f"/tasks/{task_id}/reject", {"rejected_by": "discord", "reason": reason})
        return "거절 처리했어."
    if command == "trace":
        limit = _parse_int(rest, default=5, minimum=1, maximum=20)
        payload = await _call(core.get, f"/trace/recent?limit={limit}")
        return "최근 trace\n" + format_code_block(payload)
    return f"모르는 명령이야. `help`로 목록을 볼 수 있어."


async def _run_preset(name: str, core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not name:
        return "`uptime`, `disk`, `memory`, `temp`, `artifacts`, `trace` 중 하나를 붙여줘."
    task = build_preset_task(name)
    return await _create_and_run(task, core, user_id=user_id, channel_id=channel_id)


async def _create_and_run(task: dict[str, Any], core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    created = await _call(core.post, "/tasks", task)
    if not isinstance(created, dict):
        return format_code_block(created)
    task_id = created.get("task", {}).get("task_id")
    if not task_id:
        return "Core가 task_id를 반환하지 않았어.\n" + format_code_block(created)
    payload = await _call(core.post, f"/tasks/{task_id}/run", {})
    if user_id:
        await _remember_task_reference(core, user_id=user_id, task_id=str(task_id), task=task, payload=payload)
    return await _humanize(core, payload, user_id=user_id, channel_id=channel_id)


async def _language_to_core(core: CoreClient, user_text: str, *, user_id: str | None = None, channel_id: str | None = None) -> dict[str, Any]:
    payload = await _call(core.post, "/language/to-core", {"user_text": user_text, "context": {"source": "discord", "user_id": user_id, "channel_id": channel_id}})
    if not isinstance(payload, dict):
        raise RuntimeError("language-to-core returned non-object payload")
    return payload


def _required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"{label} is empty")
    return text


async def _humanize(core: CoreClient, payload: Any, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("language-to-human requires object payload")
    result = await _call(core.post, "/language/to-human", {"core_result": payload, "style": "ko_short_no_internal", "context": {"user_id": user_id, "channel_id": channel_id}})
    if isinstance(result, dict) and result.get("reply"):
        return str(result["reply"])
    raise RuntimeError("language-to-human returned no reply")


async def _maybe_create_capability_proposal(
    text: str,
    language_payload: dict[str, Any],
    core: CoreClient,
    *,
    user_id: str | None,
    channel_id: str | None,
    proposal_view_factory: Any | None,
) -> BotResponse | None:
    payload = await _call(
        core.post,
        "/capability-proposals/from-request",
        {
            "user_text": text,
            "language_intent": language_payload,
            "context": {"source": "discord", "user_id": user_id, "channel_id": channel_id},
        },
    )
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "none")
    if kind not in {"gap", "duplicate", "ambiguous", "forbidden"}:
        return None
    if kind in {"ambiguous", "forbidden"}:
        return BotResponse(_required_text(payload.get("reply"), "capability reply"))
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
    if not proposal:
        return None
    text_reply = _format_capability_proposal_response(payload)
    view = proposal_view_factory(str(proposal.get("proposal_id"))) if proposal_view_factory and kind == "gap" else None
    return BotResponse(text_reply, view=view)


async def _maybe_route_work(
    text: str,
    language_payload: dict[str, Any],
    core: CoreClient,
    *,
    user_id: str | None,
    channel_id: str | None,
    proposal_view_factory: Any | None,
    work_view_factory: Any | None,
    activation_view_factory: Any | None = None,
    retry_view_factory: Any | None = None,
    promote_view_factory: Any | None = None,
) -> BotResponse | None:
    payload = await _call(
        core.post,
        "/work/route",
        {
            "user_text": text,
            "language_intent": language_payload,
            "context": {"source": "discord", "user_id": user_id, "channel_id": channel_id},
        },
    )
    if not isinstance(payload, dict):
        return None
    route = str(payload.get("route") or payload.get("route_decision", {}).get("route") or "clarify")
    if route == "work_status":
        response = await _handle_work("list", core, user_id=user_id, channel_id=channel_id, activation_view_factory=activation_view_factory, retry_view_factory=retry_view_factory, promote_view_factory=promote_view_factory)
        return response if isinstance(response, BotResponse) else BotResponse(response)
    if route == "runtime_task":
        return None
    if route in {"unsafe", "clarify"}:
        decision = payload.get("route_decision") if isinstance(payload.get("route_decision"), dict) else {}
        question = decision.get("clarifying_question")
        if question:
            return BotResponse(str(question))
        if route == "unsafe":
            return BotResponse("그건 바로 진행하기 위험해서 작업으로 올리지 않을게.")
        return None
    if route == "self_patch" and payload.get("kind") in {"gap", "duplicate", "ambiguous", "forbidden"}:
        if payload.get("kind") in {"ambiguous", "forbidden"}:
            return BotResponse(_required_text(payload.get("reply"), "capability reply"))
        proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
        if not proposal:
            return None
        view = proposal_view_factory(str(proposal.get("proposal_id"))) if proposal_view_factory and payload.get("kind") == "gap" else None
        return BotResponse(_format_capability_proposal_response(payload), view=view)
    if route == "external_work" and payload.get("created"):
        work_item = payload.get("work_item") if isinstance(payload.get("work_item"), dict) else {}
        if not work_item:
            return None
        view = work_view_factory(str(work_item.get("work_id"))) if work_view_factory else None
        return BotResponse(_format_work_item_response(work_item, payload), view=view)
    return None




async def _handle_prefs(rest: str, core: CoreClient, *, user_id: str | None) -> str:
    if not user_id:
        return "사용자 ID를 확인하지 못해서 선호를 저장하지 못했어."
    command, _, tail = rest.partition(" ")
    command = command.lower().strip() or "show"
    tail = tail.strip()
    if command in {"show", "list"}:
        payload = await _call(core.get, f"/memory/preferences?user_id={quote(user_id)}")
        prefs = payload.get("preferences") if isinstance(payload, dict) else []
        if not prefs:
            return "아직 저장된 선호가 없어."
        lines = ["저장된 선호"]
        for item in prefs[:20]:
            lines.append(f"- {item.get('key')}: {_format_pref_value(item.get('value_json'))}")
        return "\n".join(lines)
    if command == "set":
        key, _, raw_value = tail.partition(" ")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            return "이렇게 말해줘: `prefs set response_length short`"
        value = _parse_pref_value(raw_value)
        await _call(core.post, "/memory/preferences", {"user_id": user_id, "key": key, "value": value})
        return f"저장했어. 앞으로 `{key}`는 `{_format_pref_value(value)}`로 볼게."
    if command in {"forget", "delete"}:
        key = tail.strip()
        if not key:
            return "지울 선호 key를 적어줘. 예: `prefs forget response_length`"
        payload = await _call(core.post, "/memory/preferences/delete", {"user_id": user_id, "key": key})
        return f"지웠어. 삭제한 항목: {payload.get('deleted', 0) if isinstance(payload, dict) else 0}개"
    if command == "reset":
        payload = await _call(core.post, "/memory/preferences/delete", {"user_id": user_id})
        return f"선호를 초기화했어. 삭제한 항목: {payload.get('deleted', 0) if isinstance(payload, dict) else 0}개"
    return "`prefs show`, `prefs set <key> <value>`, `prefs forget <key>`, `prefs reset` 중 하나로 말해줘."


async def _maybe_save_conversational_preferences(text: str, core: CoreClient, *, user_id: str | None) -> str | None:
    if not user_id:
        return None
    payload = await _call(core.post, "/language/preferences", {"user_text": text, "context": {"source": "discord", "user_id": user_id}})
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") not in {"preference_update", "reject"}:
        return None
    if payload.get("kind") == "reject":
        return _required_text(payload.get("reply"), "preference reply")
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    if not candidates:
        return None
    for candidate in candidates:
        await _call(
            core.post,
            "/memory/preferences",
            {
                "user_id": user_id,
                "key": candidate.get("key"),
                "value": candidate.get("value"),
                "scope": candidate.get("scope") or "global",
                "source": candidate.get("source") or "explicit_user_request",
                "confidence": candidate.get("confidence") or payload.get("confidence") or 0.0,
            },
        )
    if len(candidates) == 1:
        return _required_text(payload.get("reply"), "preference reply")
    keys = ", ".join(str(candidate.get("key")) for candidate in candidates)
    reply = _required_text(payload.get("reply"), "preference reply")
    return f"{reply} ({keys})"


async def _handle_memory(rest: str, core: CoreClient, *, user_id: str | None, channel_id: str | None) -> str:
    if not user_id:
        return "사용자 ID를 확인하지 못해서 기억을 조회하지 못했어."
    command, _, tail = rest.partition(" ")
    command = command.lower().strip() or "recent"
    if command == "recent":
        limit = _parse_int(tail.strip(), default=8, minimum=1, maximum=20)
        payload = await _call(core.get, f"/memory/recent?user_id={quote(user_id)}&channel_id={quote(channel_id or '')}&limit={limit}")
        messages = payload.get("messages") if isinstance(payload, dict) else []
        if not messages:
            return "최근 대화 기억이 아직 없어."
        lines = ["최근 대화 기억"]
        for item in messages[-limit:]:
            role = "나" if item.get("role") == "assistant" else "사용자"
            lines.append(f"- {role}: {str(item.get('content_redacted') or '')[:120]}")
        return "\n".join(lines)
    if command == "context":
        payload = await _call(core.get, f"/memory/context?user_id={quote(user_id)}&channel_id={quote(channel_id or '')}")
        return format_code_block(payload, max_chars=1800)
    return "`memory recent` 또는 `memory context`로 말해줘."


async def _handle_work(
    rest: str,
    core: CoreClient,
    *,
    user_id: str | None = None,
    channel_id: str | None = None,
    activation_view_factory: Any | None = None,
    retry_view_factory: Any | None = None,
    promote_view_factory: Any | None = None,
) -> str | BotResponse:
    command, _, tail = rest.partition(" ")
    command = command.lower().strip() or "list"
    tail = tail.strip()
    if command in {"list", "ls"}:
        payload = await _call(core.get, "/work-items?limit=10")
        jobs_payload = await _call(core.get, "/work-jobs?limit=10")
        items = payload.get("work_items") if isinstance(payload, dict) else []
        jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else []
        status_payload = _build_work_status_payload(items if isinstance(items, list) else [], jobs if isinstance(jobs, list) else [])
        text = await _humanize(
            core,
            status_payload,
            user_id=user_id,
            channel_id=channel_id,
        )
        promotable = _first_promotable_work_id(status_payload)
        if promotable and promote_view_factory:
            return BotResponse(text, view=promote_view_factory(promotable))
        return text
    if command in {"show", "get"}:
        work_id = tail.split()[0] if tail else ""
        if not work_id:
            return "볼 작업 id를 붙여줘."
        payload = await _call(core.get, f"/work-items/{quote(work_id)}")
        item = payload.get("work_item") if isinstance(payload, dict) else {}
        if not item:
            return "그 작업을 못 찾았어."
        text = _format_work_item_response(item, payload if isinstance(payload, dict) else {})
        if item.get("type") == "external_work" and item.get("status") == "planned" and promote_view_factory:
            return BotResponse(text, view=promote_view_factory(str(item.get("work_id"))))
        if item.get("status") == "waiting_approval" and activation_view_factory:
            return BotResponse(text, view=activation_view_factory(str(item.get("work_id"))))
        if item.get("status") in {"reviewing", "blocked", "failed"} and retry_view_factory:
            return BotResponse(text, view=retry_view_factory(str(item.get("work_id"))))
        return text
    if command == "promote":
        work_id = tail.split()[0] if tail else ""
        if not work_id:
            return "개발 작업으로 전환할 작업 id를 붙여줘."
        payload = await _call(core.post, f"/work-items/{quote(work_id)}/promote-self-patch", {"actor": "discord"})
        if isinstance(payload, dict) and payload.get("child_work_item"):
            child = payload.get("child_work_item") if isinstance(payload.get("child_work_item"), dict) else {}
            queue = payload.get("queue") if isinstance(payload.get("queue"), dict) else {}
            queued = "큐에 들어갔어" if queue.get("queued") else f"큐 대기 실패: {queue.get('reason') or 'unknown'}"
            return f"개발 작업으로 전환했어: {child.get('title') or work_id}\nchild: `{child.get('work_id')}`\n{queued}"
        return format_code_block(payload)
    if command == "retry":
        work_id = tail.split()[0] if tail else ""
        if not work_id:
            return "재시도할 작업 id를 붙여줘."
        payload = await _call(core.post, f"/work-items/{quote(work_id)}/retry", {"actor": "discord"})
        if isinstance(payload, dict) and payload.get("queued"):
            job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
            return f"수정 재시도를 시작했어.\njob: `{job.get('job_id') or 'unknown'}`"
        return format_code_block(payload)
    if command == "activate":
        work_id = tail.split()[0] if tail else ""
        if not work_id:
            return "장착할 작업 id를 붙여줘."
        payload = await _call(core.post, f"/work-items/{quote(work_id)}/activate", {"actor": "discord"})
        if isinstance(payload, dict) and payload.get("activated"):
            reload_note = "\n서비스 재시작이 필요해." if payload.get("service_reload_required") else ""
            return f"장착 완료: `{payload.get('action_id') or work_id}`{_activation_verify_note(payload)}{reload_note}"
        return format_code_block(payload)
    return "`work list`, `work show <id>`, `work promote <id>`, `work retry <id>`로 볼 수 있어."


def _activation_verify_note(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    if not verification:
        return ""
    if verification.get("passed"):
        return "\n검증: catalog 등록 + smoke-test 통과"
    return "\n검증: 실패"


async def _record_message(core: CoreClient, message: Any, *, role: str, content: str) -> None:
    try:
        await _call(
            core.post,
            "/memory/messages",
            {
                "user_id": _discord_user_id(message),
                "channel_id": _discord_channel_id(message),
                "message_id": str(getattr(message, "id", "")) or None,
                "role": role,
                "content": content,
            },
        )
    except Exception:
        return


async def _remember_task_reference(core: CoreClient, *, user_id: str, task_id: str, task: dict[str, Any], payload: Any) -> None:
    try:
        status = str(payload.get("status") or "unknown") if isinstance(payload, dict) else "unknown"
        result_summary = ""
        if isinstance(payload, dict):
            action = payload.get("action") or ",".join(task.get("allowed_actions") or [])
            result_summary = f"{action} {status}"
        await _call(
            core.post,
            "/memory/task-references",
            {
                "user_id": user_id,
                "task_id": task_id,
                "short_label": str(task.get("goal") or task_id),
                "status": status,
                "result_summary": result_summary,
            },
        )
    except Exception:
        return

