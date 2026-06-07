from __future__ import annotations

from typing import Any

from .bot_utils import call_blocking as _call
from .bot_utils import is_auto_executable_task as _is_auto_executable_task
from .bot_utils import parse_int as _parse_int
from .bot_utils import split_id_reason as _split_id_reason
from .commands import build_benchmark_task, format_code_block, help_text, parse_task_json
from .core_client import CoreClient
from .formatters import format_capability_proposal_response as _format_capability_proposal_response
from .formatters import format_work_item_response as _format_work_item_response
from .language_gateway import humanize, language_to_core, required_text
from .memory_handlers import handle_memory, handle_prefs, maybe_save_conversational_preferences
from .models import BotResponse, DiscordBotConfig
from .task_commands import create_and_run, handle_do, handle_plan, run_preset
from .work_commands import activation_verify_note, handle_work


async def handle_command(
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
        return await humanize(core, payload)
    if command == "status":
        payload = await _call(core.get, "/status")
        if isinstance(payload, dict):
            return await humanize(core, payload)
        return format_code_block(payload)
    if command == "actions":
        payload = await _call(core.get, "/actions")
        if isinstance(payload, list):
            names = [f"- `{item.get('action_id')}` / risk={item.get('risk_level')} / approval={item.get('requires_approval')}" for item in payload[:20]]
            return "허용된 액션 목록\n" + "\n".join(names)
        return format_code_block(payload)
    if command == "prefs":
        return await handle_prefs(rest, core, user_id=user_id)
    if command == "memory":
        return await handle_memory(rest, core, user_id=user_id, channel_id=channel_id)
    if command == "work":
        return await handle_work(rest, core, user_id=user_id, channel_id=channel_id, activation_view_factory=activation_view_factory, retry_view_factory=retry_view_factory, promote_view_factory=promote_view_factory)
    if command == "run":
        return await run_preset(rest, core, user_id=user_id, channel_id=channel_id)
    if command == "benchmark":
        episodes = _parse_int(rest, default=3, minimum=1, maximum=50)
        task = build_benchmark_task(episodes=episodes)
        return await create_and_run(task, core, user_id=user_id, channel_id=channel_id)
    if command == "ask":
        if not rest:
            return "무엇을 물어볼지 뒤에 적어줘."
        payload = await language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
        return required_text(payload.get("reply"), "language reply")
    if command == "auto":
        return await handle_auto(
            rest,
            core,
            config,
            user_id=user_id,
            channel_id=channel_id,
            proposal_view_factory=proposal_view_factory,
            work_view_factory=work_view_factory,
            activation_view_factory=activation_view_factory,
            retry_view_factory=retry_view_factory,
            promote_view_factory=promote_view_factory,
        )
    if command == "plan":
        return await handle_plan(rest, core, user_id=user_id, channel_id=channel_id)
    if command == "do":
        return await handle_do(rest, core, user_id=user_id, channel_id=channel_id)
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
        return await humanize(core, payload, user_id=user_id, channel_id=channel_id)
    if command == "approve":
        task_id, reason = _split_id_reason(rest)
        await _call(core.post, f"/tasks/{task_id}/approve", {"approved_by": "discord", "reason": reason})
        return "승인 처리했어."
    if command == "reject":
        task_id, reason = _split_id_reason(rest)
        await _call(core.post, f"/tasks/{task_id}/reject", {"rejected_by": "discord", "reason": reason})
        return "거절 처리했어."
    if command == "trace":
        limit = _parse_int(rest, default=5, minimum=1, maximum=20)
        payload = await _call(core.get, f"/trace/recent?limit={limit}")
        return "최근 trace\n" + format_code_block(payload)
    return f"모르는 명령이야. `help`로 목록을 볼 수 있어."


async def handle_auto(
    rest: str,
    core: CoreClient,
    config: DiscordBotConfig,
    *,
    user_id: str | None,
    channel_id: str | None,
    proposal_view_factory: Any | None,
    work_view_factory: Any | None,
    activation_view_factory: Any | None,
    retry_view_factory: Any | None,
    promote_view_factory: Any | None,
) -> str | BotResponse:
    if not rest:
        return ""
    preference_reply = await maybe_save_conversational_preferences(rest, core, user_id=user_id)
    if preference_reply:
        return preference_reply
    payload = await language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
    route_response = await maybe_route_work(
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
        return await create_and_run(task, core, user_id=user_id, channel_id=channel_id)
    if isinstance(task, dict) and not config.auto_do_low_risk:
        reply = required_text(payload.get("reply"), "language reply")
        return f"{reply}\n실행하려면 `do {rest}`라고 말해줘."
    return required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")


async def maybe_create_capability_proposal(
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
        return BotResponse(required_text(payload.get("reply"), "capability reply"))
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
    if not proposal:
        return None
    text_reply = _format_capability_proposal_response(payload)
    view = proposal_view_factory(str(proposal.get("proposal_id"))) if proposal_view_factory and kind == "gap" else None
    return BotResponse(text_reply, view=view)


async def maybe_route_work(
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
        response = await handle_work("list", core, user_id=user_id, channel_id=channel_id, activation_view_factory=activation_view_factory, retry_view_factory=retry_view_factory, promote_view_factory=promote_view_factory)
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
            return BotResponse(required_text(payload.get("reply"), "capability reply"))
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
