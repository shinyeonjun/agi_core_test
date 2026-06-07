from __future__ import annotations

from typing import Any

from .bot_utils import call_blocking as _call
from .commands import build_preset_task, format_code_block
from .core_client import CoreClient
from .language_gateway import humanize, language_to_core, required_text
from .memory_handlers import remember_task_reference


async def run_preset(name: str, core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not name:
        return "`uptime`, `disk`, `memory`, `temp`, `artifacts`, `trace` 중 하나를 붙여줘."
    task = build_preset_task(name)
    return await create_and_run(task, core, user_id=user_id, channel_id=channel_id)


async def create_and_run(task: dict[str, Any], core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    created = await _call(core.post, "/tasks", task)
    if not isinstance(created, dict):
        return format_code_block(created)
    task_id = created.get("task", {}).get("task_id")
    if not task_id:
        return "Core가 task_id를 반환하지 않았어.\n" + format_code_block(created)
    payload = await _call(core.post, f"/tasks/{task_id}/run", {})
    if user_id:
        await remember_task_reference(core, user_id=user_id, task_id=str(task_id), task=task, payload=payload)
    return await humanize(core, payload, user_id=user_id, channel_id=channel_id)


async def handle_plan(rest: str, core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not rest:
        return "어떤 일을 계획할지 뒤에 적어줘."
    payload = await language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
    task = payload.get("task_spec") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")
    created = await _call(core.post, "/tasks", task)
    task_id = created.get("task", {}).get("task_id") if isinstance(created, dict) else None
    dry_run = await _call(core.post, f"/tasks/{task_id}/dry-run", {}) if task_id else {}
    chosen = dry_run.get("chosen", {}).get("safety", {}).get("decision") if isinstance(dry_run, dict) else None
    reply = required_text(payload.get("reply"), "language reply")
    if chosen in {"allow", "dry_run_only"}:
        return f"{reply}\n안전 검사까지 통과했어. 실행하려면 `do {rest}`라고 말해줘."
    return f"{reply}\n다만 바로 실행하기엔 확인이 더 필요해."


async def handle_do(rest: str, core: CoreClient, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not rest:
        return "무엇을 실행할지 뒤에 적어줘."
    payload = await language_to_core(core, rest, user_id=user_id, channel_id=channel_id)
    task = payload.get("task_spec") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return required_text(payload.get("clarifying_question") or payload.get("reply"), "language reply")
    return await create_and_run(task, core, user_id=user_id, channel_id=channel_id)
