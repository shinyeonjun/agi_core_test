from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .bot_utils import call_blocking as _call
from .commands import format_code_block
from .core_client import CoreClient
from .formatters import format_work_item_response as _format_work_item_response
from .language_gateway import humanize
from .models import BotResponse
from .work_status import build_work_status_payload, first_promotable_work_id


async def handle_work(
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
        return await list_work(core, user_id=user_id, channel_id=channel_id, promote_view_factory=promote_view_factory)
    if command in {"show", "get"}:
        return await show_work(tail, core, activation_view_factory=activation_view_factory, retry_view_factory=retry_view_factory, promote_view_factory=promote_view_factory)
    if command == "promote":
        return await promote_work(tail, core)
    if command == "retry":
        return await retry_work(tail, core)
    if command == "activate":
        return await activate_work(tail, core)
    return "`work list`, `work show <id>`, `work promote <id>`, `work retry <id>`로 볼 수 있어."


async def list_work(core: CoreClient, *, user_id: str | None, channel_id: str | None, promote_view_factory: Any | None) -> str | BotResponse:
    payload = await _call(core.get, "/work-items?limit=10")
    jobs_payload = await _call(core.get, "/work-jobs?limit=10")
    items = payload.get("work_items") if isinstance(payload, dict) else []
    jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else []
    status_payload = build_work_status_payload(items if isinstance(items, list) else [], jobs if isinstance(jobs, list) else [])
    text = await humanize(core, status_payload, user_id=user_id, channel_id=channel_id)
    promotable = first_promotable_work_id(status_payload)
    if promotable and promote_view_factory:
        return BotResponse(text, view=promote_view_factory(promotable))
    return text


async def show_work(
    raw_work_id: str,
    core: CoreClient,
    *,
    activation_view_factory: Any | None,
    retry_view_factory: Any | None,
    promote_view_factory: Any | None,
) -> str | BotResponse:
    work_id = raw_work_id.split()[0] if raw_work_id else ""
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


async def promote_work(raw_work_id: str, core: CoreClient) -> str:
    work_id = raw_work_id.split()[0] if raw_work_id else ""
    if not work_id:
        return "개발 작업으로 전환할 작업 id를 붙여줘."
    payload = await _call(core.post, f"/work-items/{quote(work_id)}/promote-self-patch", {"actor": "discord"})
    if isinstance(payload, dict) and payload.get("child_work_item"):
        child = payload.get("child_work_item") if isinstance(payload.get("child_work_item"), dict) else {}
        queue = payload.get("queue") if isinstance(payload.get("queue"), dict) else {}
        queued = "큐에 들어갔어" if queue.get("queued") else f"큐 대기 실패: {queue.get('reason') or 'unknown'}"
        return f"개발 작업으로 전환했어: {child.get('title') or work_id}\nchild: `{child.get('work_id')}`\n{queued}"
    return format_code_block(payload)


async def retry_work(raw_work_id: str, core: CoreClient) -> str:
    work_id = raw_work_id.split()[0] if raw_work_id else ""
    if not work_id:
        return "재시도할 작업 id를 붙여줘."
    payload = await _call(core.post, f"/work-items/{quote(work_id)}/retry", {"actor": "discord"})
    if isinstance(payload, dict) and payload.get("queued"):
        job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
        return f"수정 재시도를 시작했어.\njob: `{job.get('job_id') or 'unknown'}`"
    return format_code_block(payload)


async def activate_work(raw_work_id: str, core: CoreClient) -> str:
    work_id = raw_work_id.split()[0] if raw_work_id else ""
    if not work_id:
        return "장착할 작업 id를 붙여줘."
    payload = await _call(core.post, f"/work-items/{quote(work_id)}/activate", {"actor": "discord"})
    if isinstance(payload, dict) and payload.get("activated"):
        reload_note = "\n서비스 재시작이 필요해." if payload.get("service_reload_required") else ""
        return f"장착 완료: `{payload.get('action_id') or work_id}`{activation_verify_note(payload)}{reload_note}"
    return format_code_block(payload)


def activation_verify_note(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    if not verification:
        return ""
    if verification.get("passed"):
        return "\n검증: catalog 등록 + smoke-test 통과"
    return "\n검증: 실패"
