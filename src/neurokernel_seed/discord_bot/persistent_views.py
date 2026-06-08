from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .bot_utils import call_blocking as _call
from .core_client import CoreClient


async def register_persistent_work_views(client: Any, core: CoreClient, view_factories: Any) -> None:
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
        for kind, view in await persistent_views_for_work_item(core, view_factories, item):
            key = (kind, work_id)
            if key in registered:
                continue
            client.add_view(view)
            registered.add(key)
    if registered:
        print(f"[discord-views] restored persistent views={len(registered)}", flush=True)


async def persistent_views_for_work_item(core: CoreClient, view_factories: Any, item: dict[str, Any]) -> list[tuple[str, Any]]:
    work_id = str(item.get("work_id") or "").strip()
    status = str(item.get("status") or "")
    work_type = str(item.get("type") or "")
    views: list[tuple[str, Any]] = []
    if status == "proposed":
        views.append(("work", view_factories.work(work_id)))
        proposal_id = await proposal_id_for_work(core, work_id)
        if proposal_id:
            views.append(("proposal", view_factories.proposal(proposal_id)))
    if status == "waiting_approval":
        views.append(("activation", view_factories.activation(work_id)))
    if status in {"reviewing", "blocked", "failed"}:
        views.append(("retry", view_factories.retry(work_id)))
    if status == "planned" and work_type == "external_work":
        views.append(("promote", view_factories.promote(work_id)))
    return views


async def proposal_id_for_work(core: CoreClient, work_id: str) -> str | None:
    try:
        detail = await _call(core.get, f"/work-items/{quote(work_id)}")
    except Exception:
        return None
    proposal = detail.get("capability_proposal") if isinstance(detail, dict) and isinstance(detail.get("capability_proposal"), dict) else {}
    proposal_id = str(proposal.get("proposal_id") or "").strip()
    return proposal_id or None
