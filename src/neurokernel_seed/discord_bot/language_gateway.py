from __future__ import annotations

from typing import Any

from .bot_utils import call_blocking as _call
from .core_client import CoreClient


async def language_to_core(core: CoreClient, user_text: str, *, user_id: str | None = None, channel_id: str | None = None) -> dict[str, Any]:
    payload = await _call(core.post, "/language/to-core", {"user_text": user_text, "context": {"source": "discord", "user_id": user_id, "channel_id": channel_id}})
    if not isinstance(payload, dict):
        raise RuntimeError("language-to-core returned non-object payload")
    return payload


def required_text(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"{label} is empty")
    return text


async def humanize(core: CoreClient, payload: Any, *, user_id: str | None = None, channel_id: str | None = None) -> str:
    if not isinstance(payload, dict):
        raise RuntimeError("language-to-human requires object payload")
    result = await _call(core.post, "/language/to-human", {"core_result": payload, "style": "ko_short_no_internal", "context": {"user_id": user_id, "channel_id": channel_id}})
    if isinstance(result, dict) and result.get("reply"):
        return str(result["reply"])
    raise RuntimeError("language-to-human returned no reply")
