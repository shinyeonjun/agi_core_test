from __future__ import annotations

from typing import Any
from urllib.parse import quote

from .bot_utils import call_blocking as _call
from .bot_utils import discord_channel_id as _discord_channel_id
from .bot_utils import discord_user_id as _discord_user_id
from .bot_utils import format_pref_value as _format_pref_value
from .bot_utils import parse_int as _parse_int
from .bot_utils import parse_pref_value as _parse_pref_value
from .commands import format_code_block
from .core_client import CoreClient
from .language_gateway import required_text


async def handle_prefs(rest: str, core: CoreClient, *, user_id: str | None) -> str:
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


async def maybe_save_conversational_preferences(text: str, core: CoreClient, *, user_id: str | None) -> str | None:
    if not user_id:
        return None
    payload = await _call(core.post, "/language/preferences", {"user_text": text, "context": {"source": "discord", "user_id": user_id}})
    if not isinstance(payload, dict):
        return None
    if payload.get("kind") not in {"preference_update", "reject"}:
        return None
    if payload.get("kind") == "reject":
        return required_text(payload.get("reply"), "preference reply")
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
        return required_text(payload.get("reply"), "preference reply")
    keys = ", ".join(str(candidate.get("key")) for candidate in candidates)
    reply = required_text(payload.get("reply"), "preference reply")
    return f"{reply} ({keys})"


async def handle_memory(rest: str, core: CoreClient, *, user_id: str | None, channel_id: str | None) -> str:
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


async def record_message(core: CoreClient, message: Any, *, role: str, content: str) -> None:
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


async def remember_task_reference(core: CoreClient, *, user_id: str, task_id: str, task: dict[str, Any], payload: Any) -> None:
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
