from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agent.bridge.auth import DiscordAuthConfig, classify_context
from agent.bridge.formatter import compact_text, format_approval_card, format_chat_reply, redact_discord_content, split_for_discord, strip_bot_mention
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import current_profile
from agent.core.cooldown import is_ready, mark
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.goal_generator import meaningful_open_goals
from agent.core.pipeline import run_talk
from agent.core.state import load_state
from agent.core.task_queue import list_tasks
from agent.core.user_goals import cancel_goal_or_task_target
from agent.core.wake_signals import emit_wake_signal
from agent.memory.store import search_memories
from agent.scheduler.tick import run_tick

ChannelRole = Literal["chat", "approval", "summary", "update", "other"]


@dataclass(frozen=True)
class DiscordEvent:
    guild_id: str | None
    channel_id: str
    user_id: str
    message_id: str
    is_dm: bool
    was_mention: bool
    content: str
    author_is_bot: bool = False


def channel_role(channel_id: str, config: DiscordAuthConfig) -> ChannelRole:
    channel = str(channel_id)
    if config.chat_channel_id and channel == config.chat_channel_id:
        return "chat"
    if config.approval_channel_id and channel == config.approval_channel_id:
        return "approval"
    if config.summary_channel_id and channel == config.summary_channel_id:
        return "summary"
    if config.update_channel_id and channel == config.update_channel_id:
        return "update"
    return "other"


def _discord_message_seen(event: DiscordEvent) -> bool:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM discord_events
            WHERE channel_id = ? AND user_id = ? AND message_id = ?
            LIMIT 1
            """,
            (event.channel_id, event.user_id, event.message_id),
        ).fetchone()
    return row is not None


def log_discord_input(event: DiscordEvent, text: str, role: str | None = None) -> int:
    init_db()
    metadata = {
        "guild_id": event.guild_id,
        "channel_id": event.channel_id,
        "user_id": event.user_id,
        "message_id": event.message_id,
        "is_dm": event.is_dm,
        "was_mention": event.was_mention,
        "channel_role": role,
    }
    core_event_id = log_event("discord", "discord_message", redact_discord_content(text), metadata, 0.8)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO discord_events (
                created_at, guild_id, channel_id, user_id, message_id,
                is_dm, was_mention, content_redacted, event_id
            ) VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.guild_id,
                event.channel_id,
                event.user_id,
                event.message_id,
                1 if event.is_dm else 0,
                1 if event.was_mention else 0,
                redact_discord_content(text),
                core_event_id,
            ),
        )
        conn.commit()
        if cur.rowcount == 0:
            return 0
    return core_event_id


def _status_label(status: object) -> str:
    return {
        "queued": "대기 중",
        "running": "진행 중",
        "waiting_approval": "승인 대기",
        "done": "완료",
        "blocked": "막힘",
        "skipped": "건너뜀",
        "active": "진행 중",
        "proposed": "제안됨",
        "selected": "선택됨",
    }.get(str(status), compact_text(status, "알 수 없음"))


def _short_text(value: object, default: str = "없음", limit: int = 90) -> str:
    text = redact_discord_content(compact_text(value, default))
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _memory_text(row: dict[str, Any]) -> str:
    title = compact_text(row.get("title"), "")
    content = compact_text(row.get("content"), "")
    lowered = title.lower()
    generic_title = lowered.startswith("summary:") or "digital agi korean memory" in lowered or lowered in {"test memory", "audit memory"}
    return _short_text(content if generic_title and content else title or content, "기억 내용 없음", 110)


def _state_summary() -> str:
    state = load_state()
    tasks = list_tasks(limit=30)
    running = sum(1 for task in tasks if task.get("status") == "running")
    queued = sum(1 for task in tasks if task.get("status") == "queued")
    approvals = len(ApprovalStore().list_pending())
    return "\n".join(
        [
            "**Core 상태**",
            f"모드: {current_profile()}",
            f"작업: 진행 {running}개 / 대기 {queued}개",
            f"승인 대기: {approvals}건",
            f"초점: {_short_text(state.get('current_focus'), '정해진 초점 없음', 70)}",
            f"마지막 대화: {_short_text(state.get('last_user_interaction_at'), '기록 없음', 40)}",
            f"마지막 점검: {_short_text(state.get('last_idle_tick_at'), '기록 없음', 40)}",
        ]
    )


def _memory_summary(query: str) -> str:
    rows = search_memories(query or "core", limit=10)
    if not rows:
        return "**기억 요약**\n아직 보여줄 만한 기억이 없어."
    lines = ["**기억 요약**"]
    seen: set[str] = set()
    for row in rows:
        text = _memory_text(row)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        lines.append(f"{len(lines)}. {text}")
        if len(lines) >= 7:
            break
    return "\n".join(lines)


def _approval_summary() -> str:
    rows = ApprovalStore().list_pending()
    if not rows:
        return "**승인 대기**\n지금 승인할 항목은 없어."
    return "\n\n".join(format_approval_card(row) for row in rows[:10])


def _goal_summary() -> str:
    rows = meaningful_open_goals(limit=8)
    if not rows:
        return "**목표 요약**\n지금 열려 있는 의미 있는 목표는 없어."
    lines = ["**목표 요약**"]
    for row in rows[:5]:
        lines.append(f"- #{row.get('id')} {_short_text(row.get('title'), '목표', 95)} / {_status_label(row.get('status'))}")
    return "\n".join(lines)


def _work_summary() -> str:
    tasks = list_tasks(limit=12)
    approvals = ApprovalStore().list_pending()
    visible = [task for task in tasks if task.get("status") in {"queued", "running", "waiting_approval"}]
    lines = ["**작업판**"]
    if not visible:
        lines.append("지금 대기/진행 중인 작업은 없어.")
    else:
        for task in visible[:5]:
            label = "자가개선" if str(task.get("source") or "").startswith("discord_self_improvement") else "작업"
            lines.append(f"- #{task.get('id')} {label}: {_short_text(task.get('title'), '작업', 95)} / {_status_label(task.get('status'))}")
    lines.extend(["", "**승인 대기**"])
    if approvals:
        for row in approvals[:3]:
            lines.append(f"- #{row.get('id')} {_short_text(row.get('description'), '승인 필요', 90)}")
    else:
        lines.append("없어.")
    return "\n".join(lines)


def _tick_summary() -> str:
    tick = run_tick()
    result = tick.get("result") if isinstance(tick.get("result"), dict) else {}
    growth = result.get("cognitive_growth") if isinstance(result, dict) else None
    if isinstance(growth, dict):
        growth_text = f"{compact_text(growth.get('mode'), '확인')} / {compact_text(growth.get('top_curiosity'), '일반')}"
    else:
        growth_text = "새 성장 작업 없음"
    created_goal = result.get("created_goal_id")
    processed = result.get("processed_events")
    return "\n".join(
        [
            "**점검 완료**",
            f"새 목표: {created_goal if created_goal else '없음'}",
            f"정리한 이벤트: {processed if processed is not None else '확인 안 됨'}건",
            f"성장 루프: {growth_text}",
        ]
    )


def _cancel_summary(text: str) -> str:
    result = cancel_goal_or_task_target(text)
    if not result:
        return "취소할 대상을 이해하지 못했어. 예: `!cancel 524` 또는 `#524 없애줘`."
    cancel = result.get("cancel_result") if isinstance(result.get("cancel_result"), dict) else {}
    status = compact_text(cancel.get("status"))
    if status == "missing_target":
        return "취소할 번호가 필요해. 예: `!cancel 524`."
    if status == "not_found":
        return f"#{compact_text(cancel.get('target_id'))}는 목표/작업 목록에서 못 찾았어."
    return f"정리했어. 작업 {compact_text(cancel.get('cancelled_tasks'), '0')}개를 멈추고, 목표 {compact_text(cancel.get('archived_goals'), '0')}개를 목록에서 뺐어."


def handle_command(text: str, *, role: ChannelRole = "chat") -> str | None:
    if not text.startswith("!"):
        return None
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    if role == "approval" and command not in {"!approvals", "!approve", "!reject", "!detail"}:
        return "여기는 승인 채널이야. `!approvals`, `!approve <id>`, `!reject <id>`만 사용할 수 있어."
    if command == "!state":
        return _state_summary()
    if command in {"!work", "!tasks"}:
        return _work_summary()
    if command == "!goals":
        return _goal_summary()
    if command == "!tick":
        return _tick_summary()
    if command == "!memories":
        return _memory_summary(arg)
    if command == "!approvals":
        return _approval_summary()
    if command in {"!cancel", "!remove"}:
        return _cancel_summary(text)
    if command in {"!detail", "!approval"} and arg.strip().isdigit():
        rows = [row for row in ApprovalStore().list(status=None, limit=100) if int(row.get("id", -1)) == int(arg.strip())]
        return format_approval_card(rows[0]) if rows else "해당 승인 항목을 찾지 못했어."
    if command == "!approve" and arg.strip().isdigit():
        ok = ApprovalStore().approve(int(arg.strip()))
        return f"승인 완료: #{arg.strip()}\n연결된 작업이 있으면 다시 진행시킬게." if ok else "승인할 항목이 없거나 이미 처리됐어."
    if command == "!reject" and arg.strip().isdigit():
        ok = ApprovalStore().reject(int(arg.strip()))
        return f"거절 완료: #{arg.strip()}\n연결된 작업이 있으면 차단 상태로 정리했어." if ok else "거절할 항목이 없거나 이미 처리됐어."
    return "알 수 없는 명령이야. 사용 가능: `!state`, `!work`, `!goals`, `!tick`, `!memories`, `!approvals`, `!cancel <id>`, `!approve <id>`, `!reject <id>`."


def route_discord_event(event: DiscordEvent, config: DiscordAuthConfig) -> list[str]:
    role = channel_role(event.channel_id, config)
    mode = classify_context(
        user_id=event.user_id,
        channel_id=event.channel_id,
        is_dm=event.is_dm,
        was_mention=event.was_mention,
        author_is_bot=event.author_is_bot,
        config=config,
    )
    if mode in {"ignored", "denied"}:
        return []
    text = strip_bot_mention(event.content) if mode == "mention_conversation" else event.content.strip()
    if not text:
        return []
    if _discord_message_seen(event):
        log_event("discord", "discord_duplicate_message_ignored", text, {"message_id": event.message_id, "channel_role": role}, 0.35)
        return []
    core_event_id = log_discord_input(event, text, role)
    if core_event_id <= 0:
        log_event("discord", "discord_duplicate_message_ignored", text, {"message_id": event.message_id, "channel_role": role}, 0.35)
        return []
    emit_wake_signal(
        "discord_message",
        "discord",
        priority=0.96 if role == "chat" else 0.72,
        payload={"event_id": core_event_id, "message_id": event.message_id, "channel_id": event.channel_id, "channel_role": role, "is_dm": event.is_dm},
        dedupe_key=f"discord_message:{event.message_id}",
    )
    if role in {"summary", "update"}:
        log_event("discord", "discord_webhook_only_channel_input", text, {"message_id": event.message_id, "channel_role": role}, 0.5)
        return ["이 채널은 Core가 알림만 보내는 곳이야. 대화는 #대화, 승인은 #승인에서 해줘."]
    if role == "approval" and not text.startswith("!"):
        log_event("discord", "discord_approval_chat_blocked", text, {"message_id": event.message_id}, 0.55)
        return ["여기는 승인 전용 채널이야. `!approvals`, `!approve <id>`, `!reject <id>`만 사용할 수 있어."]
    ready, wait = is_ready(f"discord_answer:{event.user_id}", config.user_cooldown_seconds)
    if not ready:
        log_event("discord", "discord_chat_cooldown_suppressed", "", {"message_id": event.message_id, "channel_role": role, "wait_seconds": wait}, 0.35)
        return []
    mark(f"discord_answer:{event.user_id}", config.user_cooldown_seconds, {"channel_id": event.channel_id, "channel_role": role})
    command_output = handle_command(text, role=role)
    if command_output is not None:
        log_event("discord", "discord_command_output", command_output, {"message_id": event.message_id, "channel_role": role}, 0.6)
        return split_for_discord(command_output, config.max_response_chars)
    if role == "approval":
        return ["승인 채널에서는 일반 대화를 처리하지 않아. `!approvals`로 대기 목록을 확인해줘."]
    result = run_talk(text, source="discord", source_event_id=core_event_id, metadata={"message_id": event.message_id, "channel_role": role})
    reply = format_chat_reply(text, result)
    log_event("discord", "discord_chat_reply", reply, {"message_id": event.message_id, "channel_role": role}, 0.55)
    return split_for_discord(reply, config.max_response_chars)
