from __future__ import annotations

import json
import re
import textwrap
from typing import Any

MENTION_RE = re.compile(r"<@!?\d+>")
SECRET_PATTERNS = (
    re.compile(r"https://discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"(?i)authorization:\s*bearer\s+[^\s]+"),
    re.compile(r"(?i)(discord[_-]?bot[_-]?token|bot[_-]?token|token|api[_-]?key)\s*[:=]\s*[^\s]+"),
    re.compile(r"(?i)\b(mfa\.[A-Za-z0-9_-]+|[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27,})\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
)


def strip_bot_mention(text: str) -> str:
    return MENTION_RE.sub("", text).strip()


def split_for_discord(text: str, limit: int = 1800) -> list[str]:
    if limit < 100:
        raise ValueError("limit must be at least 100")
    chunks: list[str] = []
    current = ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(para) <= limit:
            current = para
        else:
            chunks.extend(textwrap.wrap(para, width=limit, replace_whitespace=False, drop_whitespace=False) or [para[:limit]])
    if current:
        chunks.append(current)
    return chunks or [""]


def redact_discord_content(text: str) -> str:
    result = str(text)
    for pattern in SECRET_PATTERNS:
        result = pattern.sub(lambda match: _redaction(match.group(0)), result)
    return result[:4000]


def _redaction(value: str) -> str:
    if "webhooks" in value.lower():
        return "<redacted_discord_webhook>"
    if "=" in value:
        return value.split("=", 1)[0] + "=<redacted>"
    if ":" in value and "bearer" in value.lower():
        return value.split(":", 1)[0] + ": <redacted>"
    return "<redacted_secret>"


def compact_text(value: Any, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = redact_discord_content(str(value)).strip()
    return text if text else fallback


def format_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"**{title}**"]
    for key, value in rows.items():
        lines.append(f"- {key}: {compact_text(value)}")
    return "\n".join(lines)


def format_approval_card(row: dict[str, Any]) -> str:
    proposal = row.get("proposal") or {}
    description = row.get("description") or proposal.get("description") or "\uc2b9\uc778\uc774 \ud544\uc694\ud55c \uc791\uc5c5"
    reason = proposal.get("denied_reason") or proposal.get("reason") or "\uc0ac\ub78c \ud655\uc778 \ud544\uc694"
    return "\n".join([
        f"**\uc2b9\uc778 \ud544\uc694 #{row.get('id')}**",
        f"\uc791\uc5c5: {compact_text(description)}",
        f"\uc704\ud5d8\ub3c4: {compact_text(row.get('risk_level'))}",
        f"\uc0c1\ud0dc: {compact_text(row.get('status'))}",
        f"\uc774\uc720: {compact_text(reason)}",
        "\uba85\ub839: `!approve <id>` \ub610\ub294 `!reject <id>`",
    ])


def format_action_update(row: dict[str, Any] | None) -> str:
    if not row:
        return "**\uc791\uc5c5 \uc5c5\ub370\uc774\ud2b8**\n\uc694\uccad\ud55c action\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc5b4."
    command = row.get("command_json")
    try:
        command = json.loads(command) if isinstance(command, str) else command
    except json.JSONDecodeError:
        pass
    return "\n".join([
        f"**\uc791\uc5c5 {compact_text(row.get('status'))} #{row.get('id')}**",
        f"\ud504\ub85c\ud544: {compact_text(row.get('profile'))}",
        f"\uc704\ud5d8\ub3c4: {compact_text(row.get('risk_level'))}",
        f"\uba85\ub839: `{compact_text(command)}`",
        f"\uacb0\uacfc: {compact_text(row.get('result_summary'))}",
        f"\ubc18\ud658\uac12: {compact_text(row.get('returncode'))}",
    ])


def format_daily_summary(metrics: dict[str, Any], approvals: list[dict[str, Any]], actions: list[dict[str, Any]], goals: list[dict[str, Any]]) -> str:
    pending = len([row for row in approvals if row.get("status") == "pending"])
    completed = len([row for row in actions if row.get("status") == "completed"])
    blocked = len([row for row in actions if row.get("status") == "blocked"])
    lines = [
        "**Core \uc694\uc57d**",
        f"\ud504\ub85c\ud544: {compact_text(metrics.get('current_autonomy_profile'))}",
        f"\ud3c9\uac00: {compact_text(metrics.get('last_eval_result'))} / score {compact_text(metrics.get('last_eval_score'))}",
        f"\uc791\uc5c5: \uc644\ub8cc {completed}, \ucc28\ub2e8 {blocked}, \ucd5c\uadfc {len(actions)}\uac1c \uae30\uc900",
        f"\uc2b9\uc778 \ub300\uae30: {pending}\uac74",
        f"\uae30\uc5b5/\ud68c\uace0: memory {compact_text(metrics.get('memories_count'))}, reflection {compact_text(metrics.get('reflections_count'))}",
        "",
        "**\ub2e4\uc74c\uc5d0 \ubcfc \uac83**",
    ]
    if goals:
        for index, goal in enumerate(goals[:3], start=1):
            lines.append(f"{index}. #{goal.get('id')} {compact_text(goal.get('title'))} ({compact_text(goal.get('status'))})")
    else:
        lines.append("\uc5f4\ub9b0 \ubaa9\ud45c\uac00 \uac70\uc758 \uc5c6\uc5b4. \ub2e4\uc74c \ubaa9\ud45c\ub97c \uc815\ud558\uba74 \ub3fc.")
    return "\n".join(lines)


def format_update_event(title: str, detail: str, fields: dict[str, Any] | None = None) -> str:
    lines = [f"**{compact_text(title)}**", compact_text(detail)]
    for key, value in (fields or {}).items():
        lines.append(f"- {key}: {compact_text(value)}")
    return "\n".join(lines)


def _looks_like_greeting(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"hi", "hello", "hey", "\u314e\u3147", "\ud558\uc774", "\uc548\ub155", "\uc548\ub155\ud558\uc138\uc694", "\ud5ec\ub85c", "\u3147\u3147"} or lowered.startswith(("\u314e\u3147", "\uc548\ub155"))


def _asks_for_status(text: str) -> bool:
    normalized = text.replace(" ", "")
    return any(token in text for token in ["\uc0c1\ud0dc", "\ubb50 \ud558\uace0", "\ubb50 \ud558\ub294", "\ubb50\ud574", "\uc0b4\uc544", "\uc815\uc0c1", "\uccb4\ud06c", "\ud655\uc778"]) or any(token in normalized for token in ["\ubb50\ud558\uace0", "\ubb50\ud558\ub294", "\ubb50\ud574", "\ud558\uace0\uc788", "\ud558\ub294\uc911"])


def _asks_for_help(text: str) -> bool:
    return any(token in text.lower() for token in ["\ub3c4\uc6c0", "\uba85\ub839", "help", "\ubb50 \ud560", "\uc0ac\uc6a9\ubc95", "\uae30\ub2a5"])


def format_chat_reply(user_text: str, core_result: dict[str, Any]) -> str:
    decision = core_result.get("decision") or {}
    policy = decision.get("policy_summary") or {}
    text = redact_discord_content(user_text).strip()
    if policy.get("denied"):
        reason = compact_text(policy.get("reason") or policy.get("denied_reason") or "\uc815\ucc45 \ucc28\ub2e8")
        return "\n".join([
            "\uadf8 \uc694\uccad\uc740 \uc704\ud5d8\ud560 \uc218 \uc788\uc5b4\uc11c \uc2e4\ud589\ud558\uc9c0 \uc54a\uc558\uc5b4.",
            f"\uc774\uc720: {reason}",
            "\ud544\uc694\ud558\uba74 #\uc2b9\uc778 \ucc44\ub110\uc5d0\uc11c \uc2b9\uc778 \ud56d\ubaa9\uc744 \ud655\uc778\ud574\uc918.",
        ])
    if _looks_like_greeting(text):
        return "\uc751, \uc5ec\uae30 \uc788\uc5b4. \ud3b8\ud558\uac8c \ub9d0\ud574\uc918."
    if _asks_for_help(text):
        return "\n".join([
            "\uc5ec\uae30\ub294 \ub300\ud654 \ucc44\ub110\uc774\uc57c. \uadf8\ub0e5 \uc790\uc5f0\uc5b4\ub85c \ub9d0\ud558\uba74 \ub3fc.",
            "\uc2b9\uc778\uc774 \ud544\uc694\ud55c \uc791\uc5c5\uc740 #\uc2b9\uc778, \ubcf4\uace0\uc11c\ub294 #\uc694\uc57d, \uc2e4\uc2dc\uac04 \ub85c\uadf8\ub294 #\uc5c5\ub370\uc774\ud2b8\ub85c \uac08 \uac70\uc57c.",
            "\uc790\uc138\ud55c \ub0b4\ubd80 \uc0c1\ud0dc\uac00 \ud544\uc694\ud560 \ub54c\ub9cc `!state`, `!goals`, `!approvals`\ub97c \uc368\uc918.",
        ])
    if _asks_for_status(text):
        metrics = decision.get("metrics") or {}
        profile = metrics.get("current_autonomy_profile") or decision.get("autonomy_profile") or "safe"
        eval_result = metrics.get("last_eval_result") or "unknown"
        return "\n".join([
            "\uc9c0\uae08\uc740 Discord\uc5d0\uc11c \ub300\ud654\ub97c \ub4e3\uace0, \uc624\ub80c\uc9c0\ud30c\uc774\uc5d0\uc11c \uc790\ub3d9 tick\uacfc \uc694\uc57d \ub8e8\ud504\ub97c \uc720\uc9c0\ud558\ub294 \uc911\uc774\uc57c.",
            f"\ud504\ub85c\ud544\uc740 `{profile}`\uc774\uace0, \ucd5c\uadfc \ud3c9\uac00\ub294 `{eval_result}`\ub85c \ubcf4\uc5ec.",
            "\uc790\uc138\ud55c \ub0b4\ubd80 \uc0c1\ud0dc\ub294 `!state`\ub85c \ubcfc \uc218 \uc788\uc5b4.",
        ])
    if text.endswith("?") or text.endswith("\uff1f"):
        return "\uc9c8\ubb38\uc73c\ub85c \uc774\ud574\ud588\uc5b4. \uc774\uc5b4\uc11c \ub354 \uad6c\uccb4\uc801\uc73c\ub85c \ub9d0\ud574\uc8fc\uba74 \uadf8 \uae30\uc900\uc73c\ub85c \ub3c4\uc640\uc904\uac8c."
    if len(text) <= 20:
        return "\uc751, \ub4e4\uc5c8\uc5b4. \ub2e4\uc74c\uc5d0 \ubb58 \ud558\uba74 \ub420\uc9c0 \ubc14\ub85c \ub9d0\ud574\uc918."
    return "\uc88b\uc544, \uc774\ud574\ud588\uc5b4. \uc774 \ub300\ud654 \ucc44\ub110\uc5d0\uc11c\ub294 \uc791\uc5c5 \ubc29\ud5a5\uc744 \uc790\uc5f0\uc5b4\ub85c \ubc1b\uc544\uc11c Core\uc5d0 \ubc18\uc601\ud560\uac8c."
