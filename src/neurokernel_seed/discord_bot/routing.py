from __future__ import annotations

from .models import DiscordBotConfig


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
    "improve",
    "self-improve",
    "self_improve",
    "자가개선",
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


def command_line_from_content(content: str, config: DiscordBotConfig) -> str | None:
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
