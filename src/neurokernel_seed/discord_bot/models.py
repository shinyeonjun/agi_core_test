from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
