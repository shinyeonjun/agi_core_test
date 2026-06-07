from __future__ import annotations

import os

from .bot_utils import parse_user_ids as _parse_user_ids
from .models import DiscordBotConfig


def config_from_env(
    *,
    token_env: str = "DISCORD_BOT_TOKEN",
    channel_id: int | None = None,
    core_url: str | None = None,
    prefix: str | None = None,
    allow_dms: bool | None = None,
    allowed_user_ids: tuple[int, ...] | None = None,
    reply_without_prefix: bool | None = None,
    auto_do_low_risk: bool | None = None,
    work_notify_enabled: bool | None = None,
    work_notify_interval_seconds: float | None = None,
) -> DiscordBotConfig:
    token = os.environ.get(token_env, "").strip()
    if not token:
        raise RuntimeError(f"{token_env} is required")
    resolved_channel_id = channel_id
    if resolved_channel_id is None:
        resolved_channel_id = int(required_env("DISCORD_CHANNEL_ID"))
    resolved_allow_dms = allow_dms
    if resolved_allow_dms is None:
        resolved_allow_dms = required_bool_env("DISCORD_ALLOW_DMS")
    resolved_allowed_user_ids = allowed_user_ids
    if resolved_allowed_user_ids is None:
        resolved_allowed_user_ids = _parse_user_ids(required_env("DISCORD_ALLOWED_USER_IDS"))
        if not resolved_allowed_user_ids:
            raise RuntimeError("DISCORD_ALLOWED_USER_IDS must contain at least one user id")
    resolved_reply_without_prefix = reply_without_prefix
    if resolved_reply_without_prefix is None:
        resolved_reply_without_prefix = required_bool_env("NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX")
    resolved_auto_do_low_risk = auto_do_low_risk
    if resolved_auto_do_low_risk is None:
        resolved_auto_do_low_risk = required_bool_env("NEUROKERNEL_BOT_AUTO_DO_LOW_RISK")
    resolved_work_notify_enabled = work_notify_enabled
    if resolved_work_notify_enabled is None:
        resolved_work_notify_enabled = optional_bool_env("NEUROKERNEL_BOT_WORK_NOTIFY_ENABLED", default=True)
    resolved_work_notify_interval = work_notify_interval_seconds
    if resolved_work_notify_interval is None:
        resolved_work_notify_interval = float(os.environ.get("NEUROKERNEL_BOT_WORK_NOTIFY_INTERVAL", "10"))
    return DiscordBotConfig(
        token=token,
        channel_id=resolved_channel_id,
        core_url=core_url if core_url is not None else required_env("NEUROKERNEL_CORE_URL"),
        prefix=prefix if prefix is not None else required_env_defined("NEUROKERNEL_BOT_PREFIX"),
        allow_dms=resolved_allow_dms,
        allowed_user_ids=resolved_allowed_user_ids,
        reply_without_prefix=resolved_reply_without_prefix,
        auto_do_low_risk=resolved_auto_do_low_risk,
        work_notify_enabled=resolved_work_notify_enabled,
        work_notify_interval_seconds=resolved_work_notify_interval,
    )


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def required_env_defined(name: str) -> str:
    if name not in os.environ:
        raise RuntimeError(f"{name} is required")
    return os.environ[name]


def required_bool_env(name: str) -> bool:
    value = required_env(name).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def optional_bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")
