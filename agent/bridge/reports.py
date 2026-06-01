from __future__ import annotations

from typing import Any

from agent.bridge.formatter import format_action_update, format_daily_summary, format_update_event
from agent.bridge.notifier import post_webhook
from agent.core.approvals import ApprovalStore
from agent.core.goals import list_goals
from agent.core.metrics import collect_metrics
from agent.tools.action_log import get_action_run, list_action_runs


def build_daily_summary() -> str:
    metrics = collect_metrics()
    approvals = ApprovalStore().list(status=None, limit=20)
    actions = list_action_runs(20)
    goals = list_goals(limit=10)
    return format_daily_summary(metrics, approvals, actions, goals)


def notify_daily_summary(*, dry_run: bool = False) -> dict[str, Any]:
    return post_webhook("summary", build_daily_summary(), dry_run=dry_run)


def notify_test_summary(*, dry_run: bool = False) -> dict[str, Any]:
    content = format_update_event(
        "\uc694\uc57d \ucc44\ub110 \ud14c\uc2a4\ud2b8",
        "Core \uc694\uc57d \uc6f9\ud6c5\uc774 \uc5f0\uacb0\ub410\ub294\uc9c0 \ud655\uc778\ud558\ub294 \uba54\uc2dc\uc9c0\uc57c.",
        {"\uc0c1\ud0dc": "\ud14c\uc2a4\ud2b8", "\ubbfc\uac10\uc815\ubcf4": "\uc804\uc1a1 \uc548 \ud568"},
    )
    return post_webhook("summary", content, dry_run=dry_run)


def notify_test_update(*, dry_run: bool = False) -> dict[str, Any]:
    content = format_update_event(
        "\uc5c5\ub370\uc774\ud2b8 \ucc44\ub110 \ud14c\uc2a4\ud2b8",
        "Core \uc2e4\uc2dc\uac04 \uc5c5\ub370\uc774\ud2b8 \uc6f9\ud6c5\uc774 \uc5f0\uacb0\ub410\ub294\uc9c0 \ud655\uc778\ud558\ub294 \uba54\uc2dc\uc9c0\uc57c.",
        {"\uc0c1\ud0dc": "\ud14c\uc2a4\ud2b8", "\ub2e4\uc74c": "action/eval/policy \uc774\ubca4\ud2b8"},
    )
    return post_webhook("update", content, dry_run=dry_run)


def notify_action(action_id: int, *, dry_run: bool = False) -> dict[str, Any]:
    return post_webhook("update", format_action_update(get_action_run(action_id)), dry_run=dry_run)
