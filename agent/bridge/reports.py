from __future__ import annotations

from typing import Any

from agent.bridge.formatter import compact_text, format_action_update, format_daily_summary, format_update_event
from agent.bridge.notifier import post_webhook
from agent.core.approvals import ApprovalStore
from agent.core.events import list_events
from agent.core.goals import list_goals
from agent.core.learner import list_reflections
from agent.core.metrics import collect_metrics
from agent.tools.action_log import get_action_run, list_action_runs


def build_daily_summary() -> str:
    metrics = collect_metrics()
    approvals = ApprovalStore().list(status=None, limit=20)
    actions = list_action_runs(20)
    goals = list_goals(limit=10)
    return format_daily_summary(metrics, approvals, actions, goals)



def _line(prefix: str, value: object) -> str:
    return f"- {prefix}: {compact_text(value)}"


def build_activity_summary() -> str:
    metrics = collect_metrics()
    actions = list_action_runs(10)
    goals = list_goals(limit=8)
    events = list_events(limit=12)
    reflections = list_reflections(limit=5)
    approvals = ApprovalStore().list_pending()
    last_action = actions[0] if actions else None
    last_event = events[0] if events else None
    lines = [
        "**Core \ud65c\ub3d9 \uc694\uc57d**",
        "\uc624\ub80c\uc9c0\ud30c\uc774\uc5d0\uc11c \uc790\ub3d9 \ub8e8\ud504\uac00 \ub3cc\uace0 \uc788\uc5b4.",
        "",
        "**\uc9c0\uae08 \uc0c1\ud0dc**",
        _line("\ud504\ub85c\ud544", metrics.get("current_autonomy_profile")),
        _line("\ucd5c\uadfc \ud3c9\uac00", f"{metrics.get('last_eval_result')} / {metrics.get('last_eval_score')}"),
        _line("\uc2b9\uc778 \ub300\uae30", f"{len(approvals)}\uac74"),
        _line("\ucd5c\uadfc action", f"#{last_action.get('id')} {last_action.get('status')} {last_action.get('result_summary')}" if last_action else "\uc5c6\uc74c"),
        _line("\ucd5c\uadfc event", f"#{last_event.get('id')} {last_event.get('source')}/{last_event.get('event_type')}" if last_event else "\uc5c6\uc74c"),
        "",
        "**\ucd5c\uadfc\uc5d0 \ud55c \uc77c**",
    ]
    if actions:
        for action in actions[:3]:
            lines.append(f"- action #{action.get('id')}: {compact_text(action.get('status'))} / {compact_text(action.get('result_summary'))}")
    elif events:
        for event in events[:3]:
            lines.append(f"- event #{event.get('id')}: {compact_text(event.get('event_type'))}")
    else:
        lines.append("- \uc544\uc9c1 \ucd5c\uadfc \ud65c\ub3d9\uc774 \uc801\uc5b4.")
    lines.extend(["", "**\ud559\uc2b5/\ud68c\uace0**"])
    if reflections:
        for reflection in reflections[:2]:
            lines.append(f"- #{reflection.get('id')} {compact_text(reflection.get('summary'))}")
    else:
        lines.append("- \uc544\uc9c1 \uc0c8 \ud68c\uace0\uac00 \uc5c6\uc5b4.")
    lines.extend(["", "**\ub2e4\uc74c\uc5d0 \ubcfc \uac83**"])
    open_goals = [goal for goal in goals if goal.get("status") in {"active", "proposed", "blocked", "waiting_approval"}]
    if open_goals:
        for goal in open_goals[:3]:
            lines.append(f"- #{goal.get('id')} {compact_text(goal.get('title'))} ({compact_text(goal.get('status'))})")
    else:
        lines.append("- \uc5f4\ub9b0 \ubaa9\ud45c\uac00 \uc5c6\uc5b4. \ub2e4\uc74c \uc9c0\uc2dc\ub97c \uae30\ub2e4\ub9ac\ub294 \uc911\uc774\uc57c.")
    return "\n".join(lines)


def notify_activity_summary(*, dry_run: bool = False) -> dict[str, Any]:
    return post_webhook("summary", build_activity_summary(), dry_run=dry_run)


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
