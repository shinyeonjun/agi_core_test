from __future__ import annotations

from typing import Any

from agent.bridge.formatter import compact_text, format_action_update, format_daily_summary, format_update_event
from agent.bridge.notifier import post_webhook
from agent.core.approvals import ApprovalStore
from agent.core.cooldown import is_ready, mark
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


def _ko_status(value: object) -> str:
    mapping = {
        "completed": "\uc644\ub8cc",
        "blocked": "\ucc28\ub2e8",
        "timeout": "\uc2dc\uac04 \ucd08\uacfc",
        "running": "\uc9c4\ud589 \uc911",
        "proposed": "\uc81c\uc548\ub428",
        "executed": "\uc2e4\ud589\ub428",
        "safe": "\uc548\uc804 \ubaa8\ub4dc",
        "full_device_lab": "\uc7a5\ube44 \uc2e4\ud5d8 \ubaa8\ub4dc",
        "workspace": "\uc791\uc5c5\uacf5\uac04 \ubaa8\ub4dc",
    }
    raw = compact_text(value)
    return mapping.get(raw, raw)


def _ko_summary(value: object) -> str:
    raw = compact_text(value)
    mapping = {
        "rc=0": "\uc815\uc0c1 \uc885\ub8cc",
        "timeout": "\uc2dc\uac04 \ucd08\uacfc",
        "ssh_key_access_denied": "SSH \ud0a4 \uc811\uadfc \ucc28\ub2e8",
        "profile_not_full_device_lab": "\uc2e4\ud589 \ud504\ub85c\ud544\uc774 \uc544\ub2c8\ub77c \ucc28\ub2e8",
        "root_delete_denied": "\uc704\ud5d8\ud55c \uc0ad\uc81c \ucc28\ub2e8",
        "remote_script_execution_denied": "\uc6d0\uaca9 \uc2a4\ud06c\ub9bd\ud2b8 \uc2e4\ud589 \ucc28\ub2e8",
    }
    return mapping.get(raw, raw.replace("_", " "))


def _ko_event(row: dict[str, Any] | None) -> str:
    if not row:
        return "\uc5c6\uc74c"
    key = f"{row.get('source')}/{row.get('event_type')}"
    mapping = {
        "scheduler/idle_tick": "\uc790\ub3d9 tick \uc2e4\ud589",
        "lab/lab_tick_skipped": "lab tick\uc774 \uc81c\uc548\ub9cc \uc0dd\uc131",
        "lab/lab_tick_executed": "lab action 1\uac1c \uc2e4\ud589",
        "workspace/workspace_artifact_created": "\uc791\uc5c5\uacf5\uac04 \ud30c\uc77c \uc0dd\uc131",
        "policy/policy_check": "\uc815\ucc45 \uac80\uc0ac \uc218\ud589",
        "discord/discord_chat_reply": "\ub300\ud654 \uc751\ub2f5",
        "core/assistant_output": "Core \ub300\ud654 \uc751\ub2f5 \uc0dd\uc131",
        "core/decision_created": "Core \ud310\ub2e8 \uae30\ub85d \uc0dd\uc131",
        "learner/reflection_created": "\ud68c\uace0 \uae30\ub85d \uc0dd\uc131",
    }
    return mapping.get(key, key)


def _reflection_summary(value: object) -> str:
    raw = compact_text(value)
    mapping = {
        "Recorded talk feedback, selected goal, and retrieved context.": "\ub300\ud654 \ud53c\ub4dc\ubc31\uacfc \ubaa9\ud45c/\uae30\uc5b5 \ub9e5\ub77d\uc744 \ud68c\uace0\ub85c \uc800\uc7a5\ud568",
        "Recorded idle action and cooldown state after tick.": "\uc790\ub3d9 tick \uacb0\uacfc\uc640 \ucfe8\ub2e4\uc6b4 \uc0c1\ud0dc\ub97c \ud68c\uace0\ub85c \uc800\uc7a5\ud568",
        "Lab tick generated proposals but did not execute because profile is not full_device_lab.": "lab tick\uc774 \uc81c\uc548\ub9cc \ub9cc\ub4e4\uace0 \uc2e4\ud589\uc740 \ud558\uc9c0 \uc54a\uc74c",
        "Lab tick executed one approved local action and recorded the result.": "lab tick\uc774 \uc2b9\uc778\ub41c \ub85c\uceec action 1\uac1c\ub97c \uc2e4\ud589\ud568",
    }
    return mapping.get(raw, raw)


def _is_noise_goal(goal: dict[str, Any]) -> bool:
    title = str(goal.get("title") or "").lower()
    goal_type = str(goal.get("goal_type") or "").lower()
    return goal_type == "test" or any(token in title for token in ["secret goal summary marker", "answer user input", "apply user negative feedback"])


def _interesting_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy = {"discord_message", "discord_chat_reply", "webhook_sent", "webhook_missing", "discord_command_output"}
    return [event for event in events if event.get("event_type") not in noisy]


def build_activity_summary() -> str:
    metrics = collect_metrics()
    actions = list_action_runs(10)
    goals = list_goals(limit=8)
    events = list_events(limit=12)
    reflections = list_reflections(limit=5)
    approvals = ApprovalStore().list_pending()
    last_action = actions[0] if actions else None
    interesting_events = _interesting_events(events)
    last_event = interesting_events[0] if interesting_events else (events[0] if events else None)
    lines = [
        "**Core \ud65c\ub3d9 \uc694\uc57d**",
        "\uc624\ub80c\uc9c0\ud30c\uc774\uc5d0\uc11c \uc790\ub3d9 \ub8e8\ud504\uac00 \ub3cc\uace0 \uc788\uc5b4.",
        "",
        "**\uc9c0\uae08 \uc0c1\ud0dc**",
        _line("\ud504\ub85c\ud544", _ko_status(metrics.get("current_autonomy_profile"))),
        _line("\ucd5c\uadfc \ud3c9\uac00", f"{metrics.get('last_eval_result')} / {metrics.get('last_eval_score')}"),
        _line("\uc2b9\uc778 \ub300\uae30", f"{len(approvals)}\uac74"),
        _line("\ucd5c\uadfc action", f"#{last_action.get('id')} {_ko_status(last_action.get('status'))} / {_ko_summary(last_action.get('result_summary'))}" if last_action else "\uc5c6\uc74c"),
        _line("\ucd5c\uadfc event", f"#{last_event.get('id')} {_ko_event(last_event)}" if last_event else "\uc5c6\uc74c"),
        "",
        "**\ucd5c\uadfc\uc5d0 \ud55c \uc77c**",
    ]
    if actions:
        for action in actions[:3]:
            lines.append(f"- action #{action.get('id')}: {_ko_status(action.get('status'))} / {_ko_summary(action.get('result_summary'))}")
    elif events:
        for event in interesting_events[:3] or events[:3]:
            lines.append(f"- event #{event.get('id')}: {_ko_event(event)}")
    else:
        lines.append("- \uc544\uc9c1 \ucd5c\uadfc \ud65c\ub3d9\uc774 \uc801\uc5b4.")
    lines.extend(["", "**\ud559\uc2b5/\ud68c\uace0**"])
    if reflections:
        for reflection in reflections[:2]:
            lines.append(f"- #{reflection.get('id')} {_reflection_summary(reflection.get('summary'))}")
    else:
        lines.append("- \uc544\uc9c1 \uc0c8 \ud68c\uace0\uac00 \uc5c6\uc5b4.")
    lines.extend(["", "**\ub2e4\uc74c\uc5d0 \ubcfc \uac83**"])
    open_goals = [goal for goal in goals if goal.get("status") in {"active", "proposed", "blocked", "waiting_approval"} and not _is_noise_goal(goal)]
    if open_goals:
        for goal in open_goals[:3]:
            lines.append(f"- #{goal.get('id')} {compact_text(goal.get('title'))} ({_ko_status(goal.get('status'))})")
    else:
        lines.append("- \uc5f4\ub9b0 \ubaa9\ud45c\uac00 \uc5c6\uc5b4. \ub2e4\uc74c \uc9c0\uc2dc\ub97c \uae30\ub2e4\ub9ac\ub294 \uc911\uc774\uc57c.")
    return "\n".join(lines)


def notify_activity_summary(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    if not dry_run and not force:
        ready, wait = is_ready("discord_activity_summary", 10 * 60)
        if not ready:
            return {"sent": False, "kind": "summary", "reason": "cooldown", "wait_seconds": wait}
    result = post_webhook("summary", build_activity_summary(), dry_run=dry_run)
    if result.get("sent") and not dry_run:
        mark("discord_activity_summary", 10 * 60, {"kind": "summary"})
    return result


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
