from __future__ import annotations

import json
from typing import Any

from agent.bridge.formatter import compact_text, format_action_update, format_daily_summary, format_update_event
from agent.bridge.notifier import post_webhook
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import get_autonomy_state
from agent.core.cooldown import is_ready, mark
from agent.core.events import list_events
from agent.core.goal_generator import list_goal_candidates, meaningful_open_goals
from agent.core.learner import list_reflections
from agent.core.metrics import collect_metrics
from agent.core.observability import action_failure_breakdown, action_observation, task_observation
from agent.core.self_map import self_map_brief
from agent.core.task_queue import list_tasks, task_status_counts
from agent.lab.proposals import list_action_proposals, proposal_status_counts
from agent.tools.action_log import get_action_run, list_action_runs


def build_daily_summary() -> str:
    metrics = collect_metrics()
    self_map = self_map_brief()
    approvals = ApprovalStore().list(status=None, limit=20)
    actions = list_action_runs(20)
    goals = meaningful_open_goals(limit=10)
    content = format_daily_summary(metrics, approvals, actions, goals)
    if self_map:
        content += "\n\n**\ubab8 \uc0c1\ud0dc**\n" + _line("\ucd5c\uadfc \ud655\uc778", self_map.get("summary"))
    return content


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
        "rejected": "\ud0c8\ub77d",
        "dry_run": "\ubbf8\ub9ac\ubcf4\uae30",
        "candidate": "\ud6c4\ubcf4",
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
        "command_not_found": "명령 없음",
        "approval_required": "승인 필요",
        "env_access_denied": ".env 접근 차단",
        "secret_access_denied": "민감정보 접근 차단",
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
        "lab/lab_tick_timer_skipped": "lab timer\uac00 \uc548\uc804 \ubaa8\ub4dc\ub77c \uc2e4\ud589\uc744 \uac74\ub108\ub700",
        "lab/lab_tick_executed": "lab action 1\uac1c \uc2e4\ud589",
        "lab/lab_tick_blocked": "lab tick \uc2e4\ud589 \ud6c4\ubcf4 \uc5c6\uc74c",
        "workspace/workspace_artifact_created": "\uc791\uc5c5\uacf5\uac04 \ud30c\uc77c \uc0dd\uc131",
        "policy/policy_check": "\uc815\ucc45 \uac80\uc0ac \uc218\ud589",
        "discord/discord_chat_reply": "\ub300\ud654 \uc751\ub2f5",
        "discord/action_update_notify_failed": "action \uc5c5\ub370\uc774\ud2b8 \uc54c\ub9bc \uc2e4\ud328",
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


def _decode_command(value: object) -> list[str]:
    if value is None:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return [compact_text(value)]
    if isinstance(decoded, list):
        return [compact_text(item) for item in decoded]
    return [compact_text(decoded)]


def _action_label(row: dict[str, Any]) -> str:
    summary = compact_text(row.get("result_summary"))
    command = " ".join(_decode_command(row.get("command_json"))).lower()
    if summary == "profile_not_full_device_lab":
        return "\uc548\uc804 \ubaa8\ub4dc\uc5d0\uc11c \ub85c\uceec \uc2e4\ud589 \ucc28\ub2e8"
    if "df -h" in command or command.startswith("df "):
        return "\ub514\uc2a4\ud06c \uc0c1\ud0dc \uc810\uac80"
    if command.startswith("free "):
        return "\uba54\ubaa8\ub9ac \uc0c1\ud0dc \uc810\uac80"
    if "systemctl --failed" in command:
        return "\uc2e4\ud328\ud55c \uc11c\ube44\uc2a4 \uc810\uac80"
    if command.startswith("printf "):
        return "\ud14c\uc2a4\ud2b8 action \uc2e4\ud589"
    if row.get("status") == "blocked":
        return "\uc815\ucc45\uc5d0 \uc758\ud574 action \ucc28\ub2e8"
    return "\ub85c\uceec action \ucc98\ub9ac"


def _proposal_label(row: dict[str, Any]) -> str:
    reason = compact_text(row.get("reason"))
    mapping = {
        "profile_not_full_device_lab": "\uc548\uc804 \ubaa8\ub4dc\ub77c \uc2e4\ud589 \ub300\uae30",
        "system_disk_check": "\ub514\uc2a4\ud06c \uc0c1\ud0dc \uc810\uac80 \ud6c4\ubcf4",
        "system_memory_check": "\uba54\ubaa8\ub9ac \uc0c1\ud0dc \uc810\uac80 \ud6c4\ubcf4",
        "system_failed_services_check": "\uc2e4\ud328\ud55c \uc11c\ube44\uc2a4 \uc810\uac80 \ud6c4\ubcf4",
        "lab_experiment_workspace_report": "\uc791\uc5c5\uacf5\uac04 \ub9ac\ud3ec\ud2b8 \ud6c4\ubcf4",
        "completed": "\uc2e4\ud589 \uc644\ub8cc",
    }
    return mapping.get(reason, reason.replace("_", " "))


def _format_rate(value: object) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _state_note(profile: object, autonomy: dict[str, Any]) -> str:
    if autonomy.get("catastrophic_local_destruction_allowed"):
        return "\uc704\ud5d8 \uc0ad\uc81c arm\uc774 \ucf1c\uc838 \uc788\uc5b4. \ubc14\ub85c \ud655\uc778 \ud544\uc694"
    if profile == "full_device_lab":
        return "\uc7a5\ube44 \uc2e4\ud5d8 \ubaa8\ub4dc\ub77c lab timer\uac00 \uc2b9\uc778\ub41c \ub85c\uceec action\uc744 \uc2e4\ud589\ud560 \uc218 \uc788\uc5b4"
    if profile == "workspace":
        return "\uc791\uc5c5\uacf5\uac04 \ubaa8\ub4dc\ub77c \ud30c\uc77c/\ud504\ub85c\uc81d\ud2b8 \uc911\uc2ec\uc73c\ub85c \uad00\ucc30 \uc911\uc774\uc57c"
    return "\uc548\uc804 \ubaa8\ub4dc\ub77c \uc790\ub3d9 \ub85c\uceec \uc2e4\ud589\uc740 \uc7a0\uaca8 \uc788\uc5b4"


def _is_noise_goal(goal: dict[str, Any]) -> bool:
    title = str(goal.get("title") or "").lower()
    goal_type = str(goal.get("goal_type") or "").lower()
    return goal_type == "test" or any(token in title for token in ["secret goal summary marker", "answer user input", "apply user negative feedback"])


def _interesting_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy = {"discord_message", "discord_chat_reply", "webhook_sent", "webhook_missing", "discord_command_output"}
    return [event for event in events if event.get("event_type") not in noisy]


def build_observation_dashboard() -> str:
    metrics = collect_metrics()
    autonomy = get_autonomy_state()
    actions = list_action_runs(10)
    proposals = list_action_proposals(8)
    pending_proposals = [row for row in proposals if row.get("status") not in {"executed", "rejected"}]
    proposal_counts = proposal_status_counts()
    goal_candidates = list_goal_candidates(limit=5)
    goals = meaningful_open_goals(limit=10)
    events = list_events(limit=16)
    reflections = list_reflections(limit=5)
    approvals = ApprovalStore().list_pending()
    self_map = self_map_brief()
    task_counts = task_status_counts()
    recent_tasks = list_tasks(limit=6)
    action_breakdown = action_failure_breakdown(actions)
    last_action = actions[0] if actions else None
    interesting_events = _interesting_events(events)
    last_event = interesting_events[0] if interesting_events else (events[0] if events else None)
    profile = metrics.get("current_autonomy_profile")
    lines = [
        "**Core \uad00\uc81c\ud310**",
        "\uc624\ub80c\uc9c0\ud30c\uc774\uc5d0\uc11c Core \uad00\ucc30 \ub8e8\ud504\uac00 \ub3cc\uace0 \uc788\uc5b4.",
        "",
        "**\ud55c\ub208\uc5d0**",
        _line("\ubaa8\ub4dc", f"{_ko_status(profile)} - {_state_note(profile, autonomy)}"),
        _line("\ucd5c\uadfc \ud3c9\uac00", f"{metrics.get('last_eval_result') or '\uc5c6\uc74c'} / {metrics.get('last_eval_score') if metrics.get('last_eval_score') is not None else '-'}"),
        _line("\uc2b9\uc778 \ub300\uae30", f"{len(approvals)}\uac74"),
        _line("\ucd5c\uadfc action", f"#{last_action.get('id')} {_action_label(last_action)} - {_ko_status(last_action.get('status'))} / {_ko_summary(last_action.get('result_summary'))}" if last_action else "\uc5c6\uc74c"),
        _line("\ucd5c\uadfc event", f"#{last_event.get('id')} {_ko_event(last_event)}" if last_event else "\uc5c6\uc74c"),
    ]
    if self_map:
        services = self_map.get("services") or {}
        active_count = len([state for state in services.values() if state == "active"])
        lines.append(_line("\ubab8 \uc0c1\ud0dc", self_map.get("summary")))
        lines.append(_line("\uc0c1\uc8fc \ub8e8\ud504", f"{active_count}\uac1c active / self-map #{self_map.get('id')}"))
    lines.extend([
        "",
        "**24\uc2dc\uac04 \uc9c0\ud45c**",
        _line("tick", f"{metrics.get('tick_count_24h')}\ud68c"),
        _line("Discord \uba54\uc2dc\uc9c0", f"{metrics.get('discord_messages_24h')}\uac74"),
        _line("action \uc131\uacf5\ub960", _format_rate(metrics.get("action_success_rate_24h"))),
        _line("\ucc28\ub2e8/\uc2dc\uac04\ucd08\uacfc", f"{metrics.get('action_blocked_count_24h')}\uac74 / {metrics.get('action_timeout_count_24h')}\uac74"),
        _line("critical \uc815\ucc45 \uac10\uc9c0", f"{metrics.get('policy_critical_count_24h')}\uac74"),
        _line("\uc0ac\uc6a9\uc790 \ud050", f"queued {task_counts.get('user:queued', 0)} / running {task_counts.get('user:running', 0)}"),
        _line("\uc790\uc728 \ud050", f"queued {task_counts.get('autonomous:queued', 0)} / running {task_counts.get('autonomous:running', 0)}"),
        "",
        "**실패/차단 원인**",
    ])
    if action_breakdown:
        for category, count in sorted(action_breakdown.items()):
            lines.append(f"- {category}: {count}건")
    else:
        lines.append("- 최근 action에는 실패/차단 원인이 없어.")

    lines.extend([
        "",
        "**작업 큐 상태**",
    ])
    if recent_tasks:
        for task in recent_tasks[:4]:
            observed_task = task_observation(task)
            lifecycle = observed_task.get("lifecycle") or {}
            phase = lifecycle.get("last_label") or lifecycle.get("last_phase") or "기록 없음"
            lines.append(f"- #{observed_task['id']} {compact_text(observed_task['title'])}: {observed_task['waiting_reason']} / phase: {phase} / 다음: {observed_task['next_step']}")
    else:
        lines.append("- 현재 작업 큐가 비어 있어.")

    lines.extend([
        "",
        "**\ucd5c\uadfc action**",
    ])
    if actions:
        for action in actions[:4]:
            observed_action = action_observation(action)
            lines.append(f"- #{action.get('id')} {_action_label(action)}: {_ko_status(action.get('status'))} / {observed_action['label']} / {observed_action['summary_label']}")
    else:
        lines.append("- \uc544\uc9c1 \uae30\ub85d\ub41c action\uc774 \uc5c6\uc5b4.")

    lines.extend(["", "**\ubaa9\ud45c \ud6c4\ubcf4**"])
    if goal_candidates:
        for candidate in goal_candidates[:3]:
            lines.append(f"- #{candidate.get('id')} {compact_text(candidate.get('title'))}: {_ko_status(candidate.get('status'))} / score {compact_text(candidate.get('score'))}")
    else:
        lines.append("- \uc544\uc9c1 \uc0c8 \ubaa9\ud45c \ud6c4\ubcf4\uac00 \uc5c6\uc5b4.")

    lines.extend(["", "**\uc81c\uc548 \ud050**"])
    if pending_proposals:
        active_counts = {status: count for status, count in proposal_counts.items() if status not in {"executed", "rejected"}}
        count_text = ", ".join(f"{_ko_status(status)} {count}" for status, count in sorted(active_counts.items())) or "\uc5c6\uc74c"
        lines.append(f"- \uc0c1\ud0dc \ud569\uacc4: {count_text}")
        for proposal in pending_proposals[:3]:
            lines.append(f"- #{proposal.get('id')} {_proposal_label(proposal)}: {_ko_status(proposal.get('status'))}")
    else:
        lines.append("- \ud604\uc7ac 처리 대기 중인 action 제안은 없어.")

    lines.extend(["", "**\ud559\uc2b5/\ud68c\uace0**"])
    if reflections:
        for reflection in reflections[:2]:
            lines.append(f"- #{reflection.get('id')} {_reflection_summary(reflection.get('summary'))}")
    else:
        lines.append("- \uc544\uc9c1 \uc0c8 \ud68c\uace0\uac00 \uc5c6\uc5b4.")

    lines.extend(["", "**\ub2e4\uc74c\uc5d0 \ubcfc \uac83**"])
    open_goals = goals
    if approvals:
        lines.append(f"- \uc2b9\uc778 \ub300\uae30 {len(approvals)}\uac74\ubd80\ud130 \ud655\uc778\ud574\uc918.")
    if open_goals:
        for goal in open_goals[:3]:
            lines.append(f"- #{goal.get('id')} {compact_text(goal.get('title'))} ({_ko_status(goal.get('status'))})")
    if not approvals and not open_goals:
        lines.append("- \uc5f4\ub9b0 \ubaa9\ud45c\uac00 \uc5c6\uc5b4. \ub2e4\uc74c \uc9c0\uc2dc\ub97c \uae30\ub2e4\ub9ac\ub294 \uc911\uc774\uc57c.")
    return "\n".join(lines)


def build_activity_summary() -> str:
    return build_observation_dashboard()


def notify_activity_summary(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    if not dry_run and not force:
        ready, wait = is_ready("discord_activity_summary", 10 * 60)
        if not ready:
            return {"sent": False, "kind": "summary", "reason": "cooldown", "wait_seconds": wait}
    result = post_webhook("summary", build_activity_summary(), dry_run=dry_run)
    if result.get("sent") and not dry_run:
        mark("discord_activity_summary", 10 * 60, {"kind": "summary"})
    return result


def notify_observation_dashboard(*, dry_run: bool = False, force: bool = False) -> dict[str, Any]:
    return notify_activity_summary(dry_run=dry_run, force=force)


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
