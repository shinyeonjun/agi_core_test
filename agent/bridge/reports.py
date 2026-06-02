from __future__ import annotations

import json
from typing import Any

from agent.bridge.formatter import compact_text, format_action_update, format_daily_summary, format_update_event
from agent.bridge.notifier import post_webhook
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import get_autonomy_state
from agent.core.cognitive_engine import cognitive_growth_snapshot
from agent.core.cooldown import is_ready, mark
from agent.core.events import list_events
from agent.core.goal_generator import list_goal_candidates, meaningful_open_goals
from agent.core.learner import list_reflections
from agent.core.metrics import collect_metrics
from agent.core.observability import action_failure_breakdown, action_observation, task_observation
from agent.core.operating_intelligence import operating_snapshot
from agent.core.self_map import self_map_brief
from agent.core.task_queue import list_tasks, task_status_counts
from agent.lab.proposals import list_action_proposals, proposal_status_counts
from agent.tools.action_log import get_action_run, list_action_runs


def build_daily_summary() -> str:
    metrics = collect_metrics()
    self_map = self_map_brief(max_age_seconds=300, refresh_if_stale=True, record_event_on_refresh=False)
    approvals = ApprovalStore().list(status=None, limit=20)
    actions = list_action_runs(20)
    goals = meaningful_open_goals(limit=10)
    content = format_daily_summary(metrics, approvals, actions, goals)
    if self_map:
        content += "\n\n**몸 상태**\n" + _line("최근 확인", self_map.get("summary"))
    return content


def _line(prefix: str, value: object) -> str:
    return f"- {prefix}: {compact_text(value)}"


def _ko_status(value: object) -> str:
    mapping = {
        "completed": "완료",
        "blocked": "차단",
        "timeout": "시간 초과",
        "running": "진행 중",
        "proposed": "제안됨",
        "executed": "실행됨",
        "rejected": "탈락",
        "dry_run": "미리보기",
        "candidate": "후보",
        "safe": "안전 모드",
        "full_device_lab": "장비 실험 모드",
        "workspace": "작업공간 모드",
        "queued": "대기",
        "done": "완료",
        "waiting_approval": "승인 대기",
    }
    raw = compact_text(value)
    return mapping.get(raw, raw)


def _ko_growth_mode(value: object) -> str:
    mapping = {
        "serve_user": "사용자 요청 우선",
        "stabilize": "안정화 우선",
        "explore": "탐색 우선",
        "consolidate": "정리/압축 우선",
    }
    raw = compact_text(value)
    return mapping.get(raw, raw)


def _ko_summary(value: object) -> str:
    raw = compact_text(value)
    mapping = {
        "rc=0": "정상 종료",
        "timeout": "시간 초과",
        "command_not_found": "명령 없음",
        "approval_required": "승인 필요",
        "env_access_denied": ".env 접근 차단",
        "secret_access_denied": "민감정보 접근 차단",
        "ssh_key_access_denied": "SSH 키 접근 차단",
        "profile_not_full_device_lab": "현재 모드에서 실행 차단",
        "root_delete_denied": "위험한 삭제 차단",
        "remote_script_execution_denied": "원격 스크립트 실행 차단",
    }
    return mapping.get(raw, raw.replace("_", " "))


def _ko_event(row: dict[str, Any] | None) -> str:
    if not row:
        return "없음"
    key = f"{row.get('source')}/{row.get('event_type')}"
    mapping = {
        "scheduler/idle_tick": "자동 tick 실행",
        "lab/lab_tick_skipped": "lab tick 제안 생성",
        "lab/lab_tick_timer_skipped": "lab timer 실행 건너뜀",
        "lab/lab_tick_executed": "lab action 실행",
        "lab/lab_tick_blocked": "lab 실행 후보 없음",
        "workspace/workspace_artifact_created": "작업공간 파일 생성",
        "policy/policy_check": "정책 검사 수행",
        "discord/discord_chat_reply": "대화 응답",
        "discord/action_update_notify_failed": "action 알림 실패",
        "core/assistant_output": "Core 대화 응답 생성",
        "core/decision_created": "Core 판단 기록 생성",
        "learner/reflection_created": "회고 기록 생성",
    }
    return mapping.get(key, key)


def _reflection_summary(value: object) -> str:
    raw = compact_text(value)
    mapping = {
        "Recorded talk feedback, selected goal, and retrieved context.": "대화 피드백과 목표/기억 맥락 저장",
        "Recorded idle action and cooldown state after tick.": "자동 tick 결과와 쿨다운 저장",
        "Lab tick generated proposals but did not execute because profile is not full_device_lab.": "lab tick이 제안만 만들고 실행은 건너뜀",
        "Lab tick executed one approved local action and recorded the result.": "lab tick이 승인된 로컬 action 1개 실행",
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
        return "현재 모드에서 로컬 실행 차단"
    if "df -h" in command or command.startswith("df "):
        return "디스크 상태 확인"
    if command.startswith("free "):
        return "메모리 상태 확인"
    if "systemctl --failed" in command:
        return "실패한 서비스 확인"
    if command.startswith("printf "):
        return "테스트 action 실행"
    if row.get("status") == "blocked":
        return "정책에 의해 action 차단"
    return "로컬 action 처리"


def _proposal_label(row: dict[str, Any]) -> str:
    reason = compact_text(row.get("reason"))
    mapping = {
        "profile_not_full_device_lab": "안전 모드라 실행 대기",
        "system_disk_check": "디스크 상태 점검 후보",
        "system_memory_check": "메모리 상태 점검 후보",
        "system_failed_services_check": "실패한 서비스 점검 후보",
        "lab_experiment_workspace_report": "작업공간 리포트 후보",
        "completed": "실행 완료",
    }
    return mapping.get(reason, reason.replace("_", " "))


def _format_rate(value: object) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _state_note(profile: object, autonomy: dict[str, Any]) -> str:
    if autonomy.get("catastrophic_local_destruction_allowed"):
        return "위험 삭제 arm 켜짐. 바로 확인 필요"
    if profile == "full_device_lab":
        return "승인된 로컬 점검을 자동 실행할 수 있음"
    if profile == "workspace":
        return "작업공간 중심으로 관찰 중"
    return "자동 로컬 실행은 잠김"


def _interesting_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy = {"discord_message", "discord_chat_reply", "webhook_sent", "webhook_missing", "discord_command_output"}
    return [event for event in events if event.get("event_type") not in noisy]


def _short_goal_title(value: object) -> str:
    title = compact_text(value)
    replacements = {
        "Generate a read-only Orange Pi health observation report": "오렌지파이 상태 보고서 만들기",
        "Compact old memories and reflections": "오래된 기억/회고 압축",
        "Draft a small project candidate from recent Core observations": "최근 관찰로 작은 프로젝트 후보 만들기",
        "Review recent Core state and cleanup opportunities": "Core 상태와 정리 후보 검토",
        "Create a workspace experiment report from recent Core activity": "최근 활동 기반 작업공간 리포트 만들기",
        "Improve action failure critic": "action 실패 원인 분류 개선",
        "Promote reflection patterns into skills": "반복 회고를 skill로 승격",
    }
    for raw, pretty in replacements.items():
        if title.startswith(raw):
            return pretty
    return title


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
    self_map = self_map_brief(max_age_seconds=300, refresh_if_stale=True, record_event_on_refresh=False)
    task_counts = task_status_counts()
    intelligence = operating_snapshot(persist=False)
    growth = cognitive_growth_snapshot(persist=False, limit=5)
    recent_tasks = list_tasks(limit=6)
    action_breakdown = action_failure_breakdown(actions)
    last_action = actions[0] if actions else None
    interesting_events = _interesting_events(events)
    last_event = interesting_events[0] if interesting_events else (events[0] if events else None)
    profile = metrics.get("current_autonomy_profile")

    lines = [
        "**Core 관제판**",
        "오렌지파이에서 Core 루프가 돌고 있어.",
        "",
        "**한눈에**",
        _line("모드", f"{_ko_status(profile)} - {_state_note(profile, autonomy)}"),
        _line("평가", f"{metrics.get('last_eval_result') or '없음'} / {metrics.get('last_eval_score') if metrics.get('last_eval_score') is not None else '-'}"),
        _line("승인 대기", f"{len(approvals)}건"),
        _line("최근 작업", f"#{last_action.get('id')} {_action_label(last_action)} - {_ko_status(last_action.get('status'))}" if last_action else "없음"),
        _line("최근 이벤트", f"#{last_event.get('id')} {_ko_event(last_event)}" if last_event else "없음"),
    ]
    if self_map:
        services = self_map.get("services") or {}
        active_count = len([state for state in services.values() if state == "active"])
        lines.append(_line("몸 상태", self_map.get("summary")))
        lines.append(_line("상주 루프", f"{active_count}개 active / self-map #{self_map.get('id')}"))

    lines.extend([
        "",
        "**24시간 지표**",
        _line("tick", f"{metrics.get('tick_count_24h')}회"),
        _line("Discord", f"{metrics.get('discord_messages_24h')}건"),
        _line("action 성공률", f"실행 {_format_rate(metrics.get('action_execution_success_rate_24h'))} / 전체 {_format_rate(metrics.get('action_success_rate_24h'))}"),
        _line("계획된 차단", _format_rate(metrics.get("action_planned_block_rate_24h"))),
        _line("차단/시간초과", f"{metrics.get('action_blocked_count_24h')}건 / {metrics.get('action_timeout_count_24h')}건"),
        _line("사용자 큐", f"대기 {task_counts.get('user:queued', 0)} / 실행 {task_counts.get('user:running', 0)}"),
        _line("자율 큐", f"대기 {task_counts.get('autonomous:queued', 0)} / 실행 {task_counts.get('autonomous:running', 0)}"),
    ])

    lines.extend(["", "**실패/차단 원인**"])
    if action_breakdown:
        for category, count in sorted(action_breakdown.items()):
            lines.append(f"- {category}: {count}건")
    else:
        lines.append("- 최근 action에는 실패/차단 원인이 없어.")

    lines.extend(["", "**작업 큐 상태**"])
    if recent_tasks:
        for task in recent_tasks[:3]:
            observed_task = task_observation(task)
            lifecycle = observed_task.get("lifecycle") or {}
            phase = lifecycle.get("last_label") or lifecycle.get("last_phase") or "기록 없음"
            lines.append(f"- #{observed_task['id']} {_short_goal_title(observed_task['title'])}: {observed_task['waiting_reason']} / {phase}")
    else:
        lines.append("- 현재 작업 큐가 비어 있어.")

    lines.extend(["", "**최근 action**"])
    if actions:
        for action in actions[:3]:
            observed_action = action_observation(action)
            lines.append(f"- #{action.get('id')} {_action_label(action)}: {observed_action['label']} / {observed_action['summary_label']}")
    else:
        lines.append("- 아직 기록된 action이 없어.")

    lines.extend(["", "**다음 후보**"])
    if goal_candidates:
        for candidate in goal_candidates[:3]:
            lines.append(f"- #{candidate.get('id')} {_short_goal_title(candidate.get('title'))} ({_ko_status(candidate.get('status'))})")
    else:
        lines.append("- 새 목표 후보가 없어.")

    improvements = intelligence.get("next_improvement_candidates") or []
    lines.extend(["", "**운영 지능**"])
    if improvements:
        for item in improvements[:3]:
            lines.append(f"- {_short_goal_title(item.get('title'))}: {compact_text(item.get('reason'))}")
    else:
        lines.append("- 지금 당장 급한 개선 후보는 없어.")
    critics = intelligence.get("action_critics") or []
    if critics:
        top_critic = next((row for row in critics if row.get("category") != "success"), critics[0])
        lines.append(f"- action critic: {compact_text(top_critic.get('label'))} / {compact_text(top_critic.get('recommendation'))}")
    memory_candidates = intelligence.get("memory_hygiene_candidates") or []
    if memory_candidates:
        lines.append(f"- memory: 압축/정리 후보 {len(memory_candidates)}건")
    skill_items = intelligence.get("skill_candidates") or []
    if skill_items:
        lines.append(f"- skill: 승격 후보 {len(skill_items)}건")
    inference = growth.get("active_inference") or {}
    curiosity = growth.get("curiosity") or []
    map_elites = growth.get("map_elites") or []
    if inference:
        lines.append(f"- 성장 루프: {_ko_growth_mode(inference.get('mode'))} / 압력 {inference.get('free_energy')}")
    if curiosity:
        lines.append(f"- 호기심: {compact_text(curiosity[0].get('topic'))} / 압력 {curiosity[0].get('pressure')}")
    if map_elites:
        lines.append(f"- 다양성 archive: {len(map_elites)}개 셀 유지")

    lines.extend(["", "**제안 큐**"])
    if pending_proposals:
        active_counts = {status: count for status, count in proposal_counts.items() if status not in {"executed", "rejected"}}
        count_text = ", ".join(f"{_ko_status(status)} {count}" for status, count in sorted(active_counts.items())) or "없음"
        lines.append(f"- 상태 합계: {count_text}")
        for proposal in pending_proposals[:3]:
            lines.append(f"- #{proposal.get('id')} {_proposal_label(proposal)}: {_ko_status(proposal.get('status'))}")
    else:
        lines.append("- 현재 처리 대기 중인 action 제안은 없어.")

    lines.extend(["", "**학습/회고**"])
    if reflections:
        for reflection in reflections[:2]:
            lines.append(f"- #{reflection.get('id')} {_reflection_summary(reflection.get('summary'))}")
    else:
        lines.append("- 아직 새 회고가 없어.")

    lines.extend(["", "**다음에 볼 것**"])
    if approvals:
        lines.append(f"- 승인 대기 {len(approvals)}건부터 확인해줘.")
    if goals:
        for goal in goals[:3]:
            lines.append(f"- #{goal.get('id')} {_short_goal_title(goal.get('title'))} ({_ko_status(goal.get('status'))})")
    if not approvals and not goals:
        lines.append("- 열린 목표가 없어. 다음 지시를 기다리는 중이야.")
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
        "요약 채널 테스트",
        "Core 요약 웹훅 연결을 확인하는 메시지야.",
        {"상태": "테스트", "민감정보": "전송 안 함"},
    )
    return post_webhook("summary", content, dry_run=dry_run)


def notify_test_update(*, dry_run: bool = False) -> dict[str, Any]:
    content = format_update_event(
        "업데이트 채널 테스트",
        "Core 실시간 업데이트 웹훅 연결을 확인하는 메시지야.",
        {"상태": "테스트", "다음": "action/eval/policy 이벤트"},
    )
    return post_webhook("update", content, dry_run=dry_run)


def notify_action(action_id: int, *, dry_run: bool = False) -> dict[str, Any]:
    return post_webhook("update", format_action_update(get_action_run(action_id)), dry_run=dry_run)
