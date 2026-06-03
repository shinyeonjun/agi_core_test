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
        "rejected": "보류",
        "dry_run": "미리보기",
        "candidate": "후보",
        "safe": "안전 모드",
        "full_device_lab": "장비 실험 모드",
        "workspace": "작업공간 모드",
        "queued": "대기",
        "active": "진행 중",
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
        "scheduler/idle_tick": "정기 점검 실행",
        "lab/lab_tick_skipped": "장비 점검 후보 생성",
        "lab/lab_tick_timer_skipped": "장비 점검 건너뜀",
        "lab/lab_tick_executed": "장비 점검 실행",
        "lab/lab_tick_blocked": "실행할 장비 점검 없음",
        "workspace/workspace_artifact_created": "작업공간 파일 생성",
        "policy/policy_check": "정책 검사 수행",
        "discord/discord_chat_reply": "대화 응답",
        "discord/action_update_notify_failed": "작업 알림 실패",
        "core/assistant_output": "Core 대화 응답 생성",
        "core/decision_created": "Core 판단 기록 생성",
        "learner/reflection_created": "회고 기록 생성",
        "goal/root_objectives_seeded": "기본 목표 목록 갱신",
        "reactor/wake_signal_completed": "루프 신호 처리 완료",
        "reactor/reactor_cycle_completed": "대기 중인 작업 확인 완료",
        "reactor/reactor_sleep": "다음 확인까지 대기",
    }
    return mapping.get(key, key)


def _reflection_summary(value: object) -> str:
    raw = compact_text(value)
    mapping = {
        "Recorded talk feedback, selected goal, and retrieved context.": "대화 피드백과 목표/기억 맥락 저장",
        "Recorded idle action and cooldown state after tick.": "정기 점검 결과 저장",
        "Lab tick generated proposals but did not execute because profile is not full_device_lab.": "장비 점검 후보만 만들고 실행은 건너뜀",
        "Lab tick executed one approved local action and recorded the result.": "승인된 로컬 점검 1개 실행",
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
        return "테스트 작업 실행"
    if row.get("status") == "blocked":
        return "정책에 의해 작업 차단"
    return "로컬 작업 처리"


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
    noisy = {"discord_message", "discord_chat_reply", "webhook_sent", "webhook_missing", "discord_command_output", "root_objectives_seeded"}
    return [event for event in events if event.get("event_type") not in noisy]


def _short_goal_title(value: object) -> str:
    title = compact_text(value)
    replacements = {
        "Generate a read-only Orange Pi health observation report": "오렌지파이 상태 보고서 만들기",
        "Compact old memories and reflections": "오래된 기억/회고 압축",
        "Draft a small project candidate from recent Core observations": "최근 관찰로 작은 프로젝트 후보 만들기",
        "Review recent Core state and cleanup opportunities": "Core 상태와 정리 후보 검토",
        "Create a workspace experiment report from recent Core activity": "최근 활동 기반 작업공간 리포트 만들기",
        "Improve action failure critic": "작업 실패 원인 분류 개선",
        "Promote reflection patterns into skills": "반복 회고를 배워둘 규칙으로 정리",
        "Write a research note on autonomous goal quality": "자율 목표 품질 연구 메모 작성",
        "Review recent action patterns for skill growth": "최근 작업 패턴을 보고 스킬 후보 찾기",
        "Review memory retrieval quality": "기억 검색 품질 점검",
        "Consolidate memory and reflection pressure": "쌓인 기억과 회고 정리",
        "Self maintenance": "Core 상태 정리",
        "System observation": "장비 상태 관찰",
        "Workspace experiment": "작업공간 실험",
        "Project incubation": "작은 프로젝트 후보 만들기",
        "Self-improvement proposal": "Core 개선안 작성",
        "Skill growth": "스킬 성장 후보 검토",
        "Research loop": "연구 메모 작성",
        "memory_retrieval": "기억 검색 품질",
        "memory_hygiene": "기억 정리",
        "action_failure": "작업 실패 분석",
        "skill_promotion": "스킬 승격",
    }
    for raw, pretty in replacements.items():
        if title.startswith(raw):
            return pretty
    return title


def _ko_goal_candidate_reason(candidate: dict[str, Any]) -> str:
    reason = compact_text(candidate.get("rejection_reason") or candidate.get("reason"), "")
    mapping = {
        "cooldown": "최근에 비슷한 후보를 이미 봐서 잠시 보류",
        "duplicate_open_goal": "이미 열린 비슷한 목표가 있어서 보류",
        "duplicate_recent_candidate": "최근에 나온 후보라 중복 방지",
        "low_score": "지금 우선순위가 낮음",
        "unsafe": "안전 기준에 맞지 않음",
        "": "지금은 실행하지 않고 후보로만 기록",
    }
    return mapping.get(reason, reason.replace("_", " ") if reason else mapping[""])


def _human_self_map(self_map: dict[str, Any] | None) -> str | None:
    if not self_map:
        return None
    os_name = compact_text(self_map.get("os"), "")
    host = compact_text(self_map.get("hostname"), "")
    profile = _ko_status(self_map.get("autonomy_profile"))
    services = self_map.get("services") or {}
    active_count = len([state for state in services.values() if state == "active"])
    eval_row = self_map.get("latest_eval") or {}
    eval_text = f"{eval_row.get('result')} / {eval_row.get('score')}" if eval_row else "기록 없음"
    parts = []
    if os_name:
        parts.append(os_name if not host else f"{host}의 {os_name}")
    parts.append(f"{profile}로 실행 중")
    parts.append(f"상주 루프 {active_count}개")
    parts.append(f"최근 평가 {eval_text}")
    return ", ".join(parts)


def _plain_count(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _human_sentence(value: object) -> str:
    text = compact_text(value)
    replacements = {
        "action": "작업",
        "Action": "작업",
        "skill": "스킬",
        "Skill": "스킬",
        "tick": "정기 점검",
        "Tick": "정기 점검",
        "lab": "장비 점검",
        "Lab": "장비 점검",
        "scheduler": "자동 루프",
        "Scheduler": "자동 루프",
        "self-map": "장비 상태",
        "Self-map": "장비 상태",
        "자율 스케줄러": "뒤쪽 자동 루프",
        "memory": "기억",
        "Memory": "기억",
        "reflection": "회고",
        "Reflection": "회고",
    }
    for raw, pretty in replacements.items():
        text = text.replace(raw, pretty)
    return text.replace("_", " ")


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
        "**Core 상태 요약**",
        "오렌지파이에서 정상적으로 돌고 있어.",
        "",
        "**현재 상태**",
        _line("모드", f"{_ko_status(profile)} - {_state_note(profile, autonomy)}"),
        _line("평가", f"{metrics.get('last_eval_result') or '없음'} / {metrics.get('last_eval_score') if metrics.get('last_eval_score') is not None else '-'}"),
        _line("승인 대기", f"{len(approvals)}건"),
    ]
    if self_map:
        lines.append(_line("장비 상태", _human_self_map(self_map)))

    lines.extend(["", "**최근 처리**"])
    recent_lines: list[str] = []
    if last_action:
        recent_lines.append(f"#{last_action.get('id')} {_action_label(last_action)} - {_ko_status(last_action.get('status'))}")
    for task in recent_tasks[:2]:
        observed_task = task_observation(task)
        recent_lines.append(f"#{observed_task['id']} {_short_goal_title(observed_task['title'])} - {_human_sentence(observed_task['waiting_reason'])}")
    if last_event and len(recent_lines) < 3:
        recent_lines.append(f"#{last_event.get('id')} {_ko_event(last_event)}")
    if recent_lines:
        lines.extend(f"- {line}" for line in recent_lines[:3])
    else:
        lines.append("- 아직 새로 처리한 일이 없어.")

    user_waiting = _plain_count(task_counts.get("user:queued")) + _plain_count(task_counts.get("user:running"))
    autonomous_waiting = _plain_count(task_counts.get("autonomous:queued")) + _plain_count(task_counts.get("autonomous:running"))
    blocked_count = _plain_count(metrics.get("action_blocked_count_24h"))
    timeout_count = _plain_count(metrics.get("action_timeout_count_24h"))

    improvements = intelligence.get("next_improvement_candidates") or []
    memory_candidates = intelligence.get("memory_hygiene_candidates") or []
    skill_items = intelligence.get("skill_candidates") or []
    lines.extend(["", "**주의할 점**"])
    notes: list[str] = []
    if user_waiting:
        notes.append(f"사용자 작업 {user_waiting}건이 아직 처리 중이거나 대기 중이야.")
    if autonomous_waiting:
        notes.append(f"뒤에서 처리할 자율 작업 {autonomous_waiting}건이 남아 있어.")
    if approvals:
        notes.append(f"사람 승인이 필요한 항목 {len(approvals)}건이 있어.")
    if blocked_count or timeout_count:
        notes.append(f"최근 24시간에 차단 {blocked_count}건, 시간 초과 {timeout_count}건이 있었어.")
    if memory_candidates:
        notes.append(f"기억/회고가 쌓여서 정리 후보 {len(memory_candidates)}건이 있어.")
    if skill_items:
        notes.append(f"반복된 작업 패턴에서 배워둘 후보 {len(skill_items)}건을 찾았어.")
    if not notes:
        notes.append("지금 당장 눈에 띄는 문제는 없어.")
    lines.extend(f"- {note}" for note in notes[:5])

    lines.extend(["", "**자동 제안 기준**"])
    lines.append("- 최근 기록, 실패/성공 패턴, 기억 누적, 열린 목표를 보고 안전한 후보만 만든다.")
    lines.append("- 중복이거나 우선순위가 낮거나 지금 실행할 필요가 없으면 보류로 남긴다.")

    lines.extend(["", "**다음 후보**"])
    if goal_candidates:
        for candidate in goal_candidates[:3]:
            lines.append(f"- #{candidate.get('id')} {_short_goal_title(candidate.get('title'))}: {_ko_status(candidate.get('status'))} - {_ko_goal_candidate_reason(candidate)}")
    else:
        lines.append("- 새 목표 후보가 없어.")

    lines.extend(["", "**운영 판단**"])
    if improvements:
        for item in improvements[:2]:
            lines.append(f"- {_short_goal_title(item.get('title'))}: {_human_sentence(item.get('reason'))}")
    else:
        lines.append("- 지금 당장 급한 개선 후보는 없어.")
    critics = intelligence.get("action_critics") or []
    if critics:
        top_critic = next((row for row in critics if row.get("category") != "success"), critics[0])
        lines.append(f"- 최근 작업 판단: {_human_sentence(top_critic.get('recommendation'))}")
    inference = growth.get("active_inference") or {}
    curiosity = growth.get("curiosity") or []
    if inference:
        lines.append(f"- 자율 개선 방향: {_ko_growth_mode(inference.get('mode'))}")
    if curiosity:
        lines.append(f"- 지금 가장 살펴보는 주제: {_short_goal_title(curiosity[0].get('topic'))}")

    lines.extend(["", "**실행 제안**"])
    if pending_proposals:
        active_counts = {status: count for status, count in proposal_counts.items() if status not in {"executed", "rejected"}}
        count_text = ", ".join(f"{_ko_status(status)} {count}" for status, count in sorted(active_counts.items())) or "없음"
        lines.append(f"- 상태 합계: {count_text}")
        for proposal in pending_proposals[:3]:
            lines.append(f"- #{proposal.get('id')} {_proposal_label(proposal)}: {_ko_status(proposal.get('status'))}")
    else:
        lines.append("- 지금 바로 실행 대기 중인 제안은 없어.")

    lines.extend(["", "**최근 배운 것**"])
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
