from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from agent.core.observability import action_observation, summary_label

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
    description = row.get("description") or proposal.get("description") or "승인이 필요한 작업"
    reason = proposal.get("denied_reason") or proposal.get("reason") or "사람 확인 필요"
    return "\n".join([
        f"**승인 필요 #{row.get('id')}**",
        f"작업: {compact_text(description)}",
        f"위험: {_risk_label(row.get('risk_level'))}",
        f"이유: {compact_text(reason)}",
        f"명령: `!approve {row.get('id')}` 또는 `!reject {row.get('id')}`",
    ])


def format_action_update(row: dict[str, Any] | None) -> str:
    if not row:
        return "**작업 업데이트**\n요청한 action을 찾지 못했어."
    command = _decode_command(row.get("command_json"))
    observed = action_observation(row)
    title = f"작업 {_status_title(row.get('status'))} #{row.get('id')}"
    purpose = _action_purpose(row, command)
    lines = [
        f"**{title}**",
        _action_sentence(row, purpose),
        "",
        f"영향: {_impact_label(row, command)}",
        f"다음: {observed['next_step']}",
    ]
    if observed["category"] != "success":
        lines.insert(3, f"분류: {observed['label']} - {observed['summary_label']}")
    else:
        lines.insert(3, "결과: 성공")
    if row.get("status") == "blocked":
        lines.insert(4, "결과: 차단")
    return "\n".join(lines)


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


def _command_text(command: list[str]) -> str:
    return " ".join(part for part in command if part and part != "-").strip()


def _action_purpose(row: dict[str, Any], command: list[str]) -> str:
    summary = compact_text(row.get("result_summary"))
    text = _command_text(command).lower()
    if summary == "profile_not_full_device_lab":
        return "자동 로컬 실행"
    if summary == "ssh_key_access_denied":
        return "SSH 키 접근"
    if summary == "root_delete_denied":
        return "위험한 삭제"
    if "agentctl workspace report" in text:
        return "작업공간 상태 보고서"
    if "agentctl eval run" in text:
        return "Core 평가"
    if "agentctl audit" in text:
        return "Core 안전 점검"
    if "pytest" in text:
        return "테스트"
    if text.startswith("git status") or " git status" in text:
        return "Git 상태 확인"
    if "df -h /" in text or text == "df -h /":
        return "루트 디스크 상태"
    if text.startswith("df "):
        return "디스크 상태 확인"
    if text.startswith("free "):
        return "메모리 상태 확인"
    if "systemctl --failed" in text:
        return "실패한 서비스 확인"
    if text.startswith("printf "):
        return "lab 동작 테스트"
    if row.get("status") == "blocked":
        return "정책 점검"
    return "로컬 작업"


def _action_sentence(row: dict[str, Any], purpose: str) -> str:
    status = compact_text(row.get("status"))
    if status == "completed":
        if purpose == "루트 디스크 상태":
            return "루트 디스크 상태를 확인했어."
        if purpose.endswith("확인"):
            return f"{purpose}을 끝냈어."
        if purpose.endswith("점검") or purpose.endswith("평가") or purpose == "테스트":
            return f"{purpose}을 완료했어."
        if purpose.endswith("보고서"):
            return f"{purpose}를 만들었어."
        return f"{purpose}을 완료했어."
    if status == "blocked":
        return f"{purpose}은 실행하지 않았어."
    if status == "timeout":
        return f"{purpose} 중 시간이 초과됐어."
    if status == "failed":
        return f"{purpose} 중 실패했어."
    if status == "running":
        return f"{purpose}이 진행 중이야."
    return f"{purpose} 상태가 업데이트됐어."


def _status_title(value: object) -> str:
    mapping = {
        "completed": "완료",
        "blocked": "차단",
        "timeout": "시간 초과",
        "failed": "실패",
        "running": "진행 중",
    }
    raw = compact_text(value)
    return mapping.get(raw, raw)


def _risk_label(value: object) -> str:
    mapping = {
        "low": "낮음",
        "medium": "보통",
        "high": "높음",
        "critical": "위험",
    }
    raw = compact_text(value)
    return mapping.get(raw, raw)


def _impact_label(row: dict[str, Any], command: list[str]) -> str:
    status = compact_text(row.get("status"))
    text = _command_text(command).lower()
    readonly_prefixes = ("df ", "free ", "pwd", "ls ", "find ", "git status")
    if status == "blocked":
        return "실행 안 됨, 시스템 변경 없음"
    if text.startswith(readonly_prefixes) or "systemctl --failed" in text:
        return "읽기 전용, 시스템 변경 없음"
    if "agentctl workspace report" in text:
        return "보고서 파일 생성"
    if "pytest" in text or "agentctl eval run" in text or "agentctl audit" in text:
        return "검증 실행, 시스템 설정 변경 없음"
    return "로컬 명령 1회 실행"


def format_daily_summary(metrics: dict[str, Any], approvals: list[dict[str, Any]], actions: list[dict[str, Any]], goals: list[dict[str, Any]]) -> str:
    pending = len([row for row in approvals if row.get("status") == "pending"])
    completed = len([row for row in actions if row.get("status") == "completed"])
    blocked = len([row for row in actions if row.get("status") == "blocked"])
    eval_result = compact_text(metrics.get("last_eval_result"))
    eval_score = compact_text(metrics.get("last_eval_score"))
    lines = [
        "**Core 요약**",
        f"오늘은 action {completed}개를 끝냈고, 차단은 {blocked}개였어.",
        "",
        "**한눈에**",
        f"- 평가: {eval_result} / {eval_score}",
        f"- 승인 대기: {pending}건",
        f"- 기억/회고: memory {compact_text(metrics.get('memories_count'))}, reflection {compact_text(metrics.get('reflections_count'))}",
        "",
        "**다음에 볼 것**",
    ]
    meaningful_goals = [goal for goal in goals if not _is_noise_title(goal.get("title"))]
    if meaningful_goals:
        for goal in meaningful_goals[:3]:
            lines.append(f"- #{goal.get('id')} {compact_text(goal.get('title'))} ({compact_text(goal.get('status'))})")
    else:
        lines.append("- 열린 중요 목표가 거의 없어. 다음 지시를 기다리는 중이야.")
    return "\n".join(lines)


def format_update_event(title: str, detail: str, fields: dict[str, Any] | None = None) -> str:
    lines = [f"**{compact_text(title)}**", compact_text(detail)]
    for key, value in (fields or {}).items():
        if str(key).lower() in {"profile", "risk", "risk_level", "returncode", "raw"}:
            continue
        lines.append(f"- {key}: {compact_text(value)}")
    return "\n".join(lines)


def _core_chat_text(core_result: dict[str, Any]) -> str | None:
    text = compact_text(core_result.get("text"), "").strip()
    if not text:
        return None
    lowered = text.lower()
    internal_markers = (
        "fallback renderer response",
        "core saved the input as an event",
        "- goal:",
        "- renderer:",
        "relevant memories:",
        "relevant skills:",
        "selected_goal_id",
        "user_goal_created",
        "policy_summary",
        "language_interpretation",
        "source_event_id",
        "must_include",
        "must_not_include",
        "요청은 core 작업 흐름",
        "core 답변 렌더러",
        "core가 지금 입력은 기록",
    )
    if any(marker in lowered for marker in internal_markers):
        return None
    return text


def format_chat_reply(user_text: str, core_result: dict[str, Any]) -> str:
    decision = core_result.get("decision") or {}
    policy = decision.get("policy_summary") or {}
    user_goal = decision.get("user_directed_goal") or {}
    task_result = core_result.get("task_result") or {}
    if policy.get("denied"):
        reason = compact_text(policy.get("reason") or policy.get("denied_reason") or "정책 차단")
        return f"위험해서 실행하지 않았어.\n이유: {reason}"

    if user_goal and user_goal.get("control_action"):
        return _format_user_goal(user_goal)

    natural = _natural_chat_reply(core_result)
    if natural:
        return natural

    if user_goal and task_result:
        return _format_task_result(user_goal, task_result)
    if user_goal:
        return _format_user_goal(user_goal)
    return _renderer_unavailable_reply(decision)


def _natural_chat_reply(core_result: dict[str, Any]) -> str | None:
    task_result = core_result.get("task_result") or {}
    report = compact_text(task_result.get("report"), "").strip() if isinstance(task_result, dict) else ""
    if report:
        return report[:1600]
    return _core_chat_text(core_result)


def _format_task_result(user_goal: dict[str, Any], task_result: dict[str, Any]) -> str:
    status = compact_text(task_result.get("status"))
    if status in {"user_goal_completed", "artifact_created", "completed", "done", "codex_work_completed"}:
        report = compact_text(task_result.get("report"), "")
        if report:
            return report[:1600]
        result_label = compact_text(task_result.get("artifact_type") or status)
        return f"작업 처리됨.\n결과: {result_label}"
    if status in {"codex_work_blocked", "codex_work_failed"}:
        reason = compact_text(task_result.get("reason") or status)
        return f"작업 #{compact_text(task_result.get('task_id'))}는 끝까지 못 갔어.\n이유: {reason}"
    if status == "skipped":
        return f"작업 #{compact_text(task_result.get('task_id'))}는 이번엔 건너뛰었어.\n이유: {compact_text(task_result.get('reason'))}"
    return f"작업 상태가 갱신됐어.\n상태: {status}"


def _format_user_goal(user_goal: dict[str, Any]) -> str:
    if user_goal.get("control_action") == "cancel":
        result = user_goal.get("cancel_result") if isinstance(user_goal.get("cancel_result"), dict) else {}
        status = compact_text(result.get("status"))
        if status == "missing_target":
            return "어떤 걸 취소할지 번호가 필요해. 예: `!cancel 524` 또는 `#524 없애줘`."
        if status == "not_found":
            return f"#{compact_text(result.get('target_id'))}는 목표/작업 목록에서 못 찾았어."
        tasks = int(result.get("cancelled_tasks") or 0)
        goals = int(result.get("archived_goals") or 0)
        return f"정리했어. 작업 {tasks}개를 멈추고, 목표 {goals}개를 목록에서 뺐어."
    if user_goal.get("self_improvement"):
        ticket = user_goal.get("ticket") if isinstance(user_goal.get("ticket"), dict) else {}
        if user_goal.get("status") == "blocked" or not user_goal.get("task_id"):
            return "\n".join(
                [
                    "자가개선 작업은 새로 접수하지 못했어.",
                    f"이유: {compact_text(user_goal.get('reason'), '생성 결과 없음')}",
                    "다음: 막힌 목표를 정리한 뒤 다시 시도해야 해.",
                ]
            )
        return "\n".join(
            [
                f"자가개선 작업 #{compact_text(user_goal.get('task_id'))} 접수했어.",
                "상태: 대기 중",
                f"목표: {compact_text(ticket.get('title') or user_goal.get('title'))}",
                "진행 로그는 #업데이트에 보낼게.",
                "main 반영이 필요하면 #승인에서 물어볼게.",
            ]
        )
    if not (user_goal.get("denied") or user_goal.get("status") in {"blocked", "waiting_approval"}):
        return "\n".join(
            [
                f"작업 #{compact_text(user_goal.get('task_id'))} 접수했어.",
                "상태: 대기 중",
                "진행 로그는 #업데이트에 보낼게.",
            ]
        )
    if user_goal.get("self_improvement"):
        ticket = user_goal.get("ticket") if isinstance(user_goal.get("ticket"), dict) else {}
        return "\n".join(
            [
                "자가개선 작업으로 잡았어.",
                f"목표: {compact_text(ticket.get('title') or user_goal.get('title'))}",
                f"작업: #{compact_text(user_goal.get('task_id'))}",
                "방식: 별도 작업공간에서 수정하고 테스트/audit/eval을 통과해야 해.",
                "main 반영은 승인 전에는 안 해.",
            ]
        )
    if user_goal.get("denied") or user_goal.get("status") == "blocked":
        return f"위험해서 실행 목표로는 잠가뒀어.\n이유: {compact_text(user_goal.get('reason'))}"
    if user_goal.get("status") == "waiting_approval":
        return "승인이 필요한 작업이라 대기 상태로 뒀어."
    return "작업으로 넘겼어."


def _renderer_unavailable_reply(decision: dict[str, Any]) -> str:
    interpretation = decision.get("language_interpretation") if isinstance(decision.get("language_interpretation"), dict) else {}
    target = str(interpretation.get("target") or "")
    user_input = compact_text(decision.get("user_input"), "")
    if "자율" in user_input and "생명체" in user_input:
        return "완전한 생명체나 의식은 아니야. 다만 목표, 기억, 실행, 검증, 실패 학습 루프를 강화해서 자율 에이전트처럼 운용하는 건 가능해."
    if target == "capabilities":
        return "가능한 건 대화, 기억 검색, 목표/작업 관리, 안전한 로컬 점검, 코드 작업 위임이야. 위험한 시스템 변경은 승인 없이는 못 해."
    if target == "architecture":
        return "Core는 대화 해석, 기억, 목표/작업 큐, 정책 게이트, 장비 관찰, Codex 작업 워커가 나뉘어 돌아가는 구조야."
    if target == "status":
        return "상태 확인은 가능해. 현재 작업은 `!work`, 열린 목표는 `!goals`, 장비 상태는 `!state`로 보면 돼."
    return "답변 생성이 잠깐 매끄럽지 않았어. 그래도 입력은 받았고, 작업 지시면 `!work`에서 진행 여부를 확인하면 돼."


def _is_noise_title(value: object) -> bool:
    title = compact_text(value, "").lower()
    return any(token in title for token in ["secret goal summary marker", "answer user input", "apply user negative feedback"])
