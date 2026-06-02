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


def _looks_like_greeting(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {"hi", "hello", "hey", "ㅎㅇ", "하이", "안녕", "안녕하세요", "헬로", "ㅇㅇ"} or lowered.startswith(("ㅎㅇ", "안녕"))


def _asks_for_status(text: str) -> bool:
    normalized = text.replace(" ", "")
    return any(token in text for token in ["상태", "뭐 하고", "뭐 하는", "뭐해", "살아", "정상", "체크", "확인"]) or any(
        token in normalized for token in ["뭐하고", "뭐하는", "뭐해", "하고있", "하는중"]
    )


def _asks_for_help(text: str) -> bool:
    return any(token in text.lower() for token in ["도움", "명령", "help", "뭐 할", "사용법", "기능"])


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
    )
    if any(marker in lowered for marker in internal_markers):
        return None
    return text


def format_chat_reply(user_text: str, core_result: dict[str, Any]) -> str:
    decision = core_result.get("decision") or {}
    policy = decision.get("policy_summary") or {}
    user_goal = decision.get("user_directed_goal") or {}
    task_result = core_result.get("task_result") or {}
    style_feedback = decision.get("style_feedback") or {}
    interpretation = decision.get("language_interpretation") or {}
    intent = interpretation.get("intent")
    target = interpretation.get("target")
    sentiment = interpretation.get("sentiment")
    text = redact_discord_content(user_text).strip()

    if style_feedback:
        feedback_type = compact_text(style_feedback.get("feedback_type"))
        return f"좋아. 말투 피드백으로 기억해뒀어.\n반영: {feedback_type}\n실행/정책 판단은 그대로 두고 말하는 방식만 조정할게."
    if user_goal and task_result:
        return _format_task_result(user_goal, task_result)
    if user_goal:
        return _format_user_goal(user_goal)
    if policy.get("denied"):
        reason = compact_text(policy.get("reason") or policy.get("denied_reason") or "정책 차단")
        return f"그 요청은 실행하지 않았어.\n이유: {reason}"

    rendered = _core_chat_text(core_result)
    if rendered:
        return rendered
    if target == "capabilities":
        return _capability_reply(decision)
    if intent == "feedback":
        if sentiment == "positive":
            return "좋아. 그 방향이 맞다는 피드백으로 기억해둘게."
        if sentiment == "negative":
            return "오케이. 방금 건 별로였다는 피드백으로 남기고 다음 답변에서 조정할게."
    if intent == "brainstorm":
        return "아이디어 검토로 이해했어. 장점, 걸리는 점, 다음 실험 단위로 나눠서 볼게."
    if target == "greeting" or _looks_like_greeting(text):
        return "응, 여기 있어. 편하게 말해줘."
    if target == "help" or _asks_for_help(text):
        return "그냥 자연어로 말하면 돼. 작업은 #대화, 승인은 #승인, 보고는 #요약, 실시간 로그는 #업데이트로 나눠서 처리할게."
    if target == "architecture":
        return "Core는 LanguageEngine, PolicyEngine, 기억, 목표, 실행, 평가, Discord 관제로 나뉘어 굴러가. 지금은 언어 판단은 Codex가 맡고, Core는 상태와 실행 경계를 관리하는 구조야."
    if target == "status" or _asks_for_status(text):
        metrics = decision.get("metrics") or {}
        profile = metrics.get("current_autonomy_profile") or decision.get("autonomy_profile") or "safe"
        eval_result = metrics.get("last_eval_result") or "unknown"
        return f"지금은 Discord 대화 채널을 듣고, 오렌지파이에서 tick 루프가 돌고 있어.\n모드: `{profile}`\n최근 평가: `{eval_result}`\n자세한 건 `!state`로 볼 수 있어."
    if target == "question" or text.endswith("?") or text.endswith("？"):
        return "질문으로 이해했어. 조금만 더 구체적으로 말해주면 그 기준으로 바로 답할게."
    if len(text) <= 20:
        return "응, 들었어. 다음에 뭘 하면 될지 바로 말해줘."
    return "좋아, 이해했어. 이건 작업 방향으로 받아서 Core에 반영할게."


def _format_task_result(user_goal: dict[str, Any], task_result: dict[str, Any]) -> str:
    status = compact_text(task_result.get("status"))
    if status in {"user_goal_completed", "artifact_created", "completed", "done", "codex_work_completed"}:
        report = compact_text(task_result.get("report"), "")
        report_line = f"\n요약: {report[:700]}" if report else ""
        result_label = compact_text(task_result.get("artifact_type") or status)
        return f"완료했어. 작업 #{compact_text(task_result.get('task_id'))}, 목표 #{compact_text(user_goal.get('id'))} 처리됨.\n결과: {result_label}{report_line}\n자율 스케줄러와 분리해서 사용자 요청으로 바로 처리했어."
    if status in {"codex_work_blocked", "codex_work_failed"}:
        reason = compact_text(task_result.get("reason") or status)
        return f"작업 #{compact_text(task_result.get('task_id'))}는 끝까지 못 갔어.\n이유: {reason}"
    if status == "skipped":
        return f"작업 #{compact_text(task_result.get('task_id'))}는 이번엔 건너뛰었어.\n이유: {compact_text(task_result.get('reason'))}"
    return f"작업 상태가 갱신됐어.\n상태: {status}"


def _format_user_goal(user_goal: dict[str, Any]) -> str:
    if user_goal.get("denied") or user_goal.get("status") == "blocked":
        return f"그 작업은 위험할 수 있어서 실행 목표로는 잠가뒀어.\n목표: #{user_goal.get('id')}\n이유: {compact_text(user_goal.get('reason'))}"
    if user_goal.get("status") == "waiting_approval":
        return f"좋아. 목표 #{user_goal.get('id')}로 등록했고, 승인 대기 상태야."
    return f"좋아. 목표 #{user_goal.get('id')}로 등록했어. 사용자 요청이라 자율 작업보다 먼저 볼게."


def _capability_reply(decision: dict[str, Any]) -> str:
    capabilities = decision.get("capability_map") or {}
    workers = {item.get("name"): item for item in capabilities.get("worker_mediated", []) if isinstance(item, dict)}
    codex_work = workers.get("codex_work_worker") or {}
    worker_status = compact_text(codex_work.get("status") or "unknown")
    worker_backend = compact_text(codex_work.get("backend") or "codex")
    return "\n".join([
        "냉정하게 지금 할 수 있는 건 이 정도야.",
        "- Discord 대화 받고 목표/작업으로 분류하기",
        "- 기억, 목표, 회고, 평가 결과 쌓기",
        "- 정책 안에서 오렌지파이 상태 점검하기",
        "- 승인 필요한 일은 멈추고 대기시키기",
        f"- 개발 작업 worker: `{worker_status}` / backend `{worker_backend}`",
        "약한 건 장기 계획을 끝까지 밀어붙이는 힘이라 worker 루프를 계속 키워야 해.",
    ])


def _is_noise_title(value: object) -> bool:
    title = compact_text(value, "").lower()
    return any(token in title for token in ["secret goal summary marker", "answer user input", "apply user negative feedback"])
