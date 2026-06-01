from __future__ import annotations

import re
from typing import Any

from agent.core.goals import create_goal
from agent.core.policy import PolicyEngine
from agent.core.style import detect_style_feedback

TASK_KEYWORDS = (
    "해줘", "해봐", "하자", "가자", "ㄱㄱ", "만들", "구현", "개발", "디벨롭", "정리",
    "조사", "분석", "보고서", "작성", "추가", "수정", "개선", "고쳐", "처리", "실행",
    "테스트", "검증", "설계", "생성", "빌드", "스캐폴드", "초안",
    "implement", "create", "build", "fix", "develop", "write", "summarize", "test",
)
NON_TASK_PATTERNS = (
    re.compile(r"^\s*(ㅎㅇ|하이|안녕|hello|hi|hey)\s*[.!?ㅋㅎ]*\s*$", re.IGNORECASE),
    re.compile(r"(지금\s*)?(뭐\s*하고|뭐해|상태|살아|정상|도움|명령|사용법)", re.IGNORECASE),
)


def is_user_goal_request(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.startswith("!"):
        return False
    if detect_style_feedback(cleaned):
        return False
    if any(pattern.search(cleaned) for pattern in NON_TASK_PATTERNS):
        return False
    lowered = cleaned.lower()
    return any(keyword in lowered for keyword in TASK_KEYWORDS)


def classify_user_goal_kind(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["fastapi", "api", "프로젝트", "앱", "서비스", "봇", "사이트"]):
        return "project_spec"
    if any(token in lowered for token in ["보고서", "정리", "요약", "조사", "분석", "리포트"]):
        return "report"
    if any(token in lowered for token in ["개선", "수정", "고쳐", "리팩터", "디벨롭"]):
        return "improvement_plan"
    if any(token in lowered for token in ["테스트", "검증", "실험"]):
        return "workspace_experiment"
    return "task_note"


def _title_from_text(text: str) -> str:
    title = re.sub(r"\s+", " ", text.strip())
    return title[:90] if len(title) > 90 else title


def maybe_create_user_goal(text: str, *, source_event_id: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not is_user_goal_request(text):
        return None

    policy = PolicyEngine().classify_decision(text, action_type="user_directive")
    task_kind = classify_user_goal_kind(text)
    if policy.denied:
        status = "blocked"
    elif policy.requires_approval:
        status = "waiting_approval"
    else:
        status = "active"

    goal_metadata = {
        "source": "user_directive",
        "source_event_id": source_event_id,
        "task_kind": task_kind,
        "raw_user_text": text,
        "priority_owner": "user",
        "policy": policy.to_dict(),
    }
    goal_metadata.update(metadata or {})
    goal_id = create_goal(
        _title_from_text(text),
        text,
        goal_type="user_directed",
        status=status,
        priority=0.98,
        risk_level=policy.risk_level,
        requires_approval=policy.requires_approval,
        metadata=goal_metadata,
        dedupe=True,
    )
    return {
        "id": goal_id,
        "status": status,
        "goal_type": "user_directed",
        "title": _title_from_text(text),
        "task_kind": task_kind,
        "risk_level": policy.risk_level,
        "requires_approval": policy.requires_approval,
        "denied": policy.denied,
        "reason": policy.reason,
    }
