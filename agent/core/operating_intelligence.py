from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db
from agent.core.goals import goal_metadata
from agent.core.metrics import collect_metrics
from agent.core.observability import action_observation
from agent.memory.store import list_memories
from agent.tools.action_log import list_action_runs

OPEN_GOAL_STATUSES = ("proposed", "active", "waiting_approval", "blocked")


def _json(value: dict[str, Any] | list[Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _decode_json(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _age_hours(created_at: object) -> float:
    if not created_at:
        return 0.0
    try:
        created = datetime.fromisoformat(str(created_at))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(created.tzinfo or KST) - created).total_seconds() / 3600)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def list_open_goals(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    placeholders = ",".join("?" for _ in OPEN_GOAL_STATUSES)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM goals
            WHERE status IN ({placeholders})
            ORDER BY priority DESC, id DESC
            LIMIT ?
            """,
            (*OPEN_GOAL_STATUSES, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def score_goal(goal: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or collect_metrics()
    metadata = goal_metadata(goal)
    base = float(goal.get("priority") or 0.5)
    score = base
    reasons: list[str] = [f"base={base:.2f}"]
    goal_type = str(goal.get("goal_type") or "")
    status = str(goal.get("status") or "")
    risk = str(goal.get("risk_level") or "low")
    age = _age_hours(goal.get("created_at"))

    if goal_type == "user_directed" or metadata.get("priority_owner") == "user":
        score += 0.35
        reasons.append("user_owned:+0.35")
    if status == "waiting_approval":
        score += 0.06
        reasons.append("waiting_approval:+0.06")
    if status == "blocked":
        score -= 0.22
        reasons.append("blocked:-0.22")
    if risk in {"high", "critical"}:
        score -= 0.10 if risk == "high" else 0.18
        reasons.append(f"risk_{risk}")
    if age >= 24:
        bump = min(0.12, age / 240)
        score += bump
        reasons.append(f"age:+{bump:.2f}")
    if goal_type in {"memory_cleanup", "skill_review"} and int(metrics.get("reflections_count") or 0) >= 100:
        score += 0.08
        reasons.append("reflection_pressure:+0.08")
    if goal_type == "system_observation" and float(metrics.get("action_success_rate_24h") or 0.0) < 0.85:
        score += 0.05
        reasons.append("runtime_quality:+0.05")
    if goal_type == "self_improvement_proposal":
        score += 0.04
        reasons.append("self_improvement:+0.04")

    return {
        "id": goal.get("id"),
        "title": goal.get("title"),
        "goal_type": goal_type,
        "status": status,
        "risk_level": risk,
        "old_priority": round(base, 4),
        "priority_score": round(_clamp(score), 4),
        "reasons": reasons,
    }


def ranked_goals(limit: int = 20) -> list[dict[str, Any]]:
    metrics = collect_metrics()
    rows = [score_goal(goal, metrics) for goal in list_open_goals(limit=100)]
    return sorted(rows, key=lambda row: (row["priority_score"], row.get("id") or 0), reverse=True)[:limit]


def refresh_goal_priorities(limit: int = 100) -> dict[str, Any]:
    ranked = ranked_goals(limit=limit)
    init_db()
    changed = 0
    with connect() as conn:
        for row in ranked:
            goal_id = int(row["id"])
            old = float(row["old_priority"])
            new = float(row["priority_score"])
            if abs(old - new) < 0.005:
                continue
            conn.execute(
                """
                UPDATE goals
                SET priority = ?, updated_at = ?
                WHERE id = ?
                """,
                (new, now_kst(), goal_id),
            )
            changed += 1
        conn.commit()
    return {"changed": changed, "items": ranked}


def action_critics(limit: int = 20) -> list[dict[str, Any]]:
    critics: list[dict[str, Any]] = []
    for action in list_action_runs(limit):
        observed = action_observation(action)
        category = observed["category"]
        if category == "success":
            recommendation = "성공 패턴으로 유지"
            retry = False
        elif category in {"profile_block", "approval_required"}:
            recommendation = "권한/승인 조건을 먼저 만족시킨 뒤 재시도"
            retry = False
        elif category == "timeout":
            recommendation = "timeout 증가보다 명령 축소나 단계 분리를 먼저 검토"
            retry = True
        elif category == "command_not_found":
            recommendation = "도구 존재 여부와 설치/경로 조건을 확인"
            retry = False
        elif category == "command_failed":
            recommendation = "stderr/returncode를 근거로 실패 유형을 더 세분화"
            retry = True
        else:
            recommendation = "분류 규칙 추가 검토"
            retry = False
        critics.append({
            "action_id": observed["id"],
            "category": category,
            "severity": observed["severity"],
            "label": observed["label"],
            "command": observed["command"],
            "summary_label": observed["summary_label"],
            "recommendation": recommendation,
            "retry_recommended": retry,
        })
    return critics


def memory_hygiene_candidates(limit: int = 20) -> list[dict[str, Any]]:
    rows = list_memories(limit=500)
    title_counts = Counter(str(row.get("title") or "").strip().lower() for row in rows)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        tags = _decode_json(row.get("tags_json"), [])
        title = str(row.get("title") or "")
        duplicate_count = title_counts[title.strip().lower()]
        age = _age_hours(row.get("created_at"))
        use_count = int(row.get("use_count") or 0)
        importance = float(row.get("importance") or 0.0)
        reason: str | None = None
        action = "keep"
        score = 0.0
        if duplicate_count >= 2:
            reason = "같은 제목 기억이 여러 개 있어 압축 후보"
            action = "merge"
            score = min(1.0, duplicate_count / 5)
        elif use_count == 0 and age >= 24 and importance < 0.6:
            reason = "오래됐고 거의 쓰이지 않은 낮은 중요도 기억"
            action = "archive_candidate"
            score = 0.55
        elif any(str(tag) in {"feedback", "failure"} for tag in tags) and duplicate_count >= 1:
            reason = "피드백/실패 기억은 요약 기억으로 승격 후보"
            action = "summarize"
            score = 0.5
        if reason:
            candidates.append({
                "memory_id": row.get("id"),
                "title": row.get("title"),
                "memory_type": row.get("memory_type"),
                "action": action,
                "reason": reason,
                "score": round(score, 4),
            })
    return sorted(candidates, key=lambda row: row["score"], reverse=True)[:limit]


def skill_candidates(limit: int = 10) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM reflections ORDER BY id DESC LIMIT 300").fetchall()]
    summaries = Counter(str(row.get("summary") or "").strip() for row in rows if row.get("summary"))
    candidates: list[dict[str, Any]] = []
    for summary, count in summaries.items():
        if count < 3:
            continue
        if "talk feedback" in summary.lower():
            name = "core_talk_pipeline"
            trigger = "대화 응답 품질과 말투 피드백 처리"
        elif "approved local action" in summary.lower() or "task worker" in summary.lower():
            name = "task_worker_execution_review"
            trigger = "task worker action 실행 결과 검토"
        elif "workspace artifact" in summary.lower():
            name = "workspace_artifact_completion"
            trigger = "workspace 산출물 생성 후 검증"
        else:
            name = "reflection_pattern_" + str(abs(hash(summary)))[:8]
            trigger = summary[:120]
        candidates.append({
            "name": name,
            "trigger": trigger,
            "evidence_count": count,
            "suggested_procedure": [
                "반복된 reflection 패턴 확인",
                "성공/실패 조건 분리",
                "다음 유사 작업에 재사용 가능한 절차로 저장",
            ],
        })
    return sorted(candidates, key=lambda row: row["evidence_count"], reverse=True)[:limit]


def proposal_feedback(limit: int = 20) -> dict[str, Any]:
    from agent.lab.proposals import list_action_proposals

    proposals = list_action_proposals(limit)
    counts = Counter(str(row.get("status") or "unknown") for row in proposals)
    stale = [
        {"id": row.get("id"), "status": row.get("status"), "reason": row.get("reason")}
        for row in proposals
        if row.get("status") in {"proposed", "approved_by_policy", "blocked"}
    ][:5]
    return {
        "counts": dict(counts),
        "stale_or_pending": stale,
        "recommendation": "제안이 오래 남으면 승인 조건, 중복 여부, 실행 가능 프로필을 다시 평가",
    }


def next_improvement_candidates() -> list[dict[str, Any]]:
    metrics = collect_metrics()
    items: list[dict[str, Any]] = []
    if float(metrics.get("action_success_rate_24h") or 0.0) < 0.85:
        items.append({"title": "Improve action failure critic", "reason": "24시간 action 성공률이 85% 미만", "priority": 0.86})
    if int(metrics.get("memories_count") or 0) >= 200 or int(metrics.get("reflections_count") or 0) >= 300:
        items.append({"title": "Compact old memories and reflections", "reason": "기억/회고 누적량이 커짐", "priority": 0.84})
    if int(metrics.get("skills_count") or 0) <= 3:
        items.append({"title": "Promote reflection patterns into skills", "reason": "저장된 skill 수가 낮음", "priority": 0.82})
    if int(metrics.get("queued_autonomous_tasks_count") or 0) > 0:
        items.append({"title": "Explain autonomous queue priority", "reason": "대기 중인 자율 작업이 있음", "priority": 0.74})
    return sorted(items, key=lambda row: row["priority"], reverse=True)


def record_operating_review(review_type: str, summary: str, payload: dict[str, Any], score: float = 0.0, status: str = "open") -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO operating_reviews (created_at, review_type, summary, score, status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), review_type, summary[:300], float(score), status, _json(payload)),
        )
        conn.commit()
        return int(cur.lastrowid)


def operating_snapshot(*, persist: bool = False, refresh_priorities: bool = False) -> dict[str, Any]:
    priorities = (
        refresh_goal_priorities()
        if refresh_priorities
        else {"changed": 0, "items": ranked_goals(limit=20), "refreshed": False}
    )
    critics = action_critics(limit=10)
    memory = memory_hygiene_candidates(limit=10)
    skills = skill_candidates(limit=10)
    proposals = proposal_feedback(limit=20)
    improvements = next_improvement_candidates()
    snapshot = {
        "goal_priorities": priorities,
        "action_critics": critics,
        "memory_hygiene_candidates": memory,
        "skill_candidates": skills,
        "proposal_feedback": proposals,
        "next_improvement_candidates": improvements,
    }
    if persist:
        snapshot["review_id"] = record_operating_review(
            "operating_snapshot",
            "Core operating intelligence snapshot",
            snapshot,
            score=max([item.get("priority", 0.0) for item in improvements] or [0.0]),
            status="open" if improvements else "done",
        )
    return snapshot
