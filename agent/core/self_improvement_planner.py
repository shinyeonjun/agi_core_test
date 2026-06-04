from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from agent.config.defaults import now_kst
from agent.core.cognitive_engine import cognitive_growth_snapshot
from agent.core.database import init_db
from agent.core.dependency_doctor import dependency_doctor
from agent.core.goals import create_goal
from agent.core.metrics import collect_metrics
from agent.core.operating_intelligence import operating_snapshot
from agent.core.task_queue import enqueue_task


@dataclass(frozen=True)
class SelfImprovementTicket:
    key: str
    title: str
    problem: str
    evidence: dict[str, Any]
    scope: list[str]
    success_criteria: list[str]
    verification_commands: list[str]
    risk_level: str
    priority: float


def _ticket(
    key: str,
    title: str,
    problem: str,
    *,
    evidence: dict[str, Any],
    scope: list[str],
    success_criteria: list[str],
    risk_level: str = "medium",
    priority: float = 0.7,
) -> SelfImprovementTicket:
    return SelfImprovementTicket(
        key=key,
        title=title,
        problem=problem,
        evidence=evidence,
        scope=scope,
        success_criteria=success_criteria,
        verification_commands=[
            "python -m agent.cli.agentctl test run fast --json",
            "python -m agent.cli.agentctl audit",
            "python -m agent.cli.agentctl eval run",
        ],
        risk_level=risk_level,
        priority=max(0.1, min(0.95, float(priority))),
    )


def generate_self_improvement_tickets(limit: int = 5) -> list[dict[str, Any]]:
    metrics = collect_metrics()
    growth = cognitive_growth_snapshot(persist=False, limit=5)
    operating = operating_snapshot(persist=False)
    deps = dependency_doctor()
    tickets: list[SelfImprovementTicket] = []

    if not deps.get("ok"):
        tickets.append(
            _ticket(
                "python_dependency_repair",
                "Python 실행 준비 상태 보강",
                "필수 Python 패키지가 빠져 있으면 worker와 검증 루프가 불안정해진다.",
                evidence={"missing": deps.get("missing"), "install_command": deps.get("install_command")},
                scope=["agent/core/dependency_doctor.py", "agent/cli/agentctl.py", "tests/test_dependency_doctor.py"],
                success_criteria=[
                    "agentctl deps doctor가 ok=true를 반환한다.",
                    "apt/systemd 변경 없이 venv 패키지만 다룬다.",
                ],
                risk_level="medium",
                priority=0.9,
            )
        )

    fallback_rate = float(metrics.get("renderer_fallback_rate") or 0.0)
    if fallback_rate >= 0.08:
        tickets.append(
            _ticket(
                "renderer_recovery_quality",
                "대화 렌더러 복구 품질 개선",
                "대화 렌더러 fallback 비율이 높으면 Core가 멈추거나 템플릿처럼 보인다.",
                evidence={"renderer_fallback_rate": fallback_rate, "renderer_success_rate": metrics.get("renderer_success_rate")},
                scope=["agent/renderer", "agent/bridge/formatter.py", "tests/test_renderer.py", "tests/test_discord_control_plane.py"],
                success_criteria=[
                    "fallback 문구가 내부 필드나 템플릿 냄새를 노출하지 않는다.",
                    "Discord 대화 응답 테스트가 통과한다.",
                ],
                risk_level="medium",
                priority=0.84,
            )
        )

    failure_learning = growth.get("failure_learning") or {}
    dominant = str(failure_learning.get("dominant_category") or "unknown")
    if dominant in {"verification_failed", "timeout", "tool_error", "tool_unavailable", "environment_issue", "unknown"}:
        tickets.append(
            _ticket(
                "failure_recovery_loop",
                "실패 복구 루프 강화",
                "반복 실패가 생기면 Core가 같은 방식으로 재시도하지 말고 복구 전략을 바꿔야 한다.",
                evidence={"failure_learning": failure_learning, "active_inference": growth.get("active_inference")},
                scope=["agent/core/failure.py", "agent/core/cognitive_engine.py", "agent/lab/codex_worker.py", "tests/test_failure_taxonomy.py"],
                success_criteria=[
                    "실패 원인별 복구 루트가 기록된다.",
                    "worker 결과에 다음 복구 행동이 포함된다.",
                ],
                risk_level="medium",
                priority=0.82,
            )
        )

    reflections = int(metrics.get("reflections_count") or 0)
    memories = int(metrics.get("memories_count") or 0)
    if reflections >= 500 or memories >= 250:
        tickets.append(
            _ticket(
                "memory_reflection_pressure",
                "기억과 회고 압력 제어 개선",
                "기억/회고가 많이 쌓이면 검색과 판단 품질이 떨어질 수 있다.",
                evidence={"memories": memories, "reflections": reflections, "memory_candidates": operating.get("memory_hygiene_candidates", [])[:3]},
                scope=["agent/core/memory_intelligence.py", "agent/memory", "tests/test_memory_intelligence.py"],
                success_criteria=[
                    "중복/낡은 기억 정리 후보가 명확히 나온다.",
                    "정리 후 검색 인덱스가 최신화된다.",
                ],
                risk_level="low",
                priority=0.8,
            )
        )

    if float(metrics.get("action_execution_success_rate_24h") or 1.0) < 0.9:
        tickets.append(
            _ticket(
                "action_success_learning",
                "작업 성공률 학습 개선",
                "action 성공률이 떨어지면 실패 원인을 더 구체적으로 나눠야 다음 실행이 좋아진다.",
                evidence={"action_success": metrics.get("action_execution_success_rate_24h"), "critics": operating.get("action_critics", [])[:5]},
                scope=["agent/core/observability.py", "agent/core/operating_intelligence.py", "tests/test_observability.py"],
                success_criteria=[
                    "실패/차단/시간초과 원인이 분리된다.",
                    "요약에서 다음 조치가 사람 말로 나온다.",
                ],
                risk_level="medium",
                priority=0.78,
            )
        )

    if not tickets:
        tickets.append(
            _ticket(
                "self_improvement_maintenance",
                "자가개선 배포 루프 점검",
                "현재 큰 경보는 없지만 자가개선 루프는 검증성과 설명 가능성을 주기적으로 점검해야 한다.",
                evidence={"metrics": {key: metrics.get(key) for key in ("last_eval_result", "last_eval_score", "renderer_success_rate", "memory_vector_coverage")}},
                scope=["agent/core/self_improvement_release.py", "agent/core/self_improvement_planner.py", "tests/test_self_improvement_release.py"],
                success_criteria=[
                    "release gate가 변경 위험을 잘 분류한다.",
                    "자가개선 작업은 main 직접 변경 없이 worktree로 이동한다.",
                ],
                risk_level="low",
                priority=0.62,
            )
        )

    deduped: dict[str, SelfImprovementTicket] = {}
    for ticket in tickets:
        current = deduped.get(ticket.key)
        if current is None or ticket.priority > current.priority:
            deduped[ticket.key] = ticket
    ranked = sorted(deduped.values(), key=lambda item: item.priority, reverse=True)
    return [asdict(ticket) for ticket in ranked[: max(1, int(limit))]]


def ticket_to_worker_prompt(ticket: dict[str, Any]) -> str:
    return "\n".join(
        [
            "Agent Core self-improvement code task.",
            "",
            "Goal:",
            str(ticket.get("title") or ""),
            "",
            "Problem:",
            str(ticket.get("problem") or ""),
            "",
            "Evidence:",
            json.dumps(ticket.get("evidence") or {}, ensure_ascii=False, indent=2)[:2500],
            "",
            "Allowed scope:",
            *[f"- {item}" for item in ticket.get("scope") or []],
            "",
            "Success criteria:",
            *[f"- {item}" for item in ticket.get("success_criteria") or []],
            "",
            "Verification commands:",
            *[f"- {item}" for item in ticket.get("verification_commands") or []],
            "",
            "Hard boundaries:",
            "- Use git worktree/native loop only; do not apply directly to main.",
            "- Do not read or modify .env, tokens, credentials, SSH keys, systemd, apt, or OS settings.",
            "- Keep the change small, tested, and explain remaining risk.",
        ]
    )


def enqueue_self_improvement_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    init_db()
    prompt = ticket_to_worker_prompt(ticket)
    goal_id = create_goal(
        str(ticket.get("title") or "Self-improvement code task"),
        prompt,
        goal_type="self_improvement_proposal",
        status="active",
        priority=float(ticket.get("priority") or 0.7),
        risk_level=str(ticket.get("risk_level") or "medium"),
        metadata={
            "priority_owner": "core",
            "task_kind": "code_change",
            "self_improvement_ticket": ticket,
            "raw_user_text": prompt,
            "created_at": now_kst(),
            "requires_native_loop": True,
        },
        dedupe=True,
    )
    task_id = enqueue_task(
        "autonomous",
        goal_id=goal_id,
        task_kind="code_change",
        title=str(ticket.get("title") or "Self-improvement code task"),
        source="self_improvement_planner",
        priority=float(ticket.get("priority") or 0.7),
        payload={"ticket": ticket, "worker_prompt": prompt, "requires_native_loop": True},
        idempotency_key=f"self_improvement:{ticket.get('key')}",
    )
    return {"goal_id": goal_id, "task_id": task_id, "ticket": ticket}


def enqueue_user_self_improvement_request(text: str, *, source_event_id: int | None = None, limit: int = 1) -> dict[str, Any]:
    init_db()
    tickets = generate_self_improvement_tickets(limit=limit)
    created: list[dict[str, Any]] = []
    for ticket in tickets[: max(1, int(limit))]:
        prompt = ticket_to_worker_prompt(ticket)
        title = f"사용자 요청 자가개선: {ticket.get('title') or 'Core self-improvement'}"
        goal_id = create_goal(
            title,
            prompt,
            goal_type="self_improvement_proposal",
            status="active",
            priority=max(0.9, float(ticket.get("priority") or 0.7)),
            risk_level=str(ticket.get("risk_level") or "medium"),
            metadata={
                "source": "user_self_improvement_request",
                "source_event_id": source_event_id,
                "priority_owner": "user",
                "task_kind": "code_change",
                "self_improvement_ticket": ticket,
                "raw_user_text": prompt,
                "user_request": text,
                "created_at": now_kst(),
                "requires_native_loop": True,
                "state_machine": {
                    "phase": "queued",
                    "steps": ["rank_ticket", "worktree_execute", "verify", "review", "approval", "apply_after_approval"],
                    "main_apply": "approval_required",
                },
            },
            dedupe=True,
        )
        task_id = enqueue_task(
            "user",
            goal_id=goal_id,
            task_kind="code_change",
            title=title,
            source="discord_self_improvement",
            priority=max(0.9, float(ticket.get("priority") or 0.7)),
            payload={"ticket": ticket, "worker_prompt": prompt, "requires_native_loop": True, "source_event_id": source_event_id},
            idempotency_key=f"user_self_improvement:{ticket.get('key')}",
        )
        created.append({"goal_id": goal_id, "task_id": task_id, "ticket": ticket})
    return {"created": created, "count": len(created), "source": "user_self_improvement_request"}


def enqueue_self_improvement_tickets(limit: int = 1) -> dict[str, Any]:
    tickets = generate_self_improvement_tickets(limit=limit)
    enqueued = [enqueue_self_improvement_ticket(ticket) for ticket in tickets[: max(1, int(limit))]]
    return {"created": enqueued, "count": len(enqueued)}
