from __future__ import annotations

from typing import Any

from agent.core.policy import PolicyDecision


def _intent(interpretation: dict[str, Any] | None) -> str:
    return str((interpretation or {}).get("intent") or "unknown")


def _target(interpretation: dict[str, Any] | None) -> str:
    return str((interpretation or {}).get("target") or "")


def memory_route(query: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strategy": "fts_sparse_hybrid",
        "query": query[:240],
        "selected_ids": [row.get("id") for row in memories],
        "count": len(memories),
        "reason": "사용자 입력과 관련된 장기 기억을 FTS와 sparse vector 후보에서 선택한다.",
    }


def skill_route(query: str, skills: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "strategy": "trigger_similarity",
        "query": query[:240],
        "selected_ids": [row.get("id") for row in skills],
        "selected_names": [row.get("name") for row in skills],
        "count": len(skills),
        "reason": "요청 맥락과 맞는 절차 기억을 선택한다.",
    }


def tool_routes(interpretation: dict[str, Any], policy: PolicyDecision, capability_map: dict[str, Any]) -> list[dict[str, Any]]:
    intent = _intent(interpretation)
    target = _target(interpretation)
    direct = {item.get("name"): item for item in capability_map.get("direct", [])}
    workers = {item.get("name"): item for item in capability_map.get("worker_mediated", [])}
    routes: list[dict[str, Any]] = [
        {
            "name": "policy_engine",
            "purpose": "요청 위험도와 승인 필요 여부 판단",
            "status": (direct.get("policy_engine") or {}).get("status", "unknown"),
            "risk": "low",
            "score": 0.95,
        },
        {
            "name": "memory_search",
            "purpose": "관련 장기 기억 검색",
            "status": (direct.get("memory_search") or {}).get("status", "unknown"),
            "risk": "low",
            "score": 0.82,
        },
    ]
    if intent in {"task_request", "project_request", "report_request"}:
        routes.append(
            {
                "name": "goal_task_queue",
                "purpose": "사용자 지시를 user queue로 분리 등록",
                "status": (direct.get("goal_task_queue") or {}).get("status", "unknown"),
                "risk": policy.risk_level,
                "score": 0.86,
            }
        )
    if intent in {"task_request", "project_request"} or target in {"code_change", "project_spec"}:
        worker = workers.get("codex_work_worker") or {}
        routes.append(
            {
                "name": "codex_work_worker",
                "purpose": "코드와 프로젝트 작업 위임",
                "status": worker.get("status", "unknown"),
                "risk": policy.risk_level,
                "score": 0.75 if worker.get("status") == "enabled" else 0.35,
                "backend": worker.get("backend", "codex"),
                "blockers": worker.get("blockers", []),
            }
        )
    if policy.requires_approval or policy.denied:
        routes.append(
            {
                "name": "approval_flow",
                "purpose": "승인 또는 정책 차단 경로",
                "status": "required" if policy.requires_approval else "blocked",
                "risk": policy.risk_level,
                "score": 1.0,
                "reason": policy.reason,
            }
        )
    return sorted(routes, key=lambda row: float(row.get("score") or 0.0), reverse=True)
