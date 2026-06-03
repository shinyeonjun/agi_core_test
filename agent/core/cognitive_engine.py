from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.failure import failure_report, recovery_hint
from agent.core.metrics import collect_metrics
from agent.core.operating_intelligence import action_critics, ranked_goals
from agent.memory.store import search_memories


RISK_WEIGHT = {"low": 0.0, "medium": 0.12, "high": 0.28, "critical": 0.45}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _decode_json(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def bayesian_update(successes: int, failures: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> dict[str, Any]:
    alpha = max(0.0, prior_alpha) + max(0, int(successes))
    beta = max(0.0, prior_beta) + max(0, int(failures))
    total = alpha + beta
    mean = alpha / total if total else 0.5
    confidence = min(1.0, (max(0, successes) + max(0, failures)) / 12)
    return {
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "expected_success": round(mean, 4),
        "confidence": round(confidence, 4),
        "evidence": {"successes": max(0, int(successes)), "failures": max(0, int(failures))},
    }


def failure_strategy(category: str) -> dict[str, Any]:
    category = category if category in {
        "success",
        "intent_misread",
        "bad_plan",
        "tool_error",
        "tool_unavailable",
        "permission_block",
        "policy_block",
        "profile_block",
        "missing_context",
        "verification_failed",
        "timeout",
        "environment_issue",
        "input_insufficient",
        "unknown",
    } else "unknown"
    routes = {
        "success": {"route": "reuse", "retry": False, "next_phase": "learn"},
        "verification_failed": {"route": "repair", "retry": True, "next_phase": "verify"},
        "timeout": {"route": "split", "retry": True, "next_phase": "decompose"},
        "tool_unavailable": {"route": "dependency_doctor", "retry": False, "next_phase": "observe"},
        "environment_issue": {"route": "dependency_doctor", "retry": False, "next_phase": "observe"},
        "permission_block": {"route": "approval", "retry": False, "next_phase": "route"},
        "policy_block": {"route": "safe_alternative", "retry": False, "next_phase": "define_success"},
        "profile_block": {"route": "profile_check", "retry": False, "next_phase": "route"},
        "missing_context": {"route": "collect_context", "retry": True, "next_phase": "observe"},
        "input_insufficient": {"route": "ask_or_scope", "retry": False, "next_phase": "define_success"},
        "bad_plan": {"route": "replan_smaller", "retry": True, "next_phase": "decompose"},
        "tool_error": {"route": "minimal_repro", "retry": True, "next_phase": "verify"},
        "intent_misread": {"route": "reinterpret", "retry": True, "next_phase": "observe"},
        "unknown": {"route": "classify_before_retry", "retry": False, "next_phase": "observe"},
    }
    return {
        "category": category,
        "route": routes[category]["route"],
        "retry_recommended": routes[category]["retry"],
        "next_phase": routes[category]["next_phase"],
        "recovery_hint": recovery_hint(category),
    }


def outcome_patterns(limit: int = 40) -> dict[str, Any]:
    critics = action_critics(limit=limit)
    counts = Counter(str(row.get("category") or "unknown") for row in critics)
    total = sum(counts.values())
    strategies = {category: failure_strategy(category) for category in sorted(counts)}
    dominant = counts.most_common(1)[0][0] if counts else "unknown"
    return {
        "total": total,
        "counts": dict(counts),
        "dominant_category": dominant,
        "dominant_strategy": failure_strategy(dominant),
        "strategies": strategies,
        "success_ratio": round((counts.get("success", 0) / total), 4) if total else 0.0,
    }


def curiosity_signals(metrics: dict[str, Any] | None = None, limit: int = 8) -> list[dict[str, Any]]:
    metrics = metrics or collect_metrics()
    signals = [
        {
            "topic": "memory_retrieval",
            "question": "기억 검색 품질을 더 믿을 수 있게 만들려면 무엇을 확인해야 하나?",
            "pressure": 1.0 - float(metrics.get("memory_vector_coverage") or 0.0),
            "evidence": {"memory_vector_coverage": metrics.get("memory_vector_coverage")},
        },
        {
            "topic": "action_reliability",
            "question": "최근 실행 실패는 환경 문제인지, 정책 차단인지, 명령 설계 문제인지 구분됐나?",
            "pressure": 1.0 - float(metrics.get("action_execution_success_rate_24h") or 0.0),
            "evidence": {
                "success_rate": metrics.get("action_execution_success_rate_24h"),
                "timeout": metrics.get("action_timeout_count_24h"),
                "blocked": metrics.get("action_blocked_count_24h"),
            },
        },
        {
            "topic": "memory_hygiene",
            "question": "쌓인 기억과 회고가 다음 판단에 실제로 쓰이는 형태로 압축되고 있나?",
            "pressure": min(1.0, (int(metrics.get("memories_count") or 0) / 600) + (int(metrics.get("reflections_count") or 0) / 900)),
            "evidence": {"memories": metrics.get("memories_count"), "reflections": metrics.get("reflections_count")},
        },
        {
            "topic": "goal_queue",
            "question": "사용자 작업과 자율 작업의 우선순위가 서로 밀어내지 않고 분리되어 있나?",
            "pressure": min(1.0, (int(metrics.get("queued_user_tasks_count") or 0) * 0.35) + (int(metrics.get("queued_autonomous_tasks_count") or 0) * 0.2)),
            "evidence": {"user_queue": metrics.get("queued_user_tasks_count"), "autonomous_queue": metrics.get("queued_autonomous_tasks_count")},
        },
        {
            "topic": "runtime_body",
            "question": "자기 몸 상태(self-map)가 최신이고 반복 관찰의 변화가 기록되고 있나?",
            "pressure": 0.35 if int(metrics.get("self_map_count") or 0) == 0 else 0.08,
            "evidence": {"self_map_count": metrics.get("self_map_count")},
        },
        {
            "topic": "renderer_quality",
            "question": "대화 답변은 내부 상태를 숨기지 않되 사람이 바로 이해할 만큼 자연스러운가?",
            "pressure": float(metrics.get("renderer_fallback_rate") or 0.0),
            "evidence": {"fallback_rate": metrics.get("renderer_fallback_rate"), "success_rate": metrics.get("renderer_success_rate")},
        },
    ]
    for item in signals:
        item["pressure"] = round(_clamp(float(item["pressure"])), 4)
    return sorted(signals, key=lambda row: row["pressure"], reverse=True)[:limit]


def utility_novelty_score(candidate: dict[str, Any], metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or collect_metrics()
    title = str(candidate.get("title") or "")
    risk = str(candidate.get("risk_level") or "low")
    base = float(candidate.get("priority_score") or candidate.get("score") or candidate.get("priority") or 0.5)
    novelty = float(candidate.get("novelty_score") or 0.0)
    utility = float(candidate.get("utility_score") or base)
    if not novelty:
        similar = search_memories(title, limit=3) if title else []
        novelty = 1.0 - max([float(row.get("score") or 0.0) for row in similar] or [0.0])
    queue_pressure = min(0.16, (int(metrics.get("queued_user_tasks_count") or 0) * 0.04) + (int(metrics.get("queued_autonomous_tasks_count") or 0) * 0.02))
    risk_penalty = RISK_WEIGHT.get(risk, 0.12)
    score = (base * 0.42) + (utility * 0.28) + (novelty * 0.20) + queue_pressure - risk_penalty
    return {
        "id": candidate.get("id"),
        "title": title,
        "score": round(_clamp(score), 4),
        "components": {
            "base": round(_clamp(base), 4),
            "utility": round(_clamp(utility), 4),
            "novelty": round(_clamp(novelty), 4),
            "queue_pressure": round(queue_pressure, 4),
            "risk_penalty": round(risk_penalty, 4),
        },
    }


def htn_plan_for_goal(goal: dict[str, Any]) -> dict[str, Any]:
    goal_type = str(goal.get("goal_type") or "general")
    metadata = _decode_json(goal.get("metadata_json"), {})
    owner = metadata.get("priority_owner") or ("user" if goal_type == "user_directed" else "core")
    objective = str(goal.get("description") or goal.get("title") or "")
    failure_category = str(metadata.get("failure_category") or metadata.get("last_failure_category") or "")
    fallback = failure_strategy(failure_category) if failure_category else failure_strategy("unknown")
    return {
        "goal_id": goal.get("id"),
        "owner": owner,
        "goal_type": goal_type,
        "objective": objective,
        "method": "htn_safe_execution",
        "steps": [
            {"phase": "observe", "task": "현재 상태와 관련 기억을 모은다", "output": "context_bundle"},
            {"phase": "define_success", "task": "완료 기준과 금지 경계를 분리한다", "output": "success_criteria"},
            {"phase": "decompose", "task": "작업을 읽기/계획/실행/검증 단계로 나눈다", "output": "step_plan"},
            {"phase": "route", "task": "대화, 자율 루프, Codex 작업자, 승인 큐 중 어디로 보낼지 정한다", "output": "route_decision"},
            {"phase": "verify", "task": "테스트, 감사, 관찰 결과로 성공 여부를 확인한다", "output": "evidence"},
            {"phase": "learn", "task": "결과를 기억, 회고, 스킬 후보, 흔적 신호로 남긴다", "output": "memory_update"},
        ],
        "fallback_policy": {
            "on_failure_category": failure_category or "unknown",
            "route": fallback["route"],
            "retry_recommended": fallback["retry_recommended"],
            "return_to_phase": fallback["next_phase"],
            "hint": fallback["recovery_hint"],
        },
    }


def case_based_reasoning(query: str, limit: int = 5) -> dict[str, Any]:
    cases = search_memories(query, limit=limit) if query.strip() else []
    typed = []
    for row in cases:
        report = failure_report({"title": row.get("title"), "type": row.get("memory_type"), "content": row.get("content")})
        typed.append({
            "memory_id": row.get("id"),
            "title": row.get("title"),
            "memory_type": row.get("memory_type"),
            "score": row.get("score"),
            "failure_category": report["category"],
            "strategy": failure_strategy(report["category"]),
            "reuse_hint": "비슷한 상황의 판단 재료와 복구 전략으로 사용",
        })
    return {
        "query": query,
        "cases": typed,
        "top_strategy": typed[0]["strategy"] if typed else failure_strategy("unknown"),
    }


def build_map_elites(items: list[dict[str, Any]], archive_name: str = "goal_growth") -> list[dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for item in items:
        score = float(item.get("score") or item.get("priority_score") or 0.0)
        risk = str(item.get("risk_level") or "low")
        novelty = float(item.get("components", {}).get("novelty") or item.get("novelty_score") or 0.0)
        utility = float(item.get("components", {}).get("utility") or item.get("utility_score") or score)
        axes = {
            "risk": risk,
            "novelty": "high" if novelty >= 0.66 else "medium" if novelty >= 0.33 else "low",
            "utility": "high" if utility >= 0.66 else "medium" if utility >= 0.33 else "low",
        }
        cell_key = "|".join(f"{key}:{value}" for key, value in sorted(axes.items()))
        candidate = {
            "archive_name": archive_name,
            "cell_key": cell_key,
            "axes": axes,
            "score": round(_clamp(score), 4),
            "candidate": item,
        }
        if cell_key not in cells or candidate["score"] > cells[cell_key]["score"]:
            cells[cell_key] = candidate
    return sorted(cells.values(), key=lambda row: row["score"], reverse=True)


def persist_map_elites(items: list[dict[str, Any]]) -> int:
    if not items:
        return 0
    init_db()
    ts = now_kst()
    with connect() as conn:
        for item in items:
            conn.execute(
                """
                INSERT INTO cognitive_map_elites (created_at, updated_at, archive_name, cell_key, axes_json, candidate_json, score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(archive_name, cell_key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    axes_json = excluded.axes_json,
                    candidate_json = excluded.candidate_json,
                    score = excluded.score,
                    status = 'active'
                WHERE excluded.score >= cognitive_map_elites.score
                """,
                (ts, ts, item["archive_name"], item["cell_key"], _json(item["axes"]), _json(item["candidate"]), item["score"]),
            )
        conn.commit()
    return len(items)


def add_blackboard_item(source: str, topic: str, content: str, *, confidence: float = 0.5, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, status: str = "open") -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO blackboard_items (created_at, updated_at, source, topic, content, status, confidence, tags_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, ts, source, topic, content, status, _clamp(float(confidence)), _json(tags or []), _json(metadata or {})),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_blackboard_items(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM blackboard_items"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY confidence DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def active_inference_lite(metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = metrics or collect_metrics()
    patterns = outcome_patterns(limit=30)
    uncertainty = max(
        1.0 - float(metrics.get("renderer_success_rate") or 0.0),
        1.0 - float(metrics.get("memory_vector_coverage") or 0.0),
    )
    instability = 1.0 - float(metrics.get("action_execution_success_rate_24h") or 0.0)
    user_pressure = min(1.0, int(metrics.get("queued_user_tasks_count") or 0) / 3)
    exploration_pressure = max(item["pressure"] for item in curiosity_signals(metrics, limit=3))
    free_energy = (uncertainty * 0.30) + (instability * 0.25) + (user_pressure * 0.25) + (exploration_pressure * 0.20)
    if user_pressure >= 0.34:
        mode = "serve_user"
    elif instability >= 0.35:
        mode = "stabilize"
    elif exploration_pressure >= 0.50:
        mode = "explore"
    else:
        mode = "consolidate"
    if patterns["dominant_category"] in {"verification_failed", "timeout", "tool_error"} and patterns["success_ratio"] < 0.75:
        mode = "stabilize"
    elif patterns["dominant_category"] in {"tool_unavailable", "environment_issue"}:
        mode = "repair_environment"
    return {
        "mode": mode,
        "free_energy": round(_clamp(free_energy), 4),
        "pressures": {
            "uncertainty": round(_clamp(uncertainty), 4),
            "instability": round(_clamp(instability), 4),
            "user": round(_clamp(user_pressure), 4),
            "exploration": round(_clamp(exploration_pressure), 4),
        },
        "outcome_learning": patterns,
        "recommended_strategy": patterns["dominant_strategy"],
    }


def add_stigmergy_marker(marker_type: str, target_type: str, target_id: str | int | None, reason: str, *, intensity: float = 0.5, decay_rate: float = 0.05, metadata: dict[str, Any] | None = None, status: str = "active") -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO stigmergy_markers (created_at, updated_at, marker_type, target_type, target_id, intensity, decay_rate, status, reason, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, ts, marker_type, target_type, str(target_id) if target_id is not None else None, _clamp(float(intensity)), _clamp(float(decay_rate)), status, reason, _json(metadata or {})),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_stigmergy_markers(limit: int = 20, status: str | None = "active") -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM stigmergy_markers"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY intensity DESC, id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def _recent_outcome_bayes() -> dict[str, Any]:
    metrics = collect_metrics()
    successes = int(metrics.get("action_successful_count_24h") or 0)
    failures = int(metrics.get("action_executed_count_24h") or 0) - successes
    return bayesian_update(successes, max(0, failures), prior_alpha=2.0, prior_beta=1.0)


def _blackboard_seed(snapshot: dict[str, Any]) -> dict[str, Any]:
    active = list_blackboard_items(limit=8, status="open")
    if active:
        return {"items": active, "created": 0}
    top_signal = snapshot["curiosity"][0] if snapshot.get("curiosity") else None
    if not top_signal:
        return {"items": [], "created": 0}
    item_id = add_blackboard_item(
        "cognitive_engine",
        str(top_signal["topic"]),
        str(top_signal["question"]),
        confidence=float(top_signal["pressure"]),
        tags=["curiosity", "growth"],
        metadata={"evidence": top_signal.get("evidence", {})},
    )
    return {"items": list_blackboard_items(limit=8, status="open"), "created": item_id}


def _stigmergy_seed(snapshot: dict[str, Any]) -> dict[str, Any]:
    existing = list_stigmergy_markers(limit=8, status="active")
    if existing:
        return {"items": existing, "created": 0}
    inference = snapshot.get("active_inference", {})
    marker_id = add_stigmergy_marker(
        "attention",
        "mode",
        inference.get("mode"),
        "현재 성장 루프가 다음에 집중해야 할 운영 방향",
        intensity=float(inference.get("free_energy") or 0.5),
        metadata={"pressures": inference.get("pressures", {})},
    )
    return {"items": list_stigmergy_markers(limit=8, status="active"), "created": marker_id}


def record_cognitive_snapshot(snapshot: dict[str, Any]) -> int:
    init_db()
    summary = f"mode={snapshot.get('active_inference', {}).get('mode')} curiosity={len(snapshot.get('curiosity', []))} elites={len(snapshot.get('map_elites', []))}"
    score = float(snapshot.get("active_inference", {}).get("free_energy") or 0.0)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO cognitive_snapshots (created_at, snapshot_type, summary, score, payload_json)
            VALUES (?, 'growth_algorithms', ?, ?, ?)
            """,
            (now_kst(), summary, score, _json(snapshot)),
        )
        conn.commit()
        return int(cur.lastrowid)


def cognitive_growth_snapshot(*, persist: bool = False, limit: int = 8) -> dict[str, Any]:
    metrics = collect_metrics()
    goals = ranked_goals(limit=limit)
    scored = [utility_novelty_score(goal, metrics) | {"risk_level": goal.get("risk_level"), "goal_type": goal.get("goal_type")} for goal in goals]
    htn_source = goals[0] if goals else {"title": "Core growth maintenance", "goal_type": "self_improvement_proposal", "description": "Core 상태를 관찰하고 개선 루프를 유지한다"}
    query = str(htn_source.get("title") or htn_source.get("description") or "Core growth")
    critics = action_critics(limit=20)
    critic_counts = Counter(str(row.get("category") or "unknown") for row in critics)
    snapshot: dict[str, Any] = {
        "created_at": now_kst(),
        "algorithms": [
            "curiosity_engine",
            "utility_novelty_scoring",
            "htn_planning",
            "case_based_reasoning",
            "bayesian_update",
            "map_elites_archive",
            "blackboard_architecture",
            "active_inference_lite",
            "stigmergy_markers",
        ],
        "curiosity": curiosity_signals(metrics, limit=limit),
        "utility_scoring": sorted(scored, key=lambda row: row["score"], reverse=True),
        "htn_plan": htn_plan_for_goal(htn_source),
        "case_based": case_based_reasoning(query, limit=min(5, limit)),
        "bayesian_update": {
            "action_execution": _recent_outcome_bayes(),
            "critic_counts": dict(critic_counts),
        },
        "failure_learning": outcome_patterns(limit=30),
        "map_elites": build_map_elites(scored),
        "active_inference": active_inference_lite(metrics),
        "metrics": {
            "memories": metrics.get("memories_count"),
            "reflections": metrics.get("reflections_count"),
            "action_success": metrics.get("action_execution_success_rate_24h"),
            "renderer_success": metrics.get("renderer_success_rate"),
            "vector_coverage": metrics.get("memory_vector_coverage"),
        },
    }
    snapshot["blackboard"] = _blackboard_seed(snapshot) if persist else {"items": list_blackboard_items(limit=limit, status="open"), "created": 0}
    snapshot["stigmergy"] = _stigmergy_seed(snapshot) if persist else {"items": list_stigmergy_markers(limit=limit, status="active"), "created": 0}
    if persist:
        snapshot["map_elites_persisted"] = persist_map_elites(snapshot["map_elites"])
        snapshot["snapshot_id"] = record_cognitive_snapshot(snapshot)
    return snapshot
