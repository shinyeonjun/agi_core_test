from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agent.config.defaults import now_kst
from agent.core.capabilities import collect_capability_map
from agent.core.database import connect, init_db
from agent.core.metrics import collect_metrics


CAPABILITY_EVIDENCE_REGISTRY: dict[str, dict[str, Any]] = {
    "discord_chat": {
        "files": ["agent/bridge/router.py", "agent/bridge/discord_bot.py", "agent/core/pipeline.py"],
        "tests": ["tests/test_discord_control_plane.py", "tests/test_bridge.py"],
        "summary": "Discord 입력을 Core 대화 파이프라인으로 라우팅한다.",
    },
    "memory_search": {
        "files": ["agent/memory/store.py", "agent/memory/sparse_vector.py", "agent/memory/retrieval.py"],
        "tests": ["tests/test_retrieval_efficiency.py", "tests/test_sparse_vectors.py"],
        "summary": "FTS, sparse vector, chunking으로 기억을 찾고 재사용한다.",
    },
    "goal_task_queue": {
        "files": ["agent/core/goals.py", "agent/core/task_queue.py", "agent/core/user_goals.py"],
        "tests": ["tests/test_task_queue_split.py", "tests/test_goal_generator.py"],
        "summary": "사용자 작업과 자율 작업을 분리해서 관리한다.",
    },
    "researcher_loop": {
        "files": ["agent/core/research_loop.py", "agent/core/research_ingestion.py"],
        "tests": ["tests/test_research_loop.py", "tests/test_research_ingestion.py"],
        "summary": "질문, 가설, 작은 실험, 근거, 개선 제안을 연결한다.",
    },
    "cognitive_growth_algorithms": {
        "files": ["agent/core/cognitive_engine.py", "agent/core/cognitive_graph.py", "agent/core/advanced_learning.py"],
        "tests": ["tests/test_cognitive_engine.py", "tests/test_cognitive_graph.py", "tests/test_advanced_learning.py"],
        "summary": "호기심, 그래프, 실패 사례, 회로 차단, 메모리 계층을 운영 학습에 쓴다.",
    },
    "self_improvement_code_planner": {
        "files": ["agent/core/self_improvement_planner.py", "agent/core/self_improvement_release.py", "agent/lab/codex_worker.py"],
        "tests": ["tests/test_self_improvement_planner.py", "tests/test_self_improvement_release.py", "tests/test_codex_worker_policy.py"],
        "summary": "자기개선 후보를 만들고 격리된 코드 작업과 release gate로 넘긴다.",
    },
    "codex_work_worker": {
        "files": ["agent/lab/codex_worker.py", "agent/lab/planner.py", "agent/bridge/task_notifications.py"],
        "tests": ["tests/test_capability_codex_worker.py", "tests/test_codex_worker_policy.py", "tests/test_task_queue_split.py"],
        "summary": "사용자/자율 코드 변경 작업을 worktree와 검증 루프로 처리한다.",
    },
    "self_map": {
        "files": ["agent/core/self_map.py"],
        "tests": ["tests/test_self_map.py"],
        "summary": "시크릿 값을 읽지 않고 실행 환경과 상주 루프 상태를 요약한다.",
    },
    "event_reactor": {
        "files": ["agent/core/reactor.py", "agent/core/wake_signals.py"],
        "tests": ["tests/test_reactor.py"],
        "summary": "고정 주기만이 아니라 wake signal 기반으로 필요한 일을 처리한다.",
    },
    "policy_engine": {
        "files": ["agent/core/policy.py", "agent/core/approvals.py"],
        "tests": ["tests/test_policy.py", "tests/test_discord_control_plane.py"],
        "summary": "위험한 명령, 시크릿, 시스템 변경을 승인/차단 흐름으로 보낸다.",
    },
    "codex_chat_renderer": {
        "files": ["agent/renderer/codex_renderer.py", "agent/renderer/prompts.py", "agent/renderer/validator.py"],
        "tests": ["tests/test_renderer.py", "tests/test_fallback_rule_clean_korean.py"],
        "summary": "Codex를 이용해 내부 상태를 사용자용 한국어 답변으로 렌더링한다.",
    },
}

SELF_REPORT_KEYWORDS = {
    "change": ("개선", "나아", "바뀐", "변경", "업데이트", "커밋", "많이"),
    "capability": ("뭐 할", "가능", "능력", "할 수", "기능"),
    "status": ("상태", "돌고", "살아", "지금"),
    "failure": ("왜 못", "차단", "실패", "막힘", "안됨", "안 돼"),
    "research": ("연구", "논문", "가설", "실험", "학습"),
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(["git", *args], cwd=_repo_root(), text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=10, check=False)
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _split_files(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _infer_feature_name(subject: str, files: Iterable[str]) -> str:
    joined = " ".join(files).lower()
    low = subject.lower()
    if "research_loop" in joined or "research" in low:
        return "자율 연구 루프"
    if "self_report" in joined or "capabil" in low:
        return "자기 능력 근거화"
    if "renderer" in joined:
        return "대화 렌더러 품질"
    if "test_profiles" in joined or "pytest" in low:
        return "테스트 계층화"
    if "memory" in joined:
        return "기억/검색 개선"
    if "cognitive" in joined or "advanced_learning" in joined:
        return "인지 그래프/학습 루프"
    if "codex_worker" in joined or "task_queue" in joined:
        return "코드 작업 루프"
    return subject or "Core 변경"


def _capability_deltas_for_files(files: Iterable[str]) -> list[str]:
    deltas: list[str] = []
    file_set = set(files)
    for key, evidence in CAPABILITY_EVIDENCE_REGISTRY.items():
        if any(path in file_set for path in evidence.get("files", [])):
            deltas.append(key)
    return sorted(set(deltas))


def capture_git_change(commit: str = "HEAD", *, verification: dict[str, Any] | None = None, source: str = "git") -> dict[str, Any]:
    init_db()
    commit_hash = _run_git(["rev-parse", "--short", commit])
    full_hash = _run_git(["rev-parse", commit]) or commit_hash
    if not commit_hash:
        return {"captured": False, "reason": "git_unavailable"}
    subject = _run_git(["show", "-s", "--format=%s", commit])
    created_at = _run_git(["show", "-s", "--format=%cI", commit]) or now_kst()
    files = _split_files(_run_git(["show", "--name-only", "--format=", commit]))
    feature_name = _infer_feature_name(subject, files)
    capability_delta = _capability_deltas_for_files(files)
    summary = f"{feature_name}: {subject}" if subject else feature_name
    verification_payload = verification or {"source": "git", "tests": "not_recorded"}
    if verification is None:
        with connect() as conn:
            existing = conn.execute("SELECT verification_json FROM core_change_log WHERE commit_hash=?", (full_hash,)).fetchone()
        existing_verification = _loads(existing["verification_json"], {}) if existing else {}
        if existing_verification and existing_verification.get("tests") != "not_recorded":
            verification_payload = existing_verification
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO core_change_log (created_at, commit_hash, short_hash, source, feature_name, human_summary, changed_files_json, capability_delta_json, verification_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'recorded')
            ON CONFLICT(commit_hash) DO UPDATE SET
                created_at=excluded.created_at,
                short_hash=excluded.short_hash,
                source=excluded.source,
                feature_name=excluded.feature_name,
                human_summary=excluded.human_summary,
                changed_files_json=excluded.changed_files_json,
                capability_delta_json=excluded.capability_delta_json,
                verification_json=excluded.verification_json,
                status='recorded'
            """,
            (created_at, full_hash, commit_hash, source, feature_name, summary[:500], _json(files[:80]), _json(capability_delta), _json(verification_payload)),
        )
        row = conn.execute("SELECT id FROM core_change_log WHERE commit_hash=?", (full_hash,)).fetchone()
        conn.commit()
    return {
        "captured": True,
        "id": int(row["id"]) if row else None,
        "commit_hash": full_hash,
        "short_hash": commit_hash,
        "feature_name": feature_name,
        "summary": summary,
        "changed_files": files,
        "capability_delta": capability_delta,
        "verification": verification_payload,
    }


def sync_recent_changes(limit: int = 5) -> dict[str, Any]:
    init_db()
    commits = _split_files(_run_git(["log", f"-{max(1, int(limit))}", "--format=%H"]))
    captured = [capture_git_change(commit) for commit in commits]
    return {"captured": [item for item in captured if item.get("captured")], "count": sum(1 for item in captured if item.get("captured"))}


def list_change_log(limit: int = 5) -> list[dict[str, Any]]:
    init_db()
    sync_recent_changes(limit=limit)
    with connect() as conn:
        rows = conn.execute("SELECT * FROM core_change_log ORDER BY created_at DESC, id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["changed_files"] = _loads(item.pop("changed_files_json", None), [])
        item["capability_delta"] = _loads(item.pop("capability_delta_json", None), [])
        item["verification"] = _loads(item.pop("verification_json", None), {})
        result.append(item)
    return result


def _path_exists(path: str) -> bool:
    return (_repo_root() / path).exists()


def _evidence_for_capability(name: str, status: str, description: str) -> dict[str, Any]:
    registry = CAPABILITY_EVIDENCE_REGISTRY.get(name, {})
    files = [path for path in registry.get("files", []) if _path_exists(path)]
    tests = [path for path in registry.get("tests", []) if _path_exists(path)]
    has_code = bool(files)
    has_tests = bool(tests)
    confidence = 0.45 + (0.25 if status == "enabled" else 0.0) + (0.15 if has_code else 0.0) + (0.15 if has_tests else 0.0)
    confidence = max(0.0, min(0.98, confidence))
    return {
        "capability_key": name,
        "name": name,
        "status": status,
        "summary": registry.get("summary") or description,
        "description": description,
        "evidence_files": files,
        "verification_tests": tests,
        "confidence": round(confidence, 3),
        "last_verified_at": now_kst(),
        "limits": [],
    }


def refresh_capability_evidence(capability_map: dict[str, Any] | None = None) -> dict[str, Any]:
    init_db()
    data = capability_map or collect_capability_map()
    items: list[dict[str, Any]] = []
    for group in ("direct", "worker_mediated"):
        for entry in data.get(group, []) or []:
            name = str(entry.get("name") or "")
            if not name:
                continue
            evidence = _evidence_for_capability(name, str(entry.get("status") or "unknown"), str(entry.get("description") or ""))
            evidence["group"] = group
            evidence["blockers"] = entry.get("blockers") or []
            items.append(evidence)
    if not any(item["capability_key"] == "researcher_loop" for item in items):
        items.append(_evidence_for_capability("researcher_loop", "enabled", "Research question, hypothesis, experiment, evidence, and proposal loop"))
    with connect() as conn:
        for item in items:
            conn.execute(
                """
                INSERT INTO capability_evidence (created_at, updated_at, capability_key, name, status, summary, evidence_files_json, verification_tests_json, confidence, last_verified_at, limits_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(capability_key) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    name=excluded.name,
                    status=excluded.status,
                    summary=excluded.summary,
                    evidence_files_json=excluded.evidence_files_json,
                    verification_tests_json=excluded.verification_tests_json,
                    confidence=excluded.confidence,
                    last_verified_at=excluded.last_verified_at,
                    limits_json=excluded.limits_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    now_kst(),
                    now_kst(),
                    item["capability_key"],
                    item["name"],
                    item["status"],
                    item["summary"],
                    _json(item["evidence_files"]),
                    _json(item["verification_tests"]),
                    item["confidence"],
                    item["last_verified_at"],
                    _json(item.get("limits", [])),
                    _json({"group": item.get("group"), "blockers": item.get("blockers", [])}),
                ),
            )
        conn.commit()
    return {"count": len(items), "items": items}


def list_capability_evidence(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    refresh_capability_evidence()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM capability_evidence ORDER BY confidence DESC, updated_at DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["evidence_files"] = _loads(item.pop("evidence_files_json", None), [])
        item["verification_tests"] = _loads(item.pop("verification_tests_json", None), [])
        item["limits"] = _loads(item.pop("limits_json", None), [])
        item["metadata"] = _loads(item.pop("metadata_json", None), {})
        items.append(item)
    return items


def _detect_focus(message: str) -> str:
    compact = message.replace(" ", "").lower()
    for focus, keywords in SELF_REPORT_KEYWORDS.items():
        if any(keyword.replace(" ", "").lower() in compact for keyword in keywords):
            return focus
    return "general"


def build_self_report_context(user_message: str, *, capability_map: dict[str, Any] | None = None, metrics: dict[str, Any] | None = None, limit: int = 5) -> dict[str, Any]:
    init_db()
    data = capability_map or collect_capability_map()
    metric_data = metrics or collect_metrics()
    changes = list_change_log(limit=limit)
    evidence = list_capability_evidence(limit=limit)
    focus = _detect_focus(user_message)
    recent_features = [item.get("feature_name") for item in changes if item.get("feature_name")]
    top_capabilities = [
        {
            "name": item.get("name"),
            "status": item.get("status"),
            "summary": item.get("summary"),
            "evidence_files": item.get("evidence_files", [])[:3],
            "verification_tests": item.get("verification_tests", [])[:3],
            "confidence": item.get("confidence"),
        }
        for item in evidence
    ]
    weak_points = []
    renderer_success = metric_data.get("renderer_success_rate")
    if isinstance(renderer_success, (int, float)) and renderer_success < 0.95:
        weak_points.append(f"대화 렌더러 성공률이 {renderer_success:.1%}라 fallback 병목이 남아 있음")
    if metric_data.get("queued_autonomous_tasks_count", 0):
        weak_points.append("자율 작업 큐에 대기 중인 항목이 있음")
    if not weak_points:
        weak_points.append("현재 스냅샷 기준 큰 차단은 없지만, 실제 코드 작업 성공률은 계속 관측해야 함")
    return {
        "created_at": now_kst(),
        "focus": focus,
        "basis": "core_change_log + capability_evidence + metrics + runtime capability map",
        "recent_changes": [
            {
                "short_hash": item.get("short_hash"),
                "feature_name": item.get("feature_name"),
                "summary": item.get("human_summary"),
                "changed_files": item.get("changed_files", [])[:8],
                "capability_delta": item.get("capability_delta", []),
                "verification": item.get("verification", {}),
            }
            for item in changes
        ],
        "recent_features": recent_features[:limit],
        "top_capabilities": top_capabilities,
        "verification_snapshot": {
            "last_eval_result": metric_data.get("last_eval_result"),
            "last_eval_score": metric_data.get("last_eval_score"),
            "renderer_success_rate": renderer_success,
            "memory_vector_coverage": metric_data.get("memory_vector_coverage"),
            "schema_version": None,
        },
        "capability_limits": data.get("limits", [])[:5],
        "weak_points": weak_points[:5],
        "claim_rule": "자기 능력/개선/상태를 말할 때는 최근 변경, capability evidence, 테스트/평가 결과를 근거로 말하고 실제 파일을 방금 읽은 척하지 않는다.",
    }
