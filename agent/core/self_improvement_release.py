from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence


RISK_LEVELS = ("low", "medium", "high", "critical")

SECRET_FILE_PATTERNS = (
    re.compile(r"(^|/)\.env(?:[.\w-]*)?$", re.IGNORECASE),
    re.compile(r"(^|/)\.ssh(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(id_rsa|id_ed25519|authorized_keys)$", re.IGNORECASE),
    re.compile(r"(^|/)(secrets?|credentials|token)([.\w-]*)?$", re.IGNORECASE),
    re.compile(r"(^|/)\.(npmrc|pypirc|netrc)$", re.IGNORECASE),
)

SERVICE_FILE_PATTERNS = (
    re.compile(r"(^|/)(systemd|services?)(/|$)", re.IGNORECASE),
    re.compile(r"\.service$", re.IGNORECASE),
)

DEPENDENCY_FILE_PATTERNS = (
    re.compile(r"(^|/)(requirements.*\.txt|pyproject\.toml|poetry\.lock|uv\.lock)$", re.IGNORECASE),
    re.compile(r"(^|/)(package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock)$", re.IGNORECASE),
)

MIGRATION_FILE_PATTERNS = (
    re.compile(r"(^|/)migrations?(/|$)", re.IGNORECASE),
    re.compile(r"db_migrations\.py$", re.IGNORECASE),
)


@dataclass(frozen=True)
class ReleaseGate:
    name: str
    label: str
    status: str
    reason: str
    evidence: dict[str, Any]


def _normalized_path(value: str) -> str:
    text = (value or "").replace("\\", "/").strip()
    if not text:
        return ""
    if " -> " in text:
        text = text.rsplit(" -> ", 1)[-1].strip()
    if len(text) >= 4 and text[2] == " ":
        return text[3:].strip()
    parts = text.split(maxsplit=1)
    if len(parts) == 2 and len(parts[0]) <= 3:
        return parts[1].strip()
    return text


def normalize_changed_files(changed_files: Sequence[str]) -> list[str]:
    normalized = [_normalized_path(item) for item in changed_files]
    return sorted({path[:240] for path in normalized if path})


def _matches(path: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(path) for pattern in patterns)


def _risk_rank(level: str) -> int:
    value = (level or "medium").strip().lower()
    return RISK_LEVELS.index(value) if value in RISK_LEVELS else RISK_LEVELS.index("medium")


def _max_risk(*levels: str) -> str:
    return max((level for level in levels if level), key=_risk_rank, default="medium")


def classify_release_paths(changed_files: Sequence[str]) -> dict[str, Any]:
    paths = normalize_changed_files(changed_files)
    secret_paths = [path for path in paths if _matches(path, SECRET_FILE_PATTERNS)]
    service_paths = [path for path in paths if _matches(path, SERVICE_FILE_PATTERNS)]
    dependency_paths = [path for path in paths if _matches(path, DEPENDENCY_FILE_PATTERNS)]
    migration_paths = [path for path in paths if _matches(path, MIGRATION_FILE_PATTERNS)]
    tags: list[str] = []
    if secret_paths:
        tags.append("secret_boundary")
    if service_paths:
        tags.append("service_change")
    if dependency_paths:
        tags.append("dependency_change")
    if migration_paths:
        tags.append("schema_change")
    if not tags and paths:
        tags.append("repo_change")
    inferred_risk = "low"
    if paths:
        inferred_risk = "medium"
    if dependency_paths or migration_paths:
        inferred_risk = "high"
    if service_paths:
        inferred_risk = "high"
    if secret_paths:
        inferred_risk = "critical"
    return {
        "changed_files": paths,
        "secret_paths": secret_paths,
        "service_paths": service_paths,
        "dependency_paths": dependency_paths,
        "migration_paths": migration_paths,
        "change_tags": tags,
        "inferred_risk": inferred_risk,
    }


def build_self_improvement_release_plan(
    title: str,
    *,
    request: str = "",
    risk_level: str = "medium",
    owner: str = "user",
) -> dict[str, Any]:
    normalized_risk = _max_risk(risk_level)
    return {
        "title": title,
        "request": request,
        "owner": owner,
        "risk_level": normalized_risk,
        "principles": [
            "작업은 main이 아니라 격리된 worktree에서 먼저 만든다.",
            "변경 이유, 파일, 검증 결과, 남은 위험을 한 묶음으로 남긴다.",
            "테스트, audit, eval, 리뷰, 승인 중 하나라도 비면 main 적용은 보류한다.",
            "시크릿, 인증, systemd, 패키지, DB 스키마는 더 높은 위험으로 본다.",
            "적용 전에는 되돌리는 방법이 있어야 한다.",
        ],
        "phases": [
            {"name": "scope", "label": "요청 범위 확정", "done_when": "작업 목표와 건드릴 영역이 짧게 적힌다."},
            {"name": "isolate", "label": "격리 작업", "done_when": "git worktree 또는 동등한 격리 공간에서 변경한다."},
            {"name": "implement", "label": "구현", "done_when": "변경 파일과 의도가 기록된다."},
            {"name": "verify", "label": "검증", "done_when": "관련 테스트, agentctl audit, agentctl eval run이 통과한다."},
            {"name": "review", "label": "리뷰", "done_when": "Core 또는 사람이 변경 근거와 위험을 검토한다."},
            {"name": "approve", "label": "승인", "done_when": "main 반영 전 사람이 승인한다."},
            {"name": "apply", "label": "적용", "done_when": "승인된 변경만 main에 반영한다."},
            {"name": "observe", "label": "적용 후 관찰", "done_when": "서비스 상태와 최근 오류를 확인한다."},
            {"name": "rollback", "label": "되돌리기", "done_when": "문제가 있으면 이전 commit/worktree 상태로 복구할 수 있다."},
        ],
        "research_basis": [
            {"key": "react", "lesson": "생각, 행동, 관찰을 한 흐름으로 기록한다."},
            {"key": "reflexion", "lesson": "실패와 피드백을 다음 시도에 쓰는 언어 기억으로 남긴다."},
            {"key": "self_refine", "lesson": "초안, 피드백, 수정 반복을 작은 루프로 만든다."},
            {"key": "swe_agent", "lesson": "코드 작업자는 repo 탐색, 편집, 테스트 인터페이스가 명확해야 한다."},
            {"key": "deployment_gates", "lesson": "중요 변경은 보호 규칙과 승인 없이는 적용하지 않는다."},
            {"key": "progressive_delivery", "lesson": "적용은 관찰 가능한 지표와 되돌리기 계획을 동반한다."},
        ],
    }


def _gate(name: str, label: str, status: str, reason: str, **evidence: Any) -> ReleaseGate:
    return ReleaseGate(name=name, label=label, status=status, reason=reason, evidence=evidence)


def _verification_gate(name: str, label: str, passed: bool, *, required: bool = True) -> ReleaseGate:
    if passed:
        return _gate(name, label, "pass", "통과", required=required)
    status = "fail" if required else "needs_review"
    reason = "필수 검증이 통과하지 않았다" if required else "검증 결과가 비어 있어 리뷰가 필요하다"
    return _gate(name, label, status, reason, required=required)


def evaluate_release_candidate(
    *,
    changed_files: Sequence[str],
    worktree_isolated: bool,
    tests_passed: bool,
    audit_passed: bool,
    eval_passed: bool,
    review_passed: bool,
    approval_status: str = "pending",
    risk_level: str = "medium",
    rollback_plan: str | None = None,
    secrets_touched: bool = False,
    destructive_change: bool = False,
) -> dict[str, Any]:
    path_info = classify_release_paths(changed_files)
    paths = path_info["changed_files"]
    effective_risk = _max_risk(risk_level, str(path_info["inferred_risk"]))
    approval = (approval_status or "pending").strip().lower()
    gates: list[ReleaseGate] = []

    gates.append(
        _gate(
            "scope",
            "변경 범위",
            "pass" if paths else "fail",
            "변경 파일이 확인됐다" if paths else "변경 파일이 없다",
            changed_files=paths,
        )
    )
    gates.append(
        _gate(
            "isolation",
            "작업 격리",
            "pass" if worktree_isolated else "fail",
            "격리된 작업 공간에서 만든 변경" if worktree_isolated else "main 또는 비격리 공간의 변경은 바로 적용하지 않는다",
        )
    )
    unsafe = sorted(set(path_info["secret_paths"] + (paths if secrets_touched else [])))
    gates.append(
        _gate(
            "secret_boundary",
            "시크릿 경계",
            "fail" if unsafe else "pass",
            "민감 파일/값 접근 가능성이 있다" if unsafe else "민감 파일 변경은 감지되지 않았다",
            unsafe_paths=unsafe,
        )
    )
    gates.append(
        _gate(
            "destructive_boundary",
            "파괴적 변경",
            "fail" if destructive_change else "pass",
            "파괴적 변경은 release gate에서 막는다" if destructive_change else "파괴적 변경 신호 없음",
        )
    )
    gates.extend(
        [
            _verification_gate("tests", "테스트", tests_passed),
            _verification_gate("audit", "감사", audit_passed),
            _verification_gate("eval", "평가", eval_passed),
            _verification_gate("review", "리뷰", review_passed, required=False),
        ]
    )
    rollback_required = bool(paths)
    gates.append(
        _gate(
            "rollback",
            "되돌리기",
            "pass" if (not rollback_required or bool(rollback_plan)) else "needs_review",
            "되돌리기 계획 있음" if rollback_plan else "main 반영 전 되돌리기 계획이 필요하다",
            required=rollback_required,
            rollback_plan=rollback_plan,
        )
    )
    if approval == "approved":
        approval_gate = _gate("approval", "사람 승인", "pass", "승인됨", approval_status=approval)
    elif approval == "rejected":
        approval_gate = _gate("approval", "사람 승인", "fail", "거절됨", approval_status=approval)
    else:
        approval_gate = _gate("approval", "사람 승인", "needs_approval", "main 반영 전 승인 필요", approval_status=approval)
    gates.append(approval_gate)

    failing = [gate for gate in gates if gate.status == "fail"]
    needs_review = [gate for gate in gates if gate.status == "needs_review"]
    needs_approval = [gate for gate in gates if gate.status == "needs_approval"]
    if failing:
        status = "blocked"
        decision = "적용 금지"
    elif needs_review:
        status = "needs_review"
        decision = "리뷰 보강 필요"
    elif needs_approval:
        status = "needs_approval"
        decision = "승인 대기"
    else:
        status = "ready_to_apply"
        decision = "승인된 변경이라 main 반영 가능"

    return {
        "status": status,
        "decision": decision,
        "can_apply_to_main": status == "ready_to_apply",
        "can_auto_apply": False,
        "effective_risk": effective_risk,
        "approval_required": True,
        "rollback_required": rollback_required,
        "changed_files": paths,
        "change_tags": path_info["change_tags"],
        "path_risk": path_info,
        "gates": [asdict(gate) for gate in gates],
        "failed_gates": [gate.name for gate in failing],
        "review_gates": [gate.name for gate in needs_review],
        "approval_gates": [gate.name for gate in needs_approval],
        "next_action": _next_action(status, failing, needs_review, needs_approval),
    }


def _next_action(
    status: str,
    failing: Sequence[ReleaseGate],
    needs_review: Sequence[ReleaseGate],
    needs_approval: Sequence[ReleaseGate],
) -> str:
    if status == "blocked":
        labels = ", ".join(gate.label for gate in failing[:3])
        return f"막힌 게이트를 먼저 고쳐야 한다: {labels}"
    if status == "needs_review":
        labels = ", ".join(gate.label for gate in needs_review[:3])
        return f"리뷰/증거를 보강해야 한다: {labels}"
    if status == "needs_approval":
        labels = ", ".join(gate.label for gate in needs_approval[:3])
        return f"승인을 받아야 한다: {labels}"
    return "main 반영 후 서비스 상태와 최근 이벤트를 확인한다"
