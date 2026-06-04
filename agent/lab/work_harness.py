from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.core.failure import failure_report

CONTRACT_VERSION = "0.1"

DEFAULT_PHASES = ("scope", "isolate", "implement", "verify", "review", "report")
SELF_IMPROVEMENT_PHASES = ("scope", "isolate", "implement", "verify", "review", "approve", "report")

FORBIDDEN_ACTIONS = (
    "read_or_store_secrets",
    "modify_env_or_key_files",
    "destructive_commands",
    "sudo_or_systemd_writes",
    "package_install_or_external_network",
    "unrestricted_autonomy_claims",
)

SECRET_TEXT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,'\"]+"),
    re.compile(r"(?im)^\s*[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API[_-]?KEY)[A-Z0-9_]*\s*=.*$"),
)


@dataclass(frozen=True)
class WorkHarnessContract:
    version: str
    mode: str
    backend: str
    sandbox: str
    goal_id: int | None
    task_id: int | None
    self_improvement: bool
    objective_excerpt: str
    phases: list[str]
    max_iterations: int
    verification_commands: list[str]
    acceptance_criteria: list[str]
    forbidden_actions: list[str] = field(default_factory=lambda: list(FORBIDDEN_ACTIONS))
    evidence_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def redact_harness_text(text: str | None, max_chars: int = 1200) -> str:
    result = str(text or "")
    for pattern in SECRET_TEXT_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    result = re.sub(r"\s+", " ", result).strip()
    if len(result) > max_chars:
        return result[:max_chars] + "...[truncated]"
    return result


def build_work_harness_contract(
    user_request: str,
    *,
    goal_id: int | None,
    task_id: int | None,
    backend: str,
    sandbox: str,
    max_iterations: int,
    verification_commands: list[str] | None = None,
    self_improvement: bool = False,
) -> WorkHarnessContract:
    phases = list(SELF_IMPROVEMENT_PHASES if self_improvement else DEFAULT_PHASES)
    commands = list(verification_commands or [])
    acceptance = [
        "작업 범위와 변경 이유가 보고서에 남아야 함",
        "변경 파일 목록과 위험 파일 감지가 남아야 함",
        "가능한 검증 명령 결과가 evidence ledger에 남아야 함",
        "main 반영은 release gate와 승인 흐름을 거쳐야 함",
    ]
    if self_improvement:
        acceptance.append("자가개선은 격리 worktree에서만 생성되어야 함")
    return WorkHarnessContract(
        version=CONTRACT_VERSION,
        mode="native_work_harness",
        backend=backend,
        sandbox=sandbox,
        goal_id=goal_id,
        task_id=task_id,
        self_improvement=self_improvement,
        objective_excerpt=redact_harness_text(user_request, 600),
        phases=phases,
        max_iterations=max(1, int(max_iterations)),
        verification_commands=commands,
        acceptance_criteria=acceptance,
        evidence_schema={
            "iteration": "1-based retry index",
            "codex_returncode": "Codex executor exit code",
            "changed_files": "git status --short after iteration",
            "unsafe_changed_files": "secret or credential boundary hits",
            "verification": "commands, return codes, and redacted output",
        },
    )


def format_work_harness_prompt(contract: WorkHarnessContract) -> str:
    payload = json.dumps(contract.to_dict(), ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "Work Harness Contract:",
            payload,
            "",
            "Contract rules:",
            "- Treat the contract as the source of truth for scope, safety, evidence, and completion.",
            "- Do not skip verification silently; record exact blockers when verification cannot run.",
            "- Keep implementation small enough to review and roll back.",
        ]
    )


def summarize_work_harness(contract: WorkHarnessContract) -> dict[str, Any]:
    return {
        "version": contract.version,
        "mode": contract.mode,
        "backend": contract.backend,
        "sandbox": contract.sandbox,
        "goal_id": contract.goal_id,
        "task_id": contract.task_id,
        "self_improvement": contract.self_improvement,
        "phases": contract.phases,
        "max_iterations": contract.max_iterations,
        "verification_count": len(contract.verification_commands),
        "acceptance_criteria": contract.acceptance_criteria,
    }


def verification_gate_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    latest = []
    for item in reversed(evidence):
        verification = item.get("verification") if isinstance(item, dict) else None
        if isinstance(verification, list) and verification:
            latest = [row for row in verification if isinstance(row, dict)]
            break
    passed = bool(latest) and all(row.get("returncode") == 0 for row in latest)
    return {
        "passed": passed,
        "total": len(latest),
        "failed": [str(row.get("command") or "") for row in latest if row.get("returncode") != 0],
    }


def build_work_recovery_plan(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "")
    review = result.get("code_review") if isinstance(result.get("code_review"), dict) else {}
    release_gate = result.get("release_gate") if isinstance(result.get("release_gate"), dict) else review.get("release_gate") if isinstance(review.get("release_gate"), dict) else {}
    verification_gate = result.get("verification_gate") if isinstance(result.get("verification_gate"), dict) else verification_gate_summary(result.get("evidence_ledger") if isinstance(result.get("evidence_ledger"), list) else [])
    failure = failure_report(result)
    approval_id = result.get("approval_id")
    changed_files = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
    unsafe_files = result.get("unsafe_changed_files") if isinstance(result.get("unsafe_changed_files"), list) else []

    if status == "codex_work_completed" and approval_id:
        phase = "approval"
        next_action = f"#승인 채널에서 승인 #{approval_id}을 확인한 뒤 main 반영 여부를 결정"
        retryable = False
    elif status == "codex_work_completed" and release_gate.get("status") in {"needs_review", "needs_validation"}:
        phase = "review"
        next_action = str(review.get("next_action") or release_gate.get("next_action") or "검증/리뷰 증거를 보강")
        retryable = True
    elif status == "codex_work_completed":
        phase = "done"
        next_action = "작업 보고서와 변경 파일을 확인"
        retryable = False
    elif unsafe_files:
        phase = "blocked"
        next_action = "민감 파일 변경을 제거하고 허용 범위 안에서 새 작업으로 다시 시도"
        retryable = False
    elif not verification_gate.get("passed") and verification_gate.get("total", 0) > 0:
        phase = "repair"
        failed = verification_gate.get("failed") or []
        target = failed[0] if failed else "검증 명령"
        next_action = f"실패한 검증부터 최소 재현으로 고치기: {target}"
        retryable = True
    elif status in {"codex_work_failed", "codex_work_blocked"}:
        phase = "blocked"
        next_action = failure.get("recovery_hint") or "실패 보고서를 확인하고 범위를 줄여 재시도"
        retryable = failure.get("category") in {"verification_failed", "tool_error", "timeout", "environment_issue", "unknown"}
    else:
        phase = "review"
        next_action = "작업 결과와 증거를 확인"
        retryable = False

    return {
        "phase": phase,
        "failure_category": failure.get("category"),
        "recovery_hint": failure.get("recovery_hint"),
        "next_action": next_action,
        "retryable": bool(retryable),
        "approval_id": approval_id,
        "changed_file_count": len(changed_files),
        "unsafe_file_count": len(unsafe_files),
        "verification": verification_gate,
        "release_gate_status": release_gate.get("status"),
        "review_verdict": review.get("verdict"),
    }


def build_work_operator_summary(result: dict[str, Any]) -> str:
    plan = result.get("recovery_plan") if isinstance(result.get("recovery_plan"), dict) else build_work_recovery_plan(result)
    status = str(result.get("status") or "")
    if status == "codex_work_completed" and plan.get("approval_id"):
        return f"코드는 격리 작업공간에 만들어졌고, main 반영은 승인 #{plan['approval_id']} 대기 중이야."
    if status == "codex_work_completed":
        return "코드 작업은 끝났고, 검증/리뷰 결과를 기록했어."
    if plan.get("phase") == "repair":
        return f"코드는 만들었지만 검증이 실패했어. 다음은 {plan.get('next_action')}"
    if status == "codex_work_blocked":
        return f"안전 게이트에서 멈췄어. 다음은 {plan.get('next_action')}"
    if status == "codex_work_failed":
        return f"작업자가 실패했어. 다음은 {plan.get('next_action')}"
    return str(plan.get("next_action") or "작업 결과 확인 필요")
