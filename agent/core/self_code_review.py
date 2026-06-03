from __future__ import annotations

from typing import Any

from agent.core.failure import failure_report
from agent.core.self_improvement_release import evaluate_release_candidate


def _latest_verification(result: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = result.get("evidence_ledger")
    if not isinstance(evidence, list):
        return []
    for item in reversed(evidence):
        verification = item.get("verification") if isinstance(item, dict) else None
        if isinstance(verification, list) and verification:
            return [entry for entry in verification if isinstance(entry, dict)]
    return []


def _verification_passed(result: dict[str, Any]) -> bool:
    verification = _latest_verification(result)
    return bool(verification) and all(item.get("returncode") == 0 for item in verification)


def _ran_command(result: dict[str, Any], needle: str) -> bool:
    text = " ".join(str(item.get("command") or "") for item in _latest_verification(result)).lower()
    return needle.lower() in text


def review_codex_work_result(result: dict[str, Any], *, self_improvement: bool = False) -> dict[str, Any]:
    changed_files = result.get("changed_files") if isinstance(result.get("changed_files"), list) else []
    unsafe = result.get("unsafe_changed_files") if isinstance(result.get("unsafe_changed_files"), list) else []
    status = str(result.get("status") or "")
    report = str(result.get("report") or "").strip()
    worktree_isolated = result.get("worktree_status") == "created" or str(result.get("integration_status") or "") == "worktree_pending_review"
    tests_passed = _verification_passed(result)
    audit_passed = _ran_command(result, "agentctl audit")
    eval_passed = _ran_command(result, "agentctl eval run")
    review_passed = bool(changed_files) and not unsafe and bool(report) and status == "codex_work_completed" and (worktree_isolated or not self_improvement)
    release_gate = evaluate_release_candidate(
        changed_files=changed_files,
        worktree_isolated=worktree_isolated,
        tests_passed=tests_passed,
        audit_passed=audit_passed,
        eval_passed=eval_passed,
        review_passed=review_passed,
        approval_status="pending",
        risk_level=str((result.get("policy") or {}).get("risk_level") or result.get("risk_level") or "medium"),
        rollback_plan=f"drop worktree branch {result.get('worktree_branch')}" if result.get("worktree_branch") else None,
        secrets_touched=bool(unsafe),
    )
    findings: list[dict[str, Any]] = []
    if self_improvement and not worktree_isolated:
        findings.append({"severity": "critical", "code": "self_improvement_not_isolated", "message": "자가개선 코드는 main이 아니라 별도 worktree에서만 만들어야 한다."})
    if unsafe:
        findings.append({"severity": "critical", "code": "unsafe_files_changed", "message": "민감 파일 변경이 감지됐다.", "files": unsafe})
    if status != "codex_work_completed":
        findings.append({"severity": "high", "code": "worker_not_completed", "message": "Codex worker가 완료 상태가 아니다.", "status": status})
    if not changed_files:
        findings.append({"severity": "medium", "code": "no_changed_files", "message": "변경 파일이 없어 적용할 코드가 없다."})
    if not tests_passed:
        findings.append({"severity": "medium", "code": "tests_missing_or_failed", "message": "최신 검증에서 테스트 통과 증거가 없다."})
    if not audit_passed:
        findings.append({"severity": "medium", "code": "audit_missing", "message": "agentctl audit 통과 증거가 아직 없다."})
    if not eval_passed:
        findings.append({"severity": "medium", "code": "eval_missing", "message": "agentctl eval run 통과 증거가 아직 없다."})

    if any(item["severity"] == "critical" for item in findings):
        verdict = "blocked"
    elif release_gate["status"] == "ready_to_apply":
        verdict = "ready_for_approval"
    elif release_gate["status"] == "needs_approval":
        verdict = "needs_approval"
    elif release_gate["status"] == "needs_review":
        verdict = "needs_review"
    else:
        verdict = "needs_validation" if findings else "needs_review"

    return {
        "verdict": verdict,
        "self_improvement": self_improvement,
        "findings": findings,
        "release_gate": release_gate,
        "failure_report": failure_report(result),
        "next_action": _next_action(verdict, findings, release_gate),
    }


def _next_action(verdict: str, findings: list[dict[str, Any]], release_gate: dict[str, Any]) -> str:
    if verdict == "blocked":
        return "치명적인 문제를 먼저 고쳐야 하며 main 반영은 금지된다."
    if verdict == "needs_validation":
        missing = [item["code"] for item in findings if item["severity"] == "medium"]
        return "검증 보강 필요: " + ", ".join(missing[:4])
    if verdict in {"needs_review", "needs_approval"}:
        return str(release_gate.get("next_action") or "리뷰 또는 승인 필요")
    if verdict == "ready_for_approval":
        return "승인 채널에서 main 반영 여부를 물어볼 수 있다."
    return "상태 재확인이 필요하다."
