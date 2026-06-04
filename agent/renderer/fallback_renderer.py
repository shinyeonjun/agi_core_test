from __future__ import annotations

from typing import Any

from agent.renderer.human_terms import humanize_public_text


def _grounded_self_report(decision: dict[str, Any]) -> str | None:
    context = decision.get("self_report_context") if isinstance(decision.get("self_report_context"), dict) else {}
    if not context:
        return None
    focus = str(context.get("focus") or "general")
    if focus not in {"change", "capability", "status", "failure", "research"}:
        return None
    changes = context.get("recent_changes") if isinstance(context.get("recent_changes"), list) else []
    caps = context.get("top_capabilities") if isinstance(context.get("top_capabilities"), list) else []
    weak = context.get("weak_points") if isinstance(context.get("weak_points"), list) else []
    lines = ["상태 기록 기준으로 보면, 지금은 근거를 꽤 붙여서 말할 수 있는 단계야."]
    if changes:
        first = changes[0]
        lines.append(f"최근 큰 변화: {first.get('feature_name') or 'Core 변경'}")
        if first.get("short_hash"):
            lines.append(f"근거 커밋: {first.get('short_hash')}")
    if caps:
        named = ", ".join(str(item.get("name")) for item in caps[:3] if item.get("name"))
        lines.append(f"확인된 능력 근거: {named}")
    if weak:
        lines.append(f"아직 약한 점: {weak[0]}")
    lines.append("즉, 감으로 '좋아졌다'가 아니라 변경 이력, capability evidence, 테스트/평가 기록을 보고 답하는 쪽으로 바뀌는 중이야.")
    return "\n".join(lines)


def _contract_fallback(decision: dict[str, Any]) -> str | None:
    contract = decision.get("answer_contract") if isinstance(decision.get("answer_contract"), dict) else {}
    if not contract:
        return None
    kind = str(contract.get("kind") or "general")
    context = decision.get("self_report_context") if isinstance(decision.get("self_report_context"), dict) else {}
    weak = context.get("weak_points") if isinstance(context.get("weak_points"), list) else []
    metrics = context.get("verification_snapshot") if isinstance(context.get("verification_snapshot"), dict) else decision.get("metrics", {})
    renderer_rate = metrics.get("renderer_success_rate") if isinstance(metrics, dict) else None
    renderer_text = f"답변 성공률 {float(renderer_rate) * 100:.1f}%" if isinstance(renderer_rate, (int, float)) else "답변 복구 경로"
    if kind == "advice":
        first = weak[0] if weak else f"{renderer_text}가 아직 흔들림"
        return "\n".join([
            "지금 기준으로는 이 순서가 좋아.",
            f"1순위: 답변 품질 개선. 현재 약점은 '{first}' 쪽이라, 질문에 직접 답하지 못하고 상태 안내로 빠질 수 있어.",
            "2순위: 작업 루프 실패 원인 분류 강화. 차단/검증 실패/권한 문제를 나눠 기록해야 다음 재시도가 똑똑해져.",
            "3순위: 반응 루프 안정화. 정해진 시간마다 움직이는 것보다 필요한 순간에 반응하는 쪽으로 가려면 관찰 근거가 더 필요해.",
            "즉, 기능 추가보다 지금은 답변 품질, 실패 학습, 실행 루프 신뢰성을 먼저 조이는 게 맞아.",
        ])
    if kind == "direct_question":
        return "Core 기준으로 가능한 부분과 아직 약한 부분을 나눠서 볼게. 지금 답은 상태 안내로 넘길 게 아니라 질문 자체에 직접 답해야 해."
    return None


def _chat_fallback(decision: dict[str, Any]) -> str:
    contract_reply = _contract_fallback(decision)
    if contract_reply:
        return contract_reply
    grounded = _grounded_self_report(decision)
    if grounded:
        return grounded
    interpretation = decision.get("language_interpretation") if isinstance(decision.get("language_interpretation"), dict) else {}
    target = str(interpretation.get("target") or "")
    user_input = str(decision.get("user_input") or "")
    compact_input = user_input.replace(" ", "")
    if "자율" in user_input and ("생명체" in user_input or "ㄷㄷ" in user_input):
        return "완전한 생명체나 의식은 아니야. 다만 목표, 기억, 실행, 검증, 실패 학습 루프를 단단하게 만들면 자율 에이전트처럼 운용하는 건 기술적으로 가능해."
    if target == "capabilities":
        return "가능한 건 대화, 기억 검색, 목표/작업 관리, 안전한 로컬 점검, 코드 작업 위임이야. 시스템 변경이나 민감정보 접근은 승인 없이 못 해."
    if target == "architecture":
        return "Core는 대화 해석, 기억, 목표/작업 큐, 정책 게이트, 장비 관찰, Codex 작업 워커가 나뉘어 돌아가는 구조야."
    if target == "status":
        return "상태 확인은 가능해. 자세한 현재 작업은 `!work`, 열린 목표는 `!goals`, 장비/루프 상태는 `!state`로 보면 돼."
    if target == "question" or "?" in user_input or "？" in user_input or "가능" in user_input or compact_input.endswith("안돼"):
        if "core" in user_input.lower() or "코어" in user_input:
            return "Core 기준으로 가능한 부분과 아직 안 되는 부분을 나눠서 봐야 해. 단정해서 포장하진 않을게."
        return "가능 여부를 묻는 걸로 이해했어. 단정해서 포장하진 않을게. 가능한 부분과 아직 안 되는 부분을 나눠서 봐야 해."
    return "들었어. 지금은 답변 생성이 매끄럽지 않아서 짧게만 말할게. 작업 지시라면 `!work`에서 진행 여부를 확인하면 돼."


def render(decision: dict[str, Any]) -> str:
    policy = decision.get("policy_summary", {})
    if policy.get("denied"):
        reason = policy.get("reason") or "정책상 막힘"
        return f"이건 실행하지 않을게.\n이유: {reason}"

    user_goal = decision.get("user_directed_goal") or {}
    if user_goal:
        goal_id = user_goal.get("id") or "-"
        if user_goal.get("self_improvement"):
            task_id = user_goal.get("task_id") or "-"
            return f"자가개선 작업으로 잡았어.\n목표: #{goal_id}\n작업: #{task_id}\nmain 반영은 승인 전에는 안 해."
        return f"작업으로 넘겼어.\n목표: #{goal_id}"

    return humanize_public_text(_chat_fallback(decision))