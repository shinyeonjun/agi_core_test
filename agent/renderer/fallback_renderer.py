from __future__ import annotations

from typing import Any


def _chat_fallback(decision: dict[str, Any]) -> str:
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

    return _chat_fallback(decision)
