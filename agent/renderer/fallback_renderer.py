from __future__ import annotations

from typing import Any


def render(decision: dict[str, Any]) -> str:
    policy = decision.get("policy_summary", {})
    if policy.get("denied"):
        reason = policy.get("reason") or "정책 차단"
        return f"그 요청은 실행하지 않았어.\n이유: {reason}"
    user_goal = decision.get("user_directed_goal") or {}
    if user_goal:
        goal_id = user_goal.get("id") or "-"
        return f"요청은 Core 작업 흐름에 넣어뒀어.\n목표: #{goal_id}"
    return "지금 답변 렌더러가 안정적으로 끝나지 않았어.\n근거 없이 지어내진 않을게. 같은 질문을 한 번만 다시 보내줘."
