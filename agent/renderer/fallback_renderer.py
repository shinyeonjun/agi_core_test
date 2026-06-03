from __future__ import annotations

from typing import Any


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

    goal = decision.get("selected_goal") or {}
    goal_id = decision.get("selected_goal_id") or goal.get("id") or "-"
    return "Core가 지금 입력은 기록해뒀어.\n" f"현재 기준 목표는 #{goal_id}야. 필요한 작업이면 이어서 처리할게."
