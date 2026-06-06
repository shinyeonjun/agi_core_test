from __future__ import annotations

from typing import Any


def format_capability_proposal_response(payload: dict[str, Any]) -> str:
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
    kind = str(payload.get("kind") or "gap")
    name = str(proposal.get("capability_name") or "새 능력")
    purpose = str(proposal.get("purpose") or "아직 설명이 부족해.")
    risk = str(proposal.get("risk_level") or "unknown")
    status = str(proposal.get("status") or "proposed")
    prefix = "이미 같은 능력 후보가 올라와 있어. 기존 후보에 묶어둘게." if kind == "duplicate" else "지금은 그 능력이 없어. 능력 후보로 올려둘게."
    return "\n".join(
        [
            prefix,
            "",
            f"- 이름: {name}",
            f"- 하는 일: {purpose}",
            f"- 위험도: {risk}",
            f"- 상태: {status}",
            "",
            "아래 버튼으로 개발 후보 승인/보류/거절을 고를 수 있어.",
        ]
    )


def format_work_item_response(work_item: dict[str, Any], payload: dict[str, Any]) -> str:
    title = str(work_item.get("title") or "작업")
    goal = str(work_item.get("goal") or "")
    priority = str(work_item.get("priority") or "medium")
    risk = str(work_item.get("risk_level") or "low")
    status = str(work_item.get("status") or "proposed")
    metadata = work_item.get("metadata_json") if isinstance(work_item.get("metadata_json"), dict) else {}
    deliverables = metadata.get("deliverables") if isinstance(metadata, dict) else []
    lines = [
        "좋아. 이건 바로 실행보다 작업으로 잡아두는 게 맞아.",
        "",
        f"- 이름: {title}",
        f"- 목표: {goal}",
        f"- 우선순위: {priority}",
        f"- 위험도: {risk}",
        f"- 상태: {status}",
    ]
    if isinstance(deliverables, list) and deliverables:
        lines.append("- 결과물: " + ", ".join(str(item) for item in deliverables[:5]))
    lines.extend(["", "아래 버튼으로 진행 후보 승인/보류/거절을 고를 수 있어."])
    return "\n".join(lines)
