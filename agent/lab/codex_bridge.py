from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.autonomy import get_autonomy_state
from agent.core.drives import compute_drives
from agent.core.goals import list_goals
from agent.core.metrics import collect_metrics
from agent.lab.proposals import list_action_proposals
from agent.tools.action_log import list_action_runs
from agent.workspace.executor import write_text_artifact


def _safe_action_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "profile": row.get("profile"),
        "risk_level": row.get("risk_level"),
        "status": row.get("status"),
        "returncode": row.get("returncode"),
        "result_summary": row.get("result_summary"),
    }


def build_codex_lab_context(limit: int = 10) -> dict[str, Any]:
    return {
        "created_at": now_kst(),
        "purpose": "codex_lab_planner_context",
        "safety_contract": {
            "codex_auto_execute": False,
            "secret_material_included": False,
            "allowed_output": "proposals_or_patch_plan_only",
            "human_or_policy_gate_required_for_execution": True,
        },
        "autonomy": get_autonomy_state(),
        "drives": compute_drives(),
        "open_goals": [
            {"id": row.get("id"), "title": row.get("title"), "goal_type": row.get("goal_type"), "status": row.get("status"), "priority": row.get("priority")}
            for row in list_goals(limit=limit)
        ],
        "recent_proposals": [
            {"id": row.get("id"), "command": row.get("command"), "profile": row.get("profile"), "risk_level": row.get("risk_level"), "status": row.get("status"), "reason": row.get("reason")}
            for row in list_action_proposals(limit=limit)
        ],
        "recent_actions": [_safe_action_summary(row) for row in list_action_runs(limit)],
        "metrics": collect_metrics(),
        "request": "Suggest the next safe lab proposal. Do not request secrets. Do not execute commands.",
    }


def write_codex_lab_context(limit: int = 10) -> dict[str, Any]:
    context = build_codex_lab_context(limit=limit)
    stamp = now_kst().replace(":", "").replace("+", "_")
    artifact = write_text_artifact(
        "reports",
        f"{stamp}-codex-lab-context.json",
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        "codex_lab_context",
        "Codex lab planner context",
        {"source": "codex_bridge", "auto_execute": False},
    )
    return {"artifact": artifact, "context": context}
