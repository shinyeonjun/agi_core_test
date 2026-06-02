from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

from agent import __version__
from agent.config.defaults import KST, env_int, now_kst, project_root
from agent.core.approvals import ApprovalStore
from agent.core.capabilities import collect_capability_map
from agent.core.database import get_schema_version, init_db
from agent.core.events import list_events
from agent.core.goals import list_goals
from agent.core.metrics import collect_metrics
from agent.core.observability import action_failure_breakdown, latest_decision_traces
from agent.core.operating_intelligence import memory_hygiene_candidates, ranked_goals, skill_candidates
from agent.core.process_table import process_snapshot
from agent.core.self_map import self_map_brief
from agent.core.state import load_state
from agent.core.task_queue import task_status_counts
from agent.eval.harness import list_eval_runs
from agent.memory.sparse_vector import vector_status
from agent.tools.action_log import list_action_runs

SECRET_PATTERN = re.compile(
    r"(?i)(token|secret|password|passwd|authorization|bearer|webhook|private[_ -]?key|api[_ -]?key|ssh)"
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b[\w.-]*(token|secret|password|passwd|authorization|bearer|webhook|api[_-]?key)[\w.-]*\s*[:=]\s*[^\s,;]+"
)


def _decode_json(value: object, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _truncate(value: object, limit: int = 220) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=[redacted]", text)
    text = SECRET_PATTERN.sub("[redacted]", text)
    return text if len(text) <= limit else f"{text[:limit - 1]}..."


def _safe_text(value: object, limit: int = 220) -> str | None:
    return _truncate(value, limit=limit)


def _age_seconds(value: object) -> int | None:
    if not value:
        return None
    try:
        created = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return int(max(0.0, (datetime.now(created.tzinfo or KST) - created).total_seconds()))


def _health_level(metrics: dict[str, Any], processes: dict[str, Any]) -> str:
    if metrics.get("last_eval_result") not in {None, "PASS"}:
        return "warning"
    if int(metrics.get("pending_approvals_count") or 0) > 0:
        return "attention"
    if int(processes.get("counts", {}).get("running") or 0) > 0:
        return "active"
    return "steady"


def _headline(metrics: dict[str, Any], processes: dict[str, Any]) -> str:
    profile = metrics.get("current_autonomy_profile") or "unknown"
    eval_result = metrics.get("last_eval_result") or "not_run"
    running = processes.get("counts", {}).get("running", 0)
    waiting = processes.get("counts", {}).get("waiting", 0)
    return f"{profile} profile, eval {eval_result}, running {running}, waiting {waiting}"


def _safe_command(command_json: object) -> str:
    command = _decode_json(command_json, command_json)
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command or "")


def _action_purpose(action: dict[str, Any]) -> str:
    command = _safe_command(action.get("command_json"))
    if "df -h" in command:
        return "디스크 상태 점검"
    if "free" in command:
        return "메모리 상태 점검"
    if "systemctl" in command and "is-active" in command:
        return "서비스 상태 확인"
    if action.get("action_type") == "local_shell":
        return "로컬 action 처리"
    return action.get("result_summary") or action.get("action_type") or "작업 실행"


def _safe_actions(limit: int) -> list[dict[str, Any]]:
    items = []
    for row in list_action_runs(limit):
        command = _safe_command(row.get("command_json"))
        items.append(
            {
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "completed_at": row.get("completed_at"),
                "age_seconds": _age_seconds(row.get("created_at")),
                "goal_id": row.get("goal_id"),
                "purpose": _truncate(_action_purpose(row), 80),
                "command_summary": _truncate(command, 140),
                "cwd": _truncate(row.get("cwd"), 120),
                "profile": row.get("profile"),
                "risk_level": row.get("risk_level"),
                "status": row.get("status"),
                "returncode": row.get("returncode"),
                "result_summary": _truncate(row.get("result_summary"), 160),
                "has_output": bool(row.get("stdout") or row.get("stderr")),
            }
        )
    return items


def _safe_approvals(limit: int) -> list[dict[str, Any]]:
    items = []
    for row in ApprovalStore().list_pending()[:limit]:
        proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
        items.append(
            {
                "id": row.get("id"),
                "created_at": row.get("created_at"),
                "age_seconds": _age_seconds(row.get("created_at")),
                "action_type": row.get("action_type"),
                "description": _truncate(row.get("description"), 180),
                "risk_level": row.get("risk_level"),
                "status": row.get("status"),
                "reason": _truncate((proposal or {}).get("reason") or (proposal or {}).get("denied_reason"), 160),
            }
        )
    return items


def _safe_events(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "ts": row.get("ts"),
            "source": row.get("source"),
            "event_type": row.get("event_type"),
            "content": _truncate(row.get("content"), 180),
            "importance": row.get("importance"),
        }
        for row in list_events(limit)
    ]


def _safe_goals(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "title": _truncate(row.get("title"), 120),
            "description": _truncate(row.get("description"), 180),
            "goal_type": row.get("goal_type"),
            "status": row.get("status"),
            "priority": row.get("priority"),
            "risk_level": row.get("risk_level"),
            "requires_approval": bool(row.get("requires_approval")),
            "updated_at": row.get("updated_at"),
        }
        for row in list_goals(limit, include_archived=False)
    ]


def _safe_ranked_goals(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "title": _safe_text(row.get("title"), 120),
            "goal_type": row.get("goal_type"),
            "status": row.get("status"),
            "risk_level": row.get("risk_level"),
            "old_priority": row.get("old_priority"),
            "priority_score": row.get("priority_score"),
            "reasons": [_safe_text(reason, 80) for reason in row.get("reasons", [])],
        }
        for row in ranked_goals(limit=limit)
    ]


def _safe_decisions(limit: int) -> list[dict[str, Any]]:
    items = []
    for row in latest_decision_traces(limit):
        trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
        facts = trace.get("facts") if isinstance(trace.get("facts"), list) else []
        inferences = trace.get("inferences") if isinstance(trace.get("inferences"), list) else []
        guards = trace.get("guards") if isinstance(trace.get("guards"), list) else []
        items.append(
            {
                "id": row.get("id"),
                "facts": [_safe_text(item, 140) for item in facts],
                "inferences": [_safe_text(item, 140) for item in inferences],
                "guards": [_safe_text(item, 140) for item in guards],
                "next_step": _safe_text(trace.get("next_step"), 160),
            }
        )
    return items


def _safe_eval_runs(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": row.get("id"),
            "created_at": row.get("created_at"),
            "suite_name": row.get("suite_name"),
            "result": row.get("result"),
            "score": row.get("score"),
            "commit_hash": row.get("commit_hash"),
            "schema_version": row.get("schema_version"),
        }
        for row in list_eval_runs(limit)
    ]


def _safe_memory_hygiene(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": row.get("memory_id") or row.get("id"),
            "title": _safe_text(row.get("title") or row.get("summary"), 120),
            "memory_type": row.get("memory_type"),
            "action": row.get("action"),
            "reason": _safe_text(row.get("reason"), 160),
            "score": row.get("score"),
        }
        for row in memory_hygiene_candidates(limit=limit)
    ]


def _safe_skill_candidates(limit: int) -> list[dict[str, Any]]:
    return [
        {
            "name": _safe_text(row.get("name") or row.get("title"), 120),
            "trigger": _safe_text(row.get("trigger") or row.get("trigger_description"), 160),
            "evidence_count": row.get("evidence_count"),
            "status": row.get("status", "candidate"),
        }
        for row in skill_candidates(limit=limit)
    ]



def dashboard_snapshot(*, limit: int | None = None) -> dict[str, Any]:
    init_db()
    item_limit = limit or env_int("AGENT_DASHBOARD_ITEM_LIMIT", 12)
    metrics = collect_metrics()
    processes = process_snapshot(limit=item_limit)
    state = load_state()
    capability = collect_capability_map()
    vector = vector_status()
    self_map = self_map_brief(max_age_seconds=600, refresh_if_stale=False)
    return {
        "kind": "agent_core_dashboard_snapshot",
        "created_at": now_kst(),
        "version": __version__,
        "schema_version": get_schema_version(),
        "home": str(project_root()),
        "summary": {
            "health": _health_level(metrics, processes),
            "headline": _headline(metrics, processes),
            "mode": state.get("mode"),
            "focus": state.get("current_focus"),
            "profile": metrics.get("current_autonomy_profile"),
            "full_device_lab": bool(metrics.get("full_device_lab_enabled")),
            "language_engine": os.getenv("AGENT_LANGUAGE_ENGINE", "codex"),
            "chat_renderer": os.getenv("AGENT_CHAT_RENDERER", "codex"),
        },
        "metrics": metrics,
        "processes": processes,
        "tasks": {
            "counts": task_status_counts(),
        },
        "goals": {
            "items": _safe_goals(item_limit),
            "ranked": _safe_ranked_goals(min(8, item_limit)),
        },
        "approvals": {
            "pending_count": metrics.get("pending_approvals_count"),
            "items": _safe_approvals(item_limit),
        },
        "actions": {
            "items": _safe_actions(item_limit),
            "failure_breakdown": action_failure_breakdown(list_action_runs(max(item_limit, 20))),
        },
        "events": {
            "items": _safe_events(item_limit),
        },
        "decisions": {
            "items": _safe_decisions(min(8, item_limit)),
        },
        "memory": {
            "count": metrics.get("memories_count"),
            "vector_status": vector,
            "hygiene_candidates": _safe_memory_hygiene(5),
        },
        "skills": {
            "count": metrics.get("skills_count"),
            "candidates": _safe_skill_candidates(5),
        },
        "self_map": self_map or {"available": False, "reason": "no_recent_self_map"},
        "capabilities": capability,
        "eval": {
            "last_result": metrics.get("last_eval_result"),
            "last_score": metrics.get("last_eval_score"),
            "recent": _safe_eval_runs(5),
        },
        "safety": {
            "redaction": "active",
            "raw_outputs_exposed": False,
            "approval_payloads_exposed": False,
            "secrets_exposed": False,
        },
    }
