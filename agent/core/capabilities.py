from __future__ import annotations

import os
import shutil
from typing import Any

from agent.config.defaults import env_bool, now_kst
from agent.core.autonomy import current_profile
from agent.core.metrics import collect_metrics
from agent.core.self_map import self_map_brief


def _status(enabled: bool, *, available: bool = True) -> str:
    if not available:
        return "unavailable"
    return "enabled" if enabled else "disabled"


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def codex_worker_enabled() -> bool:
    return env_bool("AGENT_CODEX_WORKER_ENABLED", True)


def collect_capability_map() -> dict[str, Any]:
    profile = current_profile()
    metrics = collect_metrics()
    runtime = self_map_brief(max_age_seconds=300, refresh_if_stale=False)
    codex_available = _codex_available()
    worker_enabled = codex_worker_enabled()
    full_device = profile == "full_device_lab"
    return {
        "created_at": now_kst(),
        "profile": profile,
        "summary": "Core can converse, remember, schedule, observe the device, run policy-gated local actions, and delegate user code-change tasks to Codex when available.",
        "direct": [
            {"name": "discord_chat", "status": "enabled", "description": "Discord chat routing through Core run_talk"},
            {"name": "memory_search", "status": "enabled", "description": "FTS plus local sparse vector memory retrieval"},
            {"name": "goal_task_queue", "status": "enabled", "description": "Separate user and autonomous task queues"},
            {"name": "self_map", "status": "enabled" if runtime else "unavailable", "description": "Safe runtime body map without secret values"},
            {"name": "scheduled_tick", "status": "enabled", "description": "Idle tick, lab tick, activity summary, and daily summary timers"},
            {"name": "policy_engine", "status": "enabled", "description": "Risk classification, denial, and approval gating"},
        ],
        "worker_mediated": [
            {
                "name": "codex_language_interpreter",
                "status": _status(os.getenv("AGENT_LANGUAGE_ENGINE", "codex").strip().lower() in {"codex", "codex_cli"}, available=codex_available),
                "description": "Codex read-only JSON interpretation of user intent",
            },
            {
                "name": "codex_chat_renderer",
                "status": _status(os.getenv("AGENT_CHAT_RENDERER", "codex").strip().lower() == "codex", available=codex_available),
                "description": "Codex read-only Korean Discord response rendering",
            },
            {
                "name": "codex_work_worker",
                "status": _status(worker_enabled and full_device, available=codex_available),
                "description": "User-triggered code-change worker for repo edits, tests, and implementation reports",
            },
        ],
        "approval_required": [
            "sudo or systemd writes",
            "package install/remove/update",
            "external network fetch",
            "local file mutation outside the worker policy",
        ],
        "forbidden": [
            "secret/token/private key reading or storage",
            "credential exfiltration",
            "root/home destructive deletion",
            "network scanning or external harm",
            "claiming AGI, consciousness, or unrestricted autonomy",
        ],
        "limits": [
            "Autonomous loop currently handles observation, reports, project specs, research notes, memory hygiene, and skill review.",
            "Large project execution is user-triggered through the Codex work worker, not free-running autonomy.",
            "The model is not fine-tuned; Core stores memories, style, tasks, reflections, and vector indexes.",
        ],
        "metrics": {
            "last_eval_result": metrics.get("last_eval_result"),
            "last_eval_score": metrics.get("last_eval_score"),
            "memory_vector_coverage": metrics.get("memory_vector_coverage"),
            "renderer_success_rate": metrics.get("renderer_success_rate"),
            "queued_user_tasks": metrics.get("queued_user_tasks_count"),
            "queued_autonomous_tasks": metrics.get("queued_autonomous_tasks_count"),
        },
    }


def capability_summary_lines(capabilities: dict[str, Any] | None = None) -> list[str]:
    data = capabilities or collect_capability_map()
    worker = {item["name"]: item for item in data.get("worker_mediated", [])}
    codex_work = worker.get("codex_work_worker", {})
    return [
        f"profile={data.get('profile')}",
        f"codex_work_worker={codex_work.get('status', 'unknown')}",
        f"eval={data.get('metrics', {}).get('last_eval_result')} / {data.get('metrics', {}).get('last_eval_score')}",
        f"vector_coverage={data.get('metrics', {}).get('memory_vector_coverage')}",
    ]
