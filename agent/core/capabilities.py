from __future__ import annotations

import os
import shutil
from typing import Any

from agent.config.defaults import env_bool, now_kst
from agent.core.autonomy import current_profile
from agent.core.metrics import collect_metrics
from agent.core.self_map import self_map_brief

ALLOWED_CODEX_WORK_SANDBOXES = {"read-only", "workspace-write"}
ALLOWED_CODEX_WORK_BACKENDS = {"codex", "native_loop"}


def _status(enabled: bool, *, available: bool = True) -> str:
    if not available:
        return "unavailable"
    return "enabled" if enabled else "disabled"


def _codex_available() -> bool:
    return shutil.which("codex") is not None


def codex_worker_enabled() -> bool:
    return env_bool("AGENT_CODEX_WORKER_ENABLED", True)


def codex_work_backend() -> str:
    value = os.getenv("AGENT_CODEX_WORK_BACKEND", "codex").strip().lower()
    return value or "codex"


def codex_work_sandbox() -> str:
    return os.getenv("AGENT_CODEX_WORK_SANDBOX", "workspace-write").strip() or "workspace-write"


def codex_worker_blockers(profile: str | None = None) -> list[str]:
    active_profile = profile or current_profile()
    blockers: list[str] = []
    if not _codex_available():
        blockers.append("codex_cli_unavailable")
    if not codex_worker_enabled():
        blockers.append("codex_worker_disabled")
    if active_profile != "full_device_lab":
        blockers.append("profile_not_full_device_lab")
    backend = codex_work_backend()
    if backend not in ALLOWED_CODEX_WORK_BACKENDS:
        blockers.append("invalid_codex_work_backend")
    sandbox = codex_work_sandbox()
    if sandbox not in ALLOWED_CODEX_WORK_SANDBOXES:
        blockers.append("invalid_codex_work_sandbox")
    return blockers


def codex_worker_available(profile: str | None = None) -> bool:
    return not codex_worker_blockers(profile)


def collect_capability_map() -> dict[str, Any]:
    profile = current_profile()
    metrics = collect_metrics()
    runtime = self_map_brief(max_age_seconds=300, refresh_if_stale=False)
    codex_available = _codex_available()
    worker_enabled = codex_worker_enabled()
    worker_blockers = codex_worker_blockers(profile)
    worker_ready = not worker_blockers
    return {
        "created_at": now_kst(),
        "profile": profile,
        "summary": "Core can converse, remember, schedule, observe the device, run policy-gated local actions, and delegate user code-change tasks to Codex when available.",
        "direct": [
            {"name": "discord_chat", "status": "enabled", "description": "Discord chat routing through Core run_talk"},
            {"name": "memory_search", "status": "enabled", "description": "FTS plus local sparse vector memory retrieval"},
            {"name": "goal_task_queue", "status": "enabled", "description": "Separate user and autonomous task queues"},
            {"name": "python_dependency_doctor", "status": "enabled", "description": "Checks venv Python requirements and can repair missing Python packages without apt/system changes"},
            {"name": "core_process_table", "status": "enabled", "description": "OS-like process view for tasks, project plans, lifecycle, progress, blockers, and next actions"},
            {"name": "control_snapshot", "status": "enabled", "description": "Redacted Discord/CLI snapshot for process table, goals, approvals, actions, memory, self-map, metrics, and safety status"},
            {"name": "project_execution_loop", "status": "enabled", "description": "User goals can be decomposed into tracked plans, steps, completion criteria, verification, and failure categories"},
            {"name": "core_pipeline_kernel", "status": "enabled", "description": "Talk decisions carry phase traces, typed decision schema, routing metadata, and recovery hints"},
            {"name": "cognitive_growth_algorithms", "status": "enabled", "description": "Curiosity, utility/novelty scoring, HTN planning, case memory, Bayesian confidence, MAP-Elites, blackboard, active-inference-lite, and stigmergy signals"},
            {"name": "cognitive_growth_pipeline", "status": "enabled", "description": "Active-inference growth snapshots can create safe autonomous queue tasks while user tasks keep priority"},
            {"name": "self_improvement_release_gate", "status": "enabled", "description": "Research-backed release plans and quality gates for isolated Core self-improvement work before main integration"},
            {"name": "event_reactor", "status": "enabled", "description": "Wake signals and agentctl reactor once/run/status move Core toward event-driven, need-driven operation"},
            {"name": "staged_project_worker", "status": "enabled", "description": "Project plans advance through planning, implementation, verification, and reporting instead of completing as an opaque single step"},
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
                "status": _status(worker_ready, available=codex_available),
                "description": "User-triggered code-change worker for repo edits, tests, and implementation reports",
                "blockers": worker_blockers,
                "backend": codex_work_backend(),
                "sandbox": codex_work_sandbox(),
            },
            {
                "name": "native_work_loop_backend",
                "status": _status(codex_work_backend() == "native_loop" and not worker_blockers, available=codex_available),
                "description": "Core-owned code work loop with worktree isolation, evidence, verification, retry, and user-task reporting",
                "blockers": worker_blockers if codex_work_backend() == "native_loop" else [],
            },
        ],
        "approval_required": [
            "sudo or systemd writes",
            "package install/remove/update",
            "external network fetch",
            "local file mutation outside the worker policy",
            "apt/system dependency installation; Python venv package repair is handled separately by dependency doctor",
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
            "Event reactor is available in parallel with existing timers; fixed timers should be weakened only after reactor stability is observed.",
            "Large project execution is user-triggered through the Codex work worker, not free-running autonomy.",
            "Self-improvement code changes are staged behind worktree isolation, tests, audit, eval, review, human approval, and rollback planning.",
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
        f"worker_backend={codex_work.get('backend', 'codex')}",
        f"eval={data.get('metrics', {}).get('last_eval_result')} / {data.get('metrics', {}).get('last_eval_score')}",
        f"vector_coverage={data.get('metrics', {}).get('memory_vector_coverage')}",
    ]
