from __future__ import annotations

import random
import time
from typing import Any

from agent.config.defaults import env_int, now_kst
from agent.core.autonomy import current_profile
from agent.core.database import init_db
from agent.core.events import log_event
from agent.core.goal_generator import generate_goal_candidates, meaningful_open_goals
from agent.core.metrics import collect_metrics
from agent.core.task_queue import doctor_tasks, list_tasks, task_status_counts
from agent.core.wake_signals import claim_wake_signal, complete_wake_signal, emit_wake_signal, list_wake_signals, wake_signal_counts
from agent.lab.planner import run_lab_tick, run_user_task, sync_open_goals_to_tasks
from agent.scheduler.idle_policy import run_idle_policy


def adaptive_sleep_seconds(status: dict[str, Any] | None = None, *, jitter: bool = True) -> int:
    status = status or reactor_status()
    pending = int((status.get("wake_signals") or {}).get("pending") or 0)
    user_waiting = int((status.get("task_counts") or {}).get("user:queued") or 0)
    autonomous_waiting = int((status.get("task_counts") or {}).get("autonomous:queued") or 0)
    pending_approvals = int((status.get("metrics") or {}).get("pending_approvals_count") or 0)
    if pending or user_waiting:
        base = 2
    elif pending_approvals:
        base = 8
    elif autonomous_waiting:
        base = 20
    else:
        base = 300
    if jitter:
        base = int(max(1, base * random.uniform(0.7, 1.4)))
    return max(1, min(1800, base))


def reactor_status() -> dict[str, Any]:
    init_db()
    metrics = collect_metrics()
    return {
        "created_at": now_kst(),
        "profile": current_profile(),
        "wake_signals": wake_signal_counts(),
        "recent_signals": list_wake_signals(limit=5),
        "task_counts": task_status_counts(),
        "metrics": {
            "pending_approvals_count": metrics.get("pending_approvals_count"),
            "meaningful_open_goals_count": metrics.get("meaningful_open_goals_count"),
            "action_execution_success_rate_24h": metrics.get("action_execution_success_rate_24h"),
            "memory_vector_coverage": metrics.get("memory_vector_coverage"),
        },
    }


def _first_task(queue_type: str) -> dict[str, Any] | None:
    rows = list_tasks(limit=1, status="queued", queue_type=queue_type)
    return rows[0] if rows else None


def _process_user_task(signal: dict[str, Any] | None) -> dict[str, Any] | None:
    task = _first_task("user")
    if not task:
        return None
    result = run_user_task(int(task["id"]))
    return {"action": "user_task_processed", "task_id": int(task["id"]), "signal_id": signal.get("id") if signal else None, "result": result}


def _process_autonomous_task(signal: dict[str, Any] | None) -> dict[str, Any] | None:
    task = _first_task("autonomous")
    if not task:
        return None
    profile = current_profile()
    if profile != "full_device_lab":
        return {
            "action": "autonomous_waiting_for_full_device_lab",
            "task_id": int(task["id"]),
            "profile": profile,
            "signal_id": signal.get("id") if signal else None,
        }
    result = run_lab_tick()
    return {"action": "autonomous_task_processed", "task_id": int(task["id"]), "signal_id": signal.get("id") if signal else None, "result": result}


def _generate_or_idle(signal: dict[str, Any] | None) -> dict[str, Any]:
    open_goals = meaningful_open_goals(limit=1)
    if not open_goals:
        generated = generate_goal_candidates(dry_run=False)
        if generated.get("created_goal_id"):
            return {"action": "goal_generated", "signal_id": signal.get("id") if signal else None, "result": generated}
    idle = run_idle_policy()
    return {"action": "idle_policy_ran", "signal_id": signal.get("id") if signal else None, "result": idle}


def reactor_once(*, dry_run: bool = False) -> dict[str, Any]:
    init_db()
    if dry_run:
        status = reactor_status()
        return {"action": "dry_run", "status": status, "next_sleep_seconds": adaptive_sleep_seconds(status, jitter=False)}
    doctor = doctor_tasks()
    signal = claim_wake_signal()
    if signal is None:
        emit_wake_signal("watchdog", "reactor", priority=0.2, payload={"reason": "no_pending_signal"}, dedupe_key="watchdog:reactor")
        signal = claim_wake_signal()
    sync = sync_open_goals_to_tasks()
    result = _process_user_task(signal)
    if result is None:
        result = _process_autonomous_task(signal)
    if result is None:
        result = _generate_or_idle(signal)
    result = {**result, "doctor": doctor, "sync": sync}
    if signal:
        complete_wake_signal(int(signal["id"]), status="done", result=result)
    log_event("reactor", "reactor_cycle_completed", str(result.get("action")), result, 0.68)
    return result


def reactor_run(*, cycles: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    max_cycles = cycles if cycles is not None else env_int("AGENT_REACTOR_MAX_CYCLES", 0)
    completed = 0
    last_result: dict[str, Any] | None = None
    while True:
        last_result = reactor_once(dry_run=dry_run)
        completed += 1
        if max_cycles and completed >= max_cycles:
            break
        sleep_seconds = adaptive_sleep_seconds(jitter=True)
        log_event("reactor", "reactor_sleep", str(sleep_seconds), {"seconds": sleep_seconds, "cycle": completed}, 0.35)
        time.sleep(sleep_seconds)
    return {"status": "stopped", "cycles": completed, "last_result": last_result}
