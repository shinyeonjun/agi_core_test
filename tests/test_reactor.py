from agent.core.autonomy import set_autonomy_profile
from agent.core.database import init_db
from agent.core.goals import create_goal, get_goal
from agent.core.reactor import adaptive_sleep_seconds, reactor_once
from agent.core.task_queue import enqueue_task, list_tasks
from agent.core.wake_signals import claim_wake_signal, emit_wake_signal, list_wake_signals, wake_signal_counts


def test_wake_signal_dedupes_pending_signals():
    init_db()

    first = emit_wake_signal("memory_pressure", "test", priority=0.4, payload={"value": 1}, dedupe_key="same")
    second = emit_wake_signal("memory_pressure", "test", priority=0.9, payload={"value": 2}, dedupe_key="same")
    rows = list_wake_signals(status="pending")

    assert first == second
    assert len(rows) == 1
    assert rows[0]["priority"] == 0.9
    assert rows[0]["occurrence_count"] == 2


def test_task_enqueue_emits_wake_signal():
    init_db()
    goal_id = create_goal("User task", "do this", goal_type="user_directed", metadata={"priority_owner": "user"}, dedupe=False)

    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="User task", source="test", priority=0.9)
    signal = claim_wake_signal()

    assert signal["signal_type"] == "user_task_queued"
    assert signal["payload"]["task_id"] == task_id


def test_reactor_once_processes_user_task_first():
    init_db()
    set_autonomy_profile("full_device_lab")
    user_goal = create_goal("User note", "make a safe note", goal_type="user_directed", metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    auto_goal = create_goal("Auto note", "autonomous note", goal_type="research_note", dedupe=False)
    user_task = enqueue_task("user", goal_id=user_goal, task_kind="task_note", title="User note", source="test", priority=0.9)
    enqueue_task("autonomous", goal_id=auto_goal, task_kind="research_note", title="Auto note", source="test", priority=0.8)

    result = reactor_once()
    tasks = list_tasks(limit=10)

    assert result["action"] == "user_task_processed"
    assert result["task_id"] == user_task
    assert next(row for row in tasks if row["id"] == user_task)["status"] == "done"


def test_reactor_safe_mode_does_not_execute_autonomous_full_device_work():
    init_db()
    set_autonomy_profile("safe")
    goal_id = create_goal("Observe system", "read-only observation", goal_type="system_observation", dedupe=False)
    task_id = enqueue_task("autonomous", goal_id=goal_id, task_kind="system_observation", title="Observe system", source="test", priority=0.8)

    result = reactor_once()
    task = next(row for row in list_tasks(limit=5) if row["id"] == task_id)

    assert result["action"] == "autonomous_waiting_for_full_device_lab"
    assert task["status"] == "queued"


def test_reactor_full_device_lab_processes_autonomous_artifact_task():
    init_db()
    set_autonomy_profile("full_device_lab")
    goal_id = create_goal("Workspace report", "safe report", goal_type="workspace_experiment", dedupe=False)
    task_id = enqueue_task("autonomous", goal_id=goal_id, task_kind="workspace_experiment", title="Workspace report", source="test", priority=0.8)

    result = reactor_once()
    task = next(row for row in list_tasks(limit=5) if row["id"] == task_id)

    assert result["action"] == "autonomous_task_processed"
    assert task["status"] == "done"


def test_reactor_generates_goal_when_no_meaningful_goal_exists():
    init_db()
    set_autonomy_profile("safe")

    result = reactor_once()

    assert result["action"] in {"goal_generated", "idle_policy_ran"}
    assert wake_signal_counts()


def test_adaptive_sleep_is_shorter_for_pending_work():
    idle = adaptive_sleep_seconds({"wake_signals": {}, "task_counts": {}, "metrics": {}}, jitter=False)
    busy = adaptive_sleep_seconds({"wake_signals": {"pending": 1}, "task_counts": {}, "metrics": {}}, jitter=False)

    assert busy < idle
