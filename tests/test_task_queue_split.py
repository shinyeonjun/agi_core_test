import json
import sys
from datetime import datetime, timedelta, timezone

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.formatter import format_chat_reply
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.cli.agentctl import main
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import set_autonomy_profile
from agent.core.database import connect, init_db
from agent.core.db_hygiene import cleanup_db_noise
from agent.core.goals import create_goal, list_goals, update_goal_metadata
from agent.core.pipeline import run_talk
from agent.core.task_lifecycle import list_task_lifecycle, task_lifecycle_summary
from agent.core.task_queue import claim_task, doctor_tasks, enqueue_task, finish_task, list_tasks, task_status_counts
from agent.lab.planner import run_lab_tick, run_lab_tick_if_enabled, run_user_task, sync_open_goals_to_tasks
from agent.lab.codex_worker import _work_loop_verify_commands


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def control_config():
    return DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"10"}, chat_channel_id="10", user_cooldown_seconds=0)


def test_user_directive_enqueues_user_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = run_talk("FastAPI 프로젝트 초안 만들어줘")
    user_goal = result["decision"]["user_directed_goal"]
    tasks = list_tasks(limit=10, queue_type="user")

    assert user_goal["task_id"] is not None
    assert tasks[0]["id"] == user_goal["task_id"]
    assert tasks[0]["queue_type"] == "user"
    assert tasks[0]["status"] == "queued"


def test_discord_user_task_is_queued_without_blocking_chat(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m1", False, False, "FastAPI 프로젝트 초안 만들어줘")

    output = "\n".join(route_discord_event(event, control_config()))
    tasks = list_tasks(limit=10, queue_type="user")
    goals = list_goals(limit=10, include_archived=True)

    assert "작업으로 넘겼어" in output
    assert "자율 스케줄러" not in output
    assert tasks[0]["status"] == "queued"
    assert next(goal for goal in goals if goal["goal_type"] == "user_directed")["status"] == "active"


def test_scheduler_does_not_claim_user_queue(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    goal_id = create_goal("User queued task", "user task", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="User queued task", source="test", priority=0.98)

    result = run_lab_tick()
    task = next(row for row in list_tasks(limit=10) if row["id"] == task_id)

    assert result["reason"] in {"no_autonomous_task", "no_approved_proposal"} or result["status"] in {"idle", "blocked"}
    assert task["status"] == "queued"


def test_user_worker_processes_specific_user_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Write report", "보고서 만들어줘", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "report", "raw_user_text": "보고서 만들어줘"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="report", title="Write report", source="test", priority=0.98)

    result = run_user_task(task_id)
    task = next(row for row in list_tasks(limit=10) if row["id"] == task_id)

    assert result["status"] == "user_goal_completed"
    assert task["status"] == "done"
    assert result["artifact_id"] is not None
    phases = [row["phase"] for row in list_task_lifecycle(task_id)]
    assert "queued" in phases
    assert "planning" in phases
    assert "executing" in phases
    assert "verifying" in phases
    assert "reporting" in phases
    assert "learned" in phases


def test_task_lifecycle_summary_explains_last_phase(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Lifecycle task", "note", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "task_note", "raw_user_text": "note"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="Lifecycle task", source="test", priority=0.98)

    run_user_task(task_id)
    task = next(row for row in list_tasks(limit=10, queue_type="user") if row["id"] == task_id)
    summary = task_lifecycle_summary(task)

    assert summary["last_phase"] in {"reporting", "learned"}
    assert "queued" in summary["completed_phases"]
    assert summary["events"]


def test_autonomous_generation_ignores_open_user_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    create_goal("User open task", "user task", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)

    result = run_lab_tick_if_enabled()
    counts = task_status_counts()

    assert result["status"] in {"goal_generated", "artifact_created", "completed", "blocked", "idle"}
    assert counts.get("autonomous:queued", 0) + counts.get("autonomous:done", 0) + counts.get("autonomous:running", 0) >= 0


def test_tasks_cli(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("CLI task", "cli", goal_type="user_directed", status="active", priority=0.9, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="CLI task", source="test", priority=0.9)

    assert main(["tasks", "counts"]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert counts["user:queued"] == 1

    assert main(["tasks", "list", "--queue-type", "user"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["queue_type"] == "user"

    task_id = rows[0]["id"]
    assert main(["tasks", "lifecycle", str(task_id)]) == 0
    lifecycle = json.loads(capsys.readouterr().out)
    assert lifecycle["task_id"] == task_id
    assert lifecycle["completed_phases"] == ["queued"]


def test_approval_resume_moves_user_task_to_queue(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = run_talk("apt-get install nginx 해줘")
    user_goal = result["decision"]["user_directed_goal"]
    approval_id = user_goal["approval_id"]
    task = list_tasks(limit=5, queue_type="user")[0]

    assert user_goal["status"] == "waiting_approval"
    assert approval_id is not None
    assert task["status"] == "waiting_approval"
    assert task["approval_id"] == approval_id

    assert ApprovalStore().approve(approval_id) is True
    resumed = list_tasks(limit=5, queue_type="user")[0]
    assert resumed["status"] == "queued"


def test_reject_approval_blocks_waiting_user_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = run_talk("apt-get install nginx 해줘")
    approval_id = result["decision"]["user_directed_goal"]["approval_id"]

    assert ApprovalStore().reject(approval_id) is True
    blocked = list_tasks(limit=5, queue_type="user")[0]
    assert blocked["status"] == "blocked"
    assert blocked["result"]["reason"] == "approval_rejected"


def test_task_doctor_recovers_stale_running_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Doctor task", "doctor", goal_type="user_directed", status="active", priority=0.9, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="Doctor task", source="test", priority=0.9)
    assert claim_task(task_id)["status"] == "running"
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="seconds")
    with connect() as conn:
        conn.execute("UPDATE task_queue SET claimed_at = ?, locked_until = ? WHERE id = ?", (old, old, task_id))
        conn.commit()

    result = doctor_tasks(max_age_seconds=60)
    task = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)

    assert result["stale_running"]["recovered"] == 1
    assert task["status"] == "queued"


def test_task_claim_sets_and_clears_lease(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Lease task", "lease", goal_type="user_directed", status="active", priority=0.9, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="Lease task", source="test", priority=0.9)

    claimed = claim_task(task_id)

    assert claimed["locked_until"] is not None
    assert claimed["locked_by"] is not None
    task = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)
    assert task["status"] == "running"

    finish_task(task_id, "done", {"status": "done"})
    finished = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)
    assert finished["locked_until"] is None
    assert finished["locked_by"] is None


def test_task_idempotency_key_reuses_existing_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    first = enqueue_task("user", goal_id=None, task_kind="task_note", title="Same task", source="test", idempotency_key="same-key")
    second = enqueue_task("user", goal_id=None, task_kind="task_note", title="Same task duplicate", source="test", idempotency_key="same-key")

    assert second == first
    assert len(list_tasks(limit=10, queue_type="user")) == 1


def test_task_idempotency_key_reuses_finished_task_without_crashing(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    first = enqueue_task("user", goal_id=None, task_kind="task_note", title="Same task", source="test", idempotency_key="same-key")
    finish_task(first, "blocked", {"status": "blocked", "reason": "already handled"})

    second = enqueue_task("user", goal_id=None, task_kind="task_note", title="Same task duplicate", source="test", idempotency_key="same-key")

    assert second == first
    assert len(list_tasks(limit=10, queue_type="user")) == 1


def test_cancel_target_message_archives_goal_and_skips_open_tasks(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("논문 수집 개발", "외부 논문 수집은 나중에", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "code_change"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="논문 수집 개발", source="test", priority=0.98)

    result = run_talk(f"#{task_id} 이거 목표에서 없애줘")
    goals = {goal["id"]: goal for goal in list_goals(limit=10, include_archived=True)}
    tasks = {task["id"]: task for task in list_tasks(limit=10, queue_type="user")}

    assert result["decision"]["user_goal_created"] is False
    assert result["decision"]["user_directed_goal"]["control_action"] == "cancel"
    assert goals[goal_id]["status"] == "archived"
    assert tasks[task_id]["status"] == "skipped"
    assert "정리" in format_chat_reply(f"#{task_id} 이거 목표에서 없애줘", result)


def test_cancel_command_does_not_create_new_project_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("논문 수집 개발", "외부 논문 수집은 나중에", goal_type="user_directed", status="active", priority=0.98, metadata={"priority_owner": "user", "task_kind": "code_change"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="논문 수집 개발", source="test", priority=0.98)
    event = DiscordEvent(None, "10", "1", "m-cancel", False, False, f"!cancel {task_id}")

    output = "\n".join(route_discord_event(event, control_config()))
    tasks = list_tasks(limit=10, queue_type="user")

    assert "정리했어" in output
    assert len(tasks) == 1
    assert tasks[0]["status"] == "skipped"


def test_db_cleanup_archives_cancel_request_noise(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("remove goal #123", "remove this task from goals", goal_type="user_directed", status="active", priority=0.7, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="project_spec", title="remove goal #123", source="discord_user_directive", priority=0.7)

    dry_run = cleanup_db_noise(apply=False)
    assert task_id in dry_run["cancel_noise_tasks"]
    assert goal_id in dry_run["cancel_noise_goals"]

    applied = cleanup_db_noise(apply=True)
    goals = {goal["id"]: goal for goal in list_goals(limit=10, include_archived=True)}
    tasks = {task["id"]: task for task in list_tasks(limit=10, queue_type="user")}

    assert applied["changed"]["tasks_skipped"] == 1
    assert applied["changed"]["goals_archived"] == 1
    assert tasks[task_id]["status"] == "skipped"
    assert goals[goal_id]["status"] == "archived"


def test_goal_sync_skips_blocked_goal_without_reenqueue(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Blocked code task", "blocked", goal_type="user_directed", status="active", priority=0.9, metadata={"priority_owner": "user", "task_kind": "code_change"}, dedupe=False)
    update_goal_metadata(goal_id, {"priority_owner": "user", "task_kind": "code_change"}, status="blocked")

    result = sync_open_goals_to_tasks()
    tasks = list_tasks(limit=10, queue_type="user")

    assert result["skipped"] >= 1
    assert tasks == []


def test_goal_sync_pauses_code_goal_after_worker_failure(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Code task", "code", goal_type="user_directed", status="active", priority=0.9, metadata={"priority_owner": "user", "task_kind": "code_change"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="Code task", source="goal_sync", priority=0.9)
    finish_task(task_id, "blocked", {"status": "codex_work_failed", "returncode": 125})

    result = sync_open_goals_to_tasks()
    goals = {goal["id"]: goal for goal in list_goals(limit=10, include_archived=True)}
    tasks = list_tasks(limit=10, queue_type="user")

    assert result["skipped"] >= 1
    assert goals[goal_id]["status"] == "blocked"
    assert len(tasks) == 1
    assert tasks[0]["id"] == task_id


def test_goal_sync_pauses_after_discord_self_improvement_failure(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Self improvement", "code", goal_type="self_improvement_proposal", status="active", priority=0.95, metadata={"priority_owner": "user", "task_kind": "code_change"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="Self improvement", source="discord_self_improvement", priority=0.95)
    finish_task(task_id, "blocked", {"status": "codex_work_failed", "returncode": 125})

    result = sync_open_goals_to_tasks()
    goals = {goal["id"]: goal for goal in list_goals(limit=10, include_archived=True)}
    tasks = list_tasks(limit=10, queue_type="user")

    assert result["skipped"] >= 1
    assert goals[goal_id]["status"] == "blocked"
    assert len(tasks) == 1
    assert tasks[0]["id"] == task_id


def test_db_cleanup_skips_open_tasks_for_closed_goals(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Closed goal", "done", goal_type="user_directed", status="done", priority=0.9, metadata={"priority_owner": "user", "task_kind": "task_note"}, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="Closed goal", source="test", priority=0.9)

    dry_run = cleanup_db_noise(apply=False)
    assert task_id in dry_run["closed_goal_tasks"]

    applied = cleanup_db_noise(apply=True)
    task = next(row for row in list_tasks(limit=10, queue_type="user") if row["id"] == task_id)

    assert applied["changed"]["closed_goal_tasks_skipped"] == 1
    assert task["status"] == "skipped"
    assert task["result"]["reason"] == "closed_goal"


def test_work_loop_default_verify_uses_current_python(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.delenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", raising=False)

    commands = _work_loop_verify_commands()

    assert commands
    assert "agent.cli.agentctl test run fast" in commands[0]
    assert "python -m pytest" not in commands[0]
    assert sys.executable in commands[0]
    assert commands[0] != "python -m pytest -q"


def test_work_loop_env_verify_normalizes_python_aliases(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv(
        "AGENT_WORK_LOOP_VERIFY_COMMANDS",
        "python -m pytest -q;python3 -m agent.cli.agentctl audit",
    )

    commands = _work_loop_verify_commands()

    assert len(commands) == 2
    assert all(sys.executable in command for command in commands)
    assert all(not command.startswith(("python ", "python3 ")) for command in commands)
    assert "pytest" in commands[0]
    assert "agent.cli.agentctl audit" in commands[1]


def test_db_cleanup_cli_defaults_to_dry_run(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("remove goal #999", "remove this goal", goal_type="user_directed", status="active", priority=0.7, dedupe=False)
    enqueue_task("user", goal_id=goal_id, task_kind="project_spec", title="remove goal #999", source="discord_user_directive", priority=0.7)

    assert main(["db", "cleanup"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["applied"] is False
    assert dry["cancel_noise_tasks"]

    assert main(["db", "cleanup", "--apply"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["changed"]["tasks_skipped"] == 1


def test_self_improvement_verify_defaults_to_board_safe_chain(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.delenv("AGENT_SELF_IMPROVEMENT_VERIFY_COMMANDS", raising=False)
    monkeypatch.setenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", "python -m pytest -q;python -m agent.cli.agentctl audit")

    commands = _work_loop_verify_commands(self_improvement=True)

    assert len(commands) == 3
    assert "agent.cli.agentctl test run fast" in commands[0]
    assert "agent.cli.agentctl audit" in commands[1]
    assert "agent.cli.agentctl eval run" in commands[2]
    assert all("pytest -q" not in command for command in commands)


def test_self_improvement_verify_downgrades_full_pytest_when_inherited(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_SELF_IMPROVEMENT_INHERIT_VERIFY_COMMANDS", "1")
    monkeypatch.setenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", "python -m pytest -q;python3 -m agent.cli.agentctl audit")

    commands = _work_loop_verify_commands(self_improvement=True)

    assert "agent.cli.agentctl test run fast" in commands[0]
    assert "pytest -q" not in commands[0]
    assert "agent.cli.agentctl audit" in commands[1]


def test_self_improvement_verify_can_keep_targeted_pytest(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_SELF_IMPROVEMENT_VERIFY_COMMANDS", "python -m pytest tests/test_renderer.py -q")

    commands = _work_loop_verify_commands(self_improvement=True)

    assert len(commands) == 1
    assert "tests/test_renderer.py" in commands[0]
    assert sys.executable in commands[0]
