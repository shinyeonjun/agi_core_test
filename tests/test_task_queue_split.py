import json
from datetime import datetime, timedelta, timezone

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.cli.agentctl import main
from agent.core.approvals import ApprovalStore
from agent.core.autonomy import set_autonomy_profile
from agent.core.database import connect, init_db
from agent.core.goals import create_goal, list_goals
from agent.core.pipeline import run_talk
from agent.core.task_queue import claim_task, doctor_tasks, enqueue_task, list_tasks, task_status_counts
from agent.lab.planner import run_lab_tick, run_lab_tick_if_enabled, run_user_task, sync_open_goals_to_tasks


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


def test_discord_user_task_runs_immediately_and_reports(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m1", False, False, "FastAPI 프로젝트 초안 만들어줘")

    output = "\n".join(route_discord_event(event, control_config()))
    tasks = list_tasks(limit=10, queue_type="user")
    goals = list_goals(limit=10, include_archived=True)

    assert "완료" in output
    assert "자율 스케줄러" in output
    assert tasks[0]["status"] == "done"
    assert next(goal for goal in goals if goal["goal_type"] == "user_directed")["status"] == "done"


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
        conn.execute("UPDATE task_queue SET claimed_at = ? WHERE id = ?", (old, task_id))
        conn.commit()

    result = doctor_tasks(max_age_seconds=60)
    task = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)

    assert result["stale_running"]["recovered"] == 1
    assert task["status"] == "queued"
