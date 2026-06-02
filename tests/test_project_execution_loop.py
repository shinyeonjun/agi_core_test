import json

from agent.cli.agentctl import main
from agent.core.database import init_db
from agent.core.pipeline import run_talk
from agent.core.process_table import get_process, list_processes, process_snapshot
from agent.core.project_execution import classify_failure_reason, get_project_plan, list_project_plans
from agent.core.task_queue import list_tasks
from agent.lab.planner import run_user_task


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def test_user_project_request_creates_execution_plan(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = run_talk("FastAPI 프로젝트 구조를 설계하고 초안까지 만들어줘")
    user_goal = result["decision"]["user_directed_goal"]
    plans = list_project_plans(limit=5)

    assert user_goal["project_plan_id"] is not None
    assert plans[0]["id"] == user_goal["project_plan_id"]
    assert plans[0]["status"] == "planned"
    plan = get_project_plan(user_goal["project_plan_id"])
    assert plan is not None
    assert len(plan["steps"]) == 4
    assert all(step["completion_criteria"] for step in plan["steps"])
    processes = list_processes(limit=5)
    assert processes[0]["pid"] == f"task:{user_goal['task_id']}"
    assert processes[0]["plan"]["plan_id"] == user_goal["project_plan_id"]
    assert processes[0]["plan"]["progress"]["total"] == 4


def test_user_task_completion_marks_project_plan_done(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = run_talk("FastAPI 프로젝트 구조를 설계하고 초안까지 만들어줘")
    task_id = result["decision"]["user_directed_goal"]["task_id"]
    task_result = run_user_task(task_id)
    plan = get_project_plan(task_result["project_plan"]["id"])
    task = list_tasks(limit=5, queue_type="user")[0]

    assert task_result["status"] == "user_goal_completed"
    assert task_result["project_plan"]["status"] == "done"
    assert task_result["project_plan"]["steps_done"] == 4
    assert plan["status"] == "done"
    assert all(step["status"] == "done" for step in plan["steps"])
    assert task["status"] == "done"
    process = get_process(f"task:{task_id}")
    assert process is not None
    assert process["state"] == "done"
    assert process["plan"]["progress"]["ratio"] == 1.0


def test_project_cli_lists_and_shows_plans(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    result = run_talk("FastAPI 프로젝트 구조를 설계하고 초안까지 만들어줘")
    plan_id = result["decision"]["user_directed_goal"]["project_plan_id"]

    assert main(["project", "plans"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["id"] == plan_id

    assert main(["project", "show", str(plan_id)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["id"] == plan_id
    assert len(plan["steps"]) == 4


def test_process_cli_lists_project_tasks(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    result = run_talk("FastAPI 프로젝트 구조를 설계하고 초안까지 만들어줘")
    task_id = result["decision"]["user_directed_goal"]["task_id"]

    assert main(["process", "snapshot", "--limit", "5"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["items"]
    assert snapshot["items"][0]["pid"] == f"task:{task_id}"
    assert snapshot["items"][0]["next"]

    assert main(["process", "show", f"task:{task_id}"]) == 0
    item = json.loads(capsys.readouterr().out)
    assert item["task_id"] == task_id
    assert item["plan"]["progress"]["total"] == 4


def test_failure_classifier_labels_common_worker_failures():
    assert classify_failure_reason({"reason": "profile_not_full_device_lab"}) == "profile_block"
    assert classify_failure_reason({"stderr": "ModuleNotFoundError: No module named agent"}) == "environment_issue"
    assert classify_failure_reason({"returncode": 124, "stderr": "timeout"}) == "timeout"
