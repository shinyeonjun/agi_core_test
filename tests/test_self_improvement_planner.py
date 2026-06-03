import json

from agent.cli.agentctl import main
from agent.core.database import init_db
from agent.core.goals import get_goal
from agent.core.self_improvement_planner import enqueue_self_improvement_tickets, generate_self_improvement_tickets, ticket_to_worker_prompt
from agent.core.task_queue import list_tasks


def test_self_improvement_ticket_has_worker_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    init_db()

    tickets = generate_self_improvement_tickets(limit=2)
    ticket = tickets[0]
    prompt = ticket_to_worker_prompt(ticket)

    assert ticket["key"]
    assert ticket["title"]
    assert ticket["problem"]
    assert ticket["scope"]
    assert "python -m pytest -q" in ticket["verification_commands"]
    assert "Use git worktree/native loop only" in prompt
    assert ".env" in prompt


def test_self_improvement_enqueue_creates_autonomous_code_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    init_db()

    result = enqueue_self_improvement_tickets(limit=1)
    created = result["created"][0]
    task = next(item for item in list_tasks(limit=5, queue_type="autonomous") if item["id"] == created["task_id"])
    goal = get_goal(created["goal_id"])
    metadata = json.loads(goal["metadata_json"])

    assert result["count"] == 1
    assert task["task_kind"] == "code_change"
    assert task["status"] == "queued"
    assert metadata["task_kind"] == "code_change"
    assert metadata["requires_native_loop"] is True
    assert metadata["self_improvement_ticket"]["key"] == created["ticket"]["key"]


def test_self_improve_cli_tickets_and_enqueue(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    init_db()

    assert main(["self-improve", "tickets", "--limit", "1"]) == 0
    tickets = json.loads(capsys.readouterr().out)
    assert len(tickets["items"]) == 1

    assert main(["self-improve", "enqueue", "--limit", "1"]) == 0
    enqueued = json.loads(capsys.readouterr().out)
    assert enqueued["count"] == 1
