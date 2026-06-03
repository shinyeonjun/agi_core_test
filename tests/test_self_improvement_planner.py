import json

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.cli.agentctl import main
from agent.core.database import init_db
from agent.core.goals import get_goal
from agent.core.self_improvement_planner import enqueue_self_improvement_tickets, enqueue_user_self_improvement_request, generate_self_improvement_tickets, ticket_to_worker_prompt
from agent.core.task_queue import list_tasks
from agent.language.engine import interpret_user_message


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def control_config():
    return DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10"},
        chat_channel_id="10",
        approval_channel_id="20",
        user_cooldown_seconds=0,
    )


def test_self_improvement_ticket_has_worker_contract(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

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
    setup_isolated(monkeypatch, tmp_path)

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
    setup_isolated(monkeypatch, tmp_path)

    assert main(["self-improve", "tickets", "--limit", "1"]) == 0
    tickets = json.loads(capsys.readouterr().out)
    assert len(tickets["items"]) == 1

    assert main(["self-improve", "enqueue", "--limit", "1"]) == 0
    enqueued = json.loads(capsys.readouterr().out)
    assert enqueued["count"] == 1


def test_user_self_improvement_request_interprets_as_action(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    interpretation = interpret_user_message("Core 자가개선 작업 한번 진행해봐", {"surface": "talk"})

    assert interpretation["intent"] == "self_improvement_request"
    assert interpretation["target"] == "self_improvement"
    assert interpretation["execution"]["requires_action"] is True
    assert interpretation["execution"]["suggested_queue_type"] == "self_improvement_code"


def test_discord_self_improvement_request_creates_user_code_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m-self", False, False, "Core 자가개선 목표 잡아서 개발 진행해봐")

    output = "\n".join(route_discord_event(event, control_config()))
    tasks = list_tasks(limit=5, queue_type="user")
    task = tasks[0]
    goal = get_goal(task["goal_id"])
    metadata = json.loads(goal["metadata_json"])

    assert "자가개선 작업으로 잡았어" in output
    assert "main 반영은 승인 전에는 안 해" in output
    assert task["task_kind"] == "code_change"
    assert task["source"] == "discord_self_improvement"
    assert task["payload"]["requires_native_loop"] is True
    assert goal["goal_type"] == "self_improvement_proposal"
    assert metadata["priority_owner"] == "user"
    assert metadata["state_machine"]["main_apply"] == "approval_required"


def test_discord_self_improvement_request_dedupes_same_ticket(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    first = enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)["created"][0]
    second = enqueue_user_self_improvement_request("Core 자가개선 다시 진행", source_event_id=456, limit=1)["created"][0]
    tasks = list_tasks(limit=10, queue_type="user")

    assert second["task_id"] == first["task_id"]
    assert len(tasks) == 1


def test_user_self_improvement_status_reports_queue_phase(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)

    assert main(["self-improve", "status", "--limit", "3"]) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["counts"]["queued"] == 1
    assert data["items"][0]["phase"] == "queued"
    assert "작업공간" in data["items"][0]["next_step"]
