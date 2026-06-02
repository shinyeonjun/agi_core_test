import json
from types import SimpleNamespace

from agent.cli.agentctl import main
from agent.core.capabilities import collect_capability_map
from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.goals import create_goal, list_goals
from agent.core.task_queue import enqueue_task, list_tasks
from agent.lab.planner import run_user_task


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "repo"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    (tmp_path / "repo").mkdir()
    init_db()


def test_capability_map_reaches_decision(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    decision = build_talk_decision("너 뭐 할 수 있어?")

    assert decision["capability_map"]["direct"]
    assert any(item["name"] == "codex_work_worker" for item in decision["capability_map"]["worker_mediated"])


def test_capability_cli(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["capability", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["profile"]
    assert data["worker_mediated"]


def test_code_change_user_task_runs_codex_worker(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_TIMEOUT", "9")
    (tmp_path / "repo" / ".git").mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M agent/core/example.py\n", stderr="")
        assert args[:2] == ["codex", "exec"]
        assert "--output-last-message" in args
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("수정 완료. 테스트는 생략했어.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)
    goal_id = create_goal(
        "Fix code",
        "코드 버그 수정해줘",
        goal_type="user_directed",
        status="active",
        priority=0.98,
        metadata={"priority_owner": "user", "task_kind": "code_change", "raw_user_text": "코드 버그 수정해줘"},
        dedupe=False,
    )
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="Fix code", source="test", priority=0.98)

    result = run_user_task(task_id)
    task = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)
    goal = next(row for row in list_goals(limit=5, include_archived=True) if row["id"] == goal_id)

    assert result["status"] == "codex_work_completed"
    assert result["artifact_type"] == "codex_work_report"
    assert task["status"] == "done"
    assert goal["status"] == "done"
