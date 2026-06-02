from agent.cli.agentctl import main
from agent.core.database import check_migrations, connect, get_schema_version, init_db


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def test_migrations_are_recorded_and_schema_is_current(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    status = check_migrations()

    assert get_schema_version() == "0.11.0-alpha"
    assert status["pending"] == []
    assert {row["version"] for row in status["known"]} >= {
        "0001_existing_db_repairs",
        "0002_task_queue_locks",
        "0003_operating_reviews",
    }
    with connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_queue)").fetchall()}
    assert {"locked_until", "locked_by", "idempotency_key", "not_before", "due_at", "max_attempts"} <= columns


def test_db_cli_check_and_migrate(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["db", "check"]) == 0
    assert "0002_task_queue_locks" in capsys.readouterr().out

    assert main(["db", "migrate"]) == 0
    assert '"count": 0' in capsys.readouterr().out
