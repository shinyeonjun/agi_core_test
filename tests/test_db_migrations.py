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

    assert get_schema_version() == "0.18.0-alpha"
    assert status["pending"] == []
    assert {row["version"] for row in status["known"]} >= {
        "0001_existing_db_repairs",
        "0002_task_queue_locks",
        "0003_operating_reviews",
        "0004_memory_intelligence_indexes",
        "0005_sparse_memory_vectors",
        "0006_project_execution_plans",
        "0007_process_table_runtime",
        "0008_control_room_dashboard",
        "0009_pipeline_kernel",
    }
    with connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_queue)").fetchall()}
        vector_columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_vectors)").fetchall()}
        plan_columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_execution_plans)").fetchall()}
        step_columns = {row["name"] for row in conn.execute("PRAGMA table_info(project_execution_steps)").fetchall()}
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(memories)").fetchall()}
    assert {"locked_until", "locked_by", "idempotency_key", "not_before", "due_at", "max_attempts"} <= columns
    assert {"memory_id", "vector_type", "dimensions", "content_hash", "vector_json", "nonzero_count", "updated_at"} <= vector_columns
    assert {"goal_id", "task_id", "objective", "status", "plan_json", "result_json"} <= plan_columns
    assert {"plan_id", "step_index", "completion_criteria_json", "verification_json", "failure_category"} <= step_columns
    assert {"idx_memories_updated", "idx_memories_last_used", "idx_memories_use_count"} <= indexes


def test_db_cli_check_and_migrate(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["db", "check"]) == 0
    assert "0002_task_queue_locks" in capsys.readouterr().out

    assert main(["db", "migrate"]) == 0
    assert '"count": 0' in capsys.readouterr().out
