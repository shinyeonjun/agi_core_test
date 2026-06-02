from agent.config.defaults import db_path, env_path, state_path, workspace_root
from agent.core.database import connect, init_db


def test_pytest_runtime_uses_isolated_paths(tmp_path):
    init_db()

    assert db_path() == tmp_path / "agent.db"
    assert state_path() == tmp_path / "state.json"
    assert env_path() == tmp_path / ".env"
    assert workspace_root() == tmp_path / "workspace"


def test_init_db_migrates_existing_task_queue_without_approval_id():
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE task_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                queue_type TEXT NOT NULL,
                status TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                goal_id INTEGER,
                task_kind TEXT NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                payload_json TEXT,
                attempts INTEGER DEFAULT 0,
                claimed_at TEXT,
                completed_at TEXT,
                result_json TEXT
            )
            """
        )
        conn.commit()

    init_db()

    with connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_queue)").fetchall()}
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(task_queue)").fetchall()}
    assert "approval_id" in columns
    assert "idx_task_queue_approval" in indexes
