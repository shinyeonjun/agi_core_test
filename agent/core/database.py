from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agent.config.defaults import db_path, ensure_runtime_dirs, env_bool, project_root

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "memory" / "schema.sql"


def _guard_pytest_live_db(path: Path) -> None:
    if "PYTEST_CURRENT_TEST" not in os.environ or env_bool("AGENT_ALLOW_LIVE_TEST_DB"):
        return
    live_path = (project_root() / "data" / "agent.db").resolve()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved == live_path:
        raise RuntimeError(
            "pytest attempted to use the live Agent Core DB. "
            "Set AGENT_CORE_DB_PATH to an isolated temp database or AGENT_ALLOW_LIVE_TEST_DB=1 explicitly."
        )


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    path = db_path()
    _guard_pytest_live_db(path)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _migrate_existing_db(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "task_queue", "approval_id", "approval_id INTEGER")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_approval ON task_queue(approval_id)")


def init_db() -> None:
    ensure_runtime_dirs()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        _migrate_existing_db(conn)
        conn.commit()


def get_schema_version() -> str | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    return str(row["value"]) if row else None
