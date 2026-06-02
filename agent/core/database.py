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


def _ensure_memory_fts(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(memories_fts)").fetchall()}
    if columns and "tags_json" not in columns:
        conn.execute("DROP TRIGGER IF EXISTS memories_ai")
        conn.execute("DROP TRIGGER IF EXISTS memories_ad")
        conn.execute("DROP TRIGGER IF EXISTS memories_au")
        conn.execute("DROP TABLE IF EXISTS memories_fts")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            title,
            content,
            tags_json,
            content='memories',
            content_rowid='id'
        )
        """
    )
    conn.executescript(
        """
        DROP TRIGGER IF EXISTS memories_ai;
        DROP TRIGGER IF EXISTS memories_ad;
        DROP TRIGGER IF EXISTS memories_au;
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(rowid, title, content, tags_json)
          VALUES (new.id, new.title, new.content, COALESCE(new.tags_json, ''));
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags_json)
          VALUES('delete', old.id, old.title, old.content, COALESCE(old.tags_json, ''));
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags_json)
          VALUES('delete', old.id, old.title, old.content, COALESCE(old.tags_json, ''));
          INSERT INTO memories_fts(rowid, title, content, tags_json)
          VALUES (new.id, new.title, new.content, COALESCE(new.tags_json, ''));
        END;
        """
    )
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")


def _migrate_existing_db(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "task_queue", "approval_id", "approval_id INTEGER")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_approval ON task_queue(approval_id)")
    _ensure_memory_fts(conn)


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
