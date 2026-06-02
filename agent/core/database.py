from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agent.config.defaults import db_path, ensure_runtime_dirs, env_bool, project_root
from agent.core.db_migrations import apply_migrations, migration_status

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


def init_db() -> None:
    ensure_runtime_dirs()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        apply_migrations(conn)
        conn.commit()


def get_schema_version() -> str | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    return str(row["value"]) if row else None


def migrate_db() -> dict[str, object]:
    ensure_runtime_dirs()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        result = apply_migrations(conn)
        conn.commit()
    return result


def check_migrations() -> dict[str, object]:
    init_db()
    with connect() as conn:
        return migration_status(conn)
