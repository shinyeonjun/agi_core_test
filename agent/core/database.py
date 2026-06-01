from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.config.defaults import db_path, ensure_runtime_dirs

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "memory" / "schema.sql"


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    ensure_runtime_dirs()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        conn.commit()


def get_schema_version() -> str | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
    return str(row["value"]) if row else None
