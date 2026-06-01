from __future__ import annotations

import sqlite3
from pathlib import Path

from agent.config.defaults import db_path, ensure_runtime_dirs

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "memory" / "schema.sql"


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    ensure_runtime_dirs()
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with connect() as conn:
        conn.executescript(schema)
        conn.commit()
