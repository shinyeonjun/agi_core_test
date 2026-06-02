from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from agent.config.defaults import now_kst


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    apply: Callable[[sqlite3.Connection], None]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    if column not in _columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _ensure_memory_fts(conn: sqlite3.Connection) -> None:
    columns = _columns(conn, "memories_fts")
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


def _migration_0001_existing_db_repairs(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "task_queue", "approval_id", "approval_id INTEGER")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_approval ON task_queue(approval_id)")
    _ensure_memory_fts(conn)


def _migration_0002_task_queue_locks(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, "task_queue", "max_attempts", "max_attempts INTEGER DEFAULT 3")
    _ensure_column(conn, "task_queue", "locked_until", "locked_until TEXT")
    _ensure_column(conn, "task_queue", "locked_by", "locked_by TEXT")
    _ensure_column(conn, "task_queue", "idempotency_key", "idempotency_key TEXT")
    _ensure_column(conn, "task_queue", "not_before", "not_before TEXT")
    _ensure_column(conn, "task_queue", "due_at", "due_at TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_locked_until ON task_queue(locked_until)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_not_before ON task_queue(not_before)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_task_queue_due_at ON task_queue(due_at)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_task_queue_idempotency ON task_queue(idempotency_key) WHERE idempotency_key IS NOT NULL")


def _migration_0003_operating_reviews(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS operating_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            review_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            status TEXT DEFAULT 'open',
            payload_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operating_reviews_created ON operating_reviews(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operating_reviews_type ON operating_reviews(review_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_operating_reviews_status ON operating_reviews(status)")


def _migration_0004_memory_intelligence_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_use_count ON memories(use_count)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reflections_summary ON reflections(summary)")


def _migration_0005_sparse_memory_vectors(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_vectors (
            memory_id INTEGER NOT NULL,
            vector_type TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            nonzero_count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(memory_id, vector_type)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_type ON memory_vectors(vector_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_hash ON memory_vectors(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_vectors_updated ON memory_vectors(updated_at)")


def _migration_0006_project_execution_plans(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_execution_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            goal_id INTEGER,
            task_id INTEGER,
            source TEXT NOT NULL,
            owner TEXT NOT NULL,
            title TEXT NOT NULL,
            objective TEXT NOT NULL,
            status TEXT NOT NULL,
            priority REAL DEFAULT 0.5,
            current_step_index INTEGER DEFAULT 0,
            plan_json TEXT,
            result_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_plans_goal ON project_execution_plans(goal_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_plans_task ON project_execution_plans(task_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_plans_status ON project_execution_plans(status)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_execution_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            plan_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            task_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            completion_criteria_json TEXT,
            verification_json TEXT,
            failure_category TEXT,
            result_json TEXT,
            queued_task_id INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_steps_plan ON project_execution_steps(plan_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_project_steps_status ON project_execution_steps(status)")
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.14.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0007_process_table_runtime(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.15.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0008_control_room_dashboard(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.16.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001_existing_db_repairs", "Backfill approval task links and memory FTS schema", _migration_0001_existing_db_repairs),
    Migration("0002_task_queue_locks", "Add task queue lease, idempotency, and scheduling fields", _migration_0002_task_queue_locks),
    Migration("0003_operating_reviews", "Ensure operating intelligence review storage", _migration_0003_operating_reviews),
    Migration("0004_memory_intelligence_indexes", "Add memory and reflection indexes for compaction and retrieval", _migration_0004_memory_intelligence_indexes),
    Migration("0005_sparse_memory_vectors", "Add local sparse vector storage for memory reranking", _migration_0005_sparse_memory_vectors),
    Migration("0006_project_execution_plans", "Add project execution plans and step tracking", _migration_0006_project_execution_plans),
    Migration("0007_process_table_runtime", "Align schema for process table and staged project runtime", _migration_0007_process_table_runtime),
    Migration("0008_control_room_dashboard", "Align schema for control room dashboard runtime", _migration_0008_control_room_dashboard),
)


def ensure_schema_migrations_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
        """
    )


def apply_migrations(conn: sqlite3.Connection) -> dict[str, object]:
    ensure_schema_migrations_table(conn)
    applied_rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    applied = {str(row["version"]) for row in applied_rows}
    newly_applied: list[str] = []
    for migration in MIGRATIONS:
        if migration.version in applied:
            continue
        migration.apply(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (migration.version, now_kst(), migration.description),
        )
        newly_applied.append(migration.version)
    return {
        "applied": newly_applied,
        "known": [migration.version for migration in MIGRATIONS],
        "count": len(newly_applied),
    }


def migration_status(conn: sqlite3.Connection) -> dict[str, object]:
    ensure_schema_migrations_table(conn)
    rows = conn.execute("SELECT version, applied_at, description FROM schema_migrations ORDER BY version").fetchall()
    applied = {str(row["version"]): dict(row) for row in rows}
    return {
        "known": [
            {
                "version": migration.version,
                "description": migration.description,
                "applied": migration.version in applied,
                "applied_at": applied.get(migration.version, {}).get("applied_at"),
            }
            for migration in MIGRATIONS
        ],
        "pending": [migration.version for migration in MIGRATIONS if migration.version not in applied],
    }
