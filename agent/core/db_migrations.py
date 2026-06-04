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


def _migration_0009_pipeline_kernel(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.17.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0010_cognitive_growth_algorithms(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS blackboard_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source TEXT NOT NULL,
            topic TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            confidence REAL DEFAULT 0.5,
            tags_json TEXT,
            metadata_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_status ON blackboard_items(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_topic ON blackboard_items(topic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blackboard_confidence ON blackboard_items(confidence)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_map_elites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            archive_name TEXT NOT NULL,
            cell_key TEXT NOT NULL,
            axes_json TEXT NOT NULL,
            candidate_json TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'active',
            UNIQUE(archive_name, cell_key)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_map_archive ON cognitive_map_elites(archive_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_map_score ON cognitive_map_elites(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_map_status ON cognitive_map_elites(status)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stigmergy_markers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            marker_type TEXT NOT NULL,
            target_type TEXT NOT NULL,
            target_id TEXT,
            intensity REAL DEFAULT 0.5,
            decay_rate REAL DEFAULT 0.05,
            status TEXT NOT NULL DEFAULT 'active',
            reason TEXT NOT NULL,
            metadata_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stigmergy_status ON stigmergy_markers(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stigmergy_type ON stigmergy_markers(marker_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stigmergy_intensity ON stigmergy_markers(intensity)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            snapshot_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_snapshots_created ON cognitive_snapshots(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_snapshots_type ON cognitive_snapshots(snapshot_type)")
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.18.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0011_wake_signals_reactor(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS wake_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            source TEXT NOT NULL,
            priority REAL DEFAULT 0.5,
            status TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT,
            dedupe_key TEXT,
            occurrence_count INTEGER DEFAULT 1,
            not_before TEXT,
            expires_at TEXT,
            claimed_at TEXT,
            completed_at TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_signals_status ON wake_signals(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_signals_type ON wake_signals(signal_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_signals_priority ON wake_signals(priority)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wake_signals_not_before ON wake_signals(not_before)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_wake_signals_pending_dedupe ON wake_signals(dedupe_key) WHERE status = 'pending' AND dedupe_key IS NOT NULL")
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.19.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0012_discord_message_dedupe(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        DELETE FROM discord_events
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM discord_events
            GROUP BY channel_id, user_id, message_id
        )
        """
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_discord_events_message_unique ON discord_events(channel_id, user_id, message_id)")


def _migration_0013_cognitive_graph_layer(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            node_type TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.7,
            freshness REAL DEFAULT 0.5,
            risk REAL DEFAULT 0.0,
            success_rate REAL,
            revisit_score REAL DEFAULT 0.5,
            metadata_json TEXT,
            archived INTEGER DEFAULT 0,
            UNIQUE(node_type, source_type, source_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_type ON cognitive_nodes(node_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_source ON cognitive_nodes(source_type, source_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_revisit ON cognitive_nodes(revisit_score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_updated ON cognitive_nodes(updated_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_archived ON cognitive_nodes(archived)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            from_node_id INTEGER NOT NULL,
            to_node_id INTEGER NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.7,
            evidence_json TEXT,
            archived INTEGER DEFAULT 0,
            UNIQUE(from_node_id, to_node_id, edge_type)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_edges_from ON cognitive_edges(from_node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_edges_to ON cognitive_edges(to_node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_edges_type ON cognitive_edges(edge_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_edges_weight ON cognitive_edges(weight)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_activations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            node_id INTEGER NOT NULL,
            context TEXT,
            reason TEXT,
            score REAL NOT NULL,
            components_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_activations_created ON cognitive_activations(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_activations_node ON cognitive_activations(node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_activations_score ON cognitive_activations(score)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cognitive_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            trace_type TEXT NOT NULL,
            decision_id TEXT,
            root_node_id INTEGER,
            summary TEXT,
            trace_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_traces_created ON cognitive_traces(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_traces_type ON cognitive_traces(trace_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_traces_decision ON cognitive_traces(decision_id)")
    conn.execute(
        """
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.20.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )


def _migration_0014_advanced_learning_loops(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_rollups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            level INTEGER NOT NULL,
            cluster_key TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_memory_ids_json TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            metadata_json TEXT,
            archived INTEGER DEFAULT 0,
            UNIQUE(level, cluster_key)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rollups_level ON memory_rollups(level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rollups_score ON memory_rollups(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rollups_archived ON memory_rollups(archived)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graph_community_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            community_key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            node_ids_json TEXT NOT NULL,
            edge_ids_json TEXT NOT NULL,
            score REAL DEFAULT 0.0,
            metadata_json TEXT,
            archived INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_summaries_score ON graph_community_summaries(score)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_summaries_archived ON graph_community_summaries(archived)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS failure_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            task_id INTEGER,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            summary TEXT,
            evidence_json TEXT,
            strategy_json TEXT,
            resolved INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            UNIQUE(source_type, source_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_cases_category ON failure_cases(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_cases_task ON failure_cases(task_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_failure_cases_resolved ON failure_cases(resolved)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS circuit_breakers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            breaker_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            category TEXT NOT NULL,
            failure_count INTEGER DEFAULT 0,
            threshold_count INTEGER DEFAULT 3,
            cooldown_seconds INTEGER DEFAULT 1800,
            last_failure_at TEXT,
            opened_at TEXT,
            metadata_json TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_circuit_breakers_status ON circuit_breakers(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_circuit_breakers_category ON circuit_breakers(category)")
    conn.execute("""
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '0.21.0-alpha')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    """)


MIGRATIONS: tuple[Migration, ...] = (
    Migration("0001_existing_db_repairs", "Backfill approval task links and memory FTS schema", _migration_0001_existing_db_repairs),
    Migration("0002_task_queue_locks", "Add task queue lease, idempotency, and scheduling fields", _migration_0002_task_queue_locks),
    Migration("0003_operating_reviews", "Ensure operating intelligence review storage", _migration_0003_operating_reviews),
    Migration("0004_memory_intelligence_indexes", "Add memory and reflection indexes for compaction and retrieval", _migration_0004_memory_intelligence_indexes),
    Migration("0005_sparse_memory_vectors", "Add local sparse vector storage for memory reranking", _migration_0005_sparse_memory_vectors),
    Migration("0006_project_execution_plans", "Add project execution plans and step tracking", _migration_0006_project_execution_plans),
    Migration("0007_process_table_runtime", "Align schema for process table and staged project runtime", _migration_0007_process_table_runtime),
    Migration("0008_control_room_dashboard", "Align schema for control room dashboard runtime", _migration_0008_control_room_dashboard),
    Migration("0009_pipeline_kernel", "Align schema for pipeline trace and typed decision routing runtime", _migration_0009_pipeline_kernel),
    Migration("0010_cognitive_growth_algorithms", "Add blackboard, stigmergy, MAP-Elites, and cognitive snapshot storage", _migration_0010_cognitive_growth_algorithms),
    Migration("0011_wake_signals_reactor", "Add wake signal storage for event-driven reactor runtime", _migration_0011_wake_signals_reactor),
    Migration("0012_discord_message_dedupe", "Deduplicate Discord events and enforce message idempotency", _migration_0012_discord_message_dedupe),
    Migration("0013_cognitive_graph_layer", "Add cognitive graph nodes, edges, activations, and traces", _migration_0013_cognitive_graph_layer),
    Migration("0014_advanced_learning_loops", "Add memory rollups, graph summaries, failure cases, and circuit breakers", _migration_0014_advanced_learning_loops),
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
