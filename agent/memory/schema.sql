CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT,
    importance REAL DEFAULT 0.5,
    processed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tags_json TEXT,
    importance REAL DEFAULT 0.5,
    confidence REAL DEFAULT 0.7,
    source_event_id INTEGER,
    last_used_at TEXT,
    use_count INTEGER DEFAULT 0,
    archived INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(archived);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    goal_type TEXT NOT NULL,
    status TEXT NOT NULL,
    priority REAL DEFAULT 0.5,
    risk_level TEXT DEFAULT 'low',
    requires_approval INTEGER DEFAULT 0,
    parent_goal_id INTEGER,
    due_at TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
CREATE INDEX IF NOT EXISTS idx_goals_type ON goals(goal_type);
CREATE INDEX IF NOT EXISTS idx_goals_priority ON goals(priority);

CREATE TABLE IF NOT EXISTS goal_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER NOT NULL,
    depends_on_goal_id INTEGER NOT NULL,
    dependency_type TEXT DEFAULT 'requires'
);

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    confidence REAL DEFAULT 0.6,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tool_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    action TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    approved INTEGER DEFAULT 0,
    success INTEGER,
    result_summary TEXT,
    raw_output TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    action_type TEXT NOT NULL,
    description TEXT NOT NULL,
    proposed_payload_json TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
CREATE INDEX IF NOT EXISTS idx_approvals_risk ON approvals(risk_level);

CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_event_id INTEGER,
    goal_id INTEGER,
    summary TEXT NOT NULL,
    learned_json TEXT,
    followup_goal_json TEXT,
    confidence REAL DEFAULT 0.7
);

CREATE TABLE IF NOT EXISTS pending_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    reason TEXT,
    priority REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS system_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    cpu_temp REAL,
    mem_used_mb INTEGER,
    mem_available_mb INTEGER,
    swap_used_mb INTEGER,
    disk_used_percent REAL,
    failed_services_json TEXT,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS renderer_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    rendered_text TEXT,
    success INTEGER,
    validation_result_json TEXT,
    duration_ms INTEGER,
    error TEXT
);


CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    user_input TEXT,
    selected_goal_id INTEGER,
    decision_json TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    risk_level TEXT DEFAULT 'low',
    renderer TEXT DEFAULT 'fallback'
);

CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_kind ON decisions(kind);
CREATE INDEX IF NOT EXISTS idx_decisions_goal ON decisions(selected_goal_id);


CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', '0.2.0-pre');
