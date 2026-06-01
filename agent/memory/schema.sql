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

CREATE TABLE IF NOT EXISTS policy_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    action_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    requires_approval INTEGER NOT NULL,
    denied INTEGER NOT NULL,
    reason TEXT,
    matched_rules_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_created ON policy_decisions(created_at);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_risk ON policy_decisions(risk_level);

CREATE TABLE IF NOT EXISTS discord_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'discord',
    guild_id TEXT,
    channel_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    is_dm INTEGER DEFAULT 0,
    was_mention INTEGER DEFAULT 0,
    content_redacted TEXT NOT NULL,
    event_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_discord_events_user ON discord_events(user_id);
CREATE INDEX IF NOT EXISTS idx_discord_events_channel ON discord_events(channel_id);

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
CREATE INDEX IF NOT EXISTS idx_reflections_goal ON reflections(goal_id);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    trigger_description TEXT NOT NULL,
    procedure_json TEXT NOT NULL,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    tags_json TEXT,
    last_used_at TEXT,
    archived INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skills_archived ON skills(archived);
CREATE INDEX IF NOT EXISTS idx_skills_confidence ON skills(confidence);

CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    suite_name TEXT NOT NULL,
    result TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    details_json TEXT,
    commit_hash TEXT,
    schema_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_eval_runs_suite ON eval_runs(suite_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_result ON eval_runs(result);

CREATE TABLE IF NOT EXISTS cooldowns (
    key TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    seconds INTEGER NOT NULL,
    metadata_json TEXT
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

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title,
    content,
    tags,
    content='memories',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, title, content, tags)
  VALUES (new.id, new.title, new.content, COALESCE(new.tags_json, ''));
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
  VALUES('delete', old.id, old.title, old.content, COALESCE(old.tags_json, ''));
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
  VALUES('delete', old.id, old.title, old.content, COALESCE(old.tags_json, ''));
  INSERT INTO memories_fts(rowid, title, content, tags)
  VALUES (new.id, new.title, new.content, COALESCE(new.tags_json, ''));
END;

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT OR REPLACE INTO schema_meta (key, value)
VALUES ('schema_version', '0.6.0-alpha');
