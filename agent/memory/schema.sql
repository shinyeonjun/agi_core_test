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

CREATE TABLE IF NOT EXISTS style_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    profile_json TEXT NOT NULL,
    confidence REAL DEFAULT 0.6,
    active INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_style_profiles_active ON style_profiles(active);

CREATE TABLE IF NOT EXISTS style_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_text TEXT NOT NULL,
    target_response_id INTEGER,
    feedback_type TEXT NOT NULL,
    feedback_text TEXT,
    extracted_preference_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_style_feedback_type ON style_feedback(feedback_type);

CREATE TABLE IF NOT EXISTS style_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    label TEXT NOT NULL,
    input_text TEXT,
    good_response TEXT,
    bad_response TEXT,
    reason TEXT,
    tags_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_style_examples_label ON style_examples(label);

CREATE TABLE IF NOT EXISTS interpretation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_event_id INTEGER,
    engine TEXT NOT NULL,
    input_text TEXT NOT NULL,
    result_json TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    accepted INTEGER DEFAULT 0,
    fallback_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_interpretation_logs_created ON interpretation_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_interpretation_logs_engine ON interpretation_logs(engine);

CREATE TABLE IF NOT EXISTS language_interpretation_cache (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    engine TEXT NOT NULL,
    normalized_input TEXT NOT NULL,
    result_json TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_language_cache_engine ON language_interpretation_cache(engine);
CREATE INDEX IF NOT EXISTS idx_language_cache_updated ON language_interpretation_cache(updated_at);

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

CREATE TABLE IF NOT EXISTS self_maps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    summary TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    changed INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_self_maps_created ON self_maps(created_at);
CREATE INDEX IF NOT EXISTS idx_self_maps_fingerprint ON self_maps(fingerprint);

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
    tags_json,
    content='memories',
    content_rowid='id'
);
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


CREATE TABLE IF NOT EXISTS workspace_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    title TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    bytes INTEGER DEFAULT 0,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_workspace_artifacts_created ON workspace_artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_workspace_artifacts_type ON workspace_artifacts(artifact_type);

CREATE TABLE IF NOT EXISTS project_specs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT NOT NULL,
    objective TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    artifact_id INTEGER,
    spec_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_project_specs_status ON project_specs(status);


CREATE TABLE IF NOT EXISTS action_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    goal_id INTEGER,
    action_type TEXT NOT NULL,
    command_json TEXT NOT NULL,
    cwd TEXT,
    profile TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL,
    returncode INTEGER,
    stdout TEXT,
    stderr TEXT,
    before_snapshot_json TEXT,
    after_snapshot_json TEXT,
    result_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_runs_created ON action_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_action_runs_profile ON action_runs(profile);
CREATE INDEX IF NOT EXISTS idx_action_runs_status ON action_runs(status);

CREATE TABLE IF NOT EXISTS action_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    goal_id INTEGER,
    command TEXT NOT NULL,
    cwd TEXT,
    profile TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_action_proposals_created ON action_proposals(created_at);
CREATE INDEX IF NOT EXISTS idx_action_proposals_goal ON action_proposals(goal_id);
CREATE INDEX IF NOT EXISTS idx_action_proposals_profile ON action_proposals(profile);
CREATE INDEX IF NOT EXISTS idx_action_proposals_status ON action_proposals(status);

CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    queue_type TEXT NOT NULL,
    status TEXT NOT NULL,
    priority REAL DEFAULT 0.5,
    goal_id INTEGER,
    approval_id INTEGER,
    task_kind TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT,
    payload_json TEXT,
    attempts INTEGER DEFAULT 0,
    claimed_at TEXT,
    completed_at TEXT,
    result_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_type_status ON task_queue(queue_type, status);
CREATE INDEX IF NOT EXISTS idx_task_queue_goal ON task_queue(goal_id);
CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(priority);

CREATE TABLE IF NOT EXISTS task_lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    task_id INTEGER NOT NULL,
    queue_type TEXT,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_lifecycle_task ON task_lifecycle_events(task_id);
CREATE INDEX IF NOT EXISTS idx_task_lifecycle_phase ON task_lifecycle_events(phase);
CREATE INDEX IF NOT EXISTS idx_task_lifecycle_created ON task_lifecycle_events(created_at);

CREATE TABLE IF NOT EXISTS root_objectives (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    objective_type TEXT NOT NULL UNIQUE,
    priority REAL DEFAULT 0.5,
    enabled INTEGER DEFAULT 1,
    cooldown_seconds INTEGER DEFAULT 21600,
    last_used_at TEXT,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_root_objectives_enabled ON root_objectives(enabled);
CREATE INDEX IF NOT EXISTS idx_root_objectives_type ON root_objectives(objective_type);

CREATE TABLE IF NOT EXISTS generated_goal_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    root_objective_id INTEGER,
    title TEXT NOT NULL,
    description TEXT,
    goal_type TEXT NOT NULL,
    novelty_score REAL DEFAULT 0.0,
    utility_score REAL DEFAULT 0.0,
    risk_level TEXT DEFAULT 'low',
    score REAL DEFAULT 0.0,
    status TEXT NOT NULL,
    rejection_reason TEXT,
    generated_goal_id INTEGER,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_generated_goal_candidates_created ON generated_goal_candidates(created_at);
CREATE INDEX IF NOT EXISTS idx_generated_goal_candidates_status ON generated_goal_candidates(status);
CREATE INDEX IF NOT EXISTS idx_generated_goal_candidates_goal ON generated_goal_candidates(generated_goal_id);

CREATE TABLE IF NOT EXISTS operating_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    review_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'open',
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_operating_reviews_created ON operating_reviews(created_at);
CREATE INDEX IF NOT EXISTS idx_operating_reviews_type ON operating_reviews(review_type);
CREATE INDEX IF NOT EXISTS idx_operating_reviews_status ON operating_reviews(status);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT OR REPLACE INTO schema_meta (key, value)
VALUES ('schema_version', '0.11.0-alpha');
