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
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at);
CREATE INDEX IF NOT EXISTS idx_memories_use_count ON memories(use_count);

CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id INTEGER NOT NULL,
    vector_type TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    nonzero_count INTEGER DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(memory_id, vector_type)
);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_type ON memory_vectors(vector_type);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_hash ON memory_vectors(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_vectors_updated ON memory_vectors(updated_at);

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
CREATE INDEX IF NOT EXISTS idx_reflections_created ON reflections(created_at);
CREATE INDEX IF NOT EXISTS idx_reflections_summary ON reflections(summary);

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
    max_attempts INTEGER DEFAULT 3,
    claimed_at TEXT,
    locked_until TEXT,
    locked_by TEXT,
    idempotency_key TEXT,
    not_before TEXT,
    due_at TEXT,
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
);
CREATE INDEX IF NOT EXISTS idx_project_plans_goal ON project_execution_plans(goal_id);
CREATE INDEX IF NOT EXISTS idx_project_plans_task ON project_execution_plans(task_id);
CREATE INDEX IF NOT EXISTS idx_project_plans_status ON project_execution_plans(status);

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
);
CREATE INDEX IF NOT EXISTS idx_project_steps_plan ON project_execution_steps(plan_id);
CREATE INDEX IF NOT EXISTS idx_project_steps_status ON project_execution_steps(status);

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
);
CREATE INDEX IF NOT EXISTS idx_blackboard_status ON blackboard_items(status);
CREATE INDEX IF NOT EXISTS idx_blackboard_topic ON blackboard_items(topic);
CREATE INDEX IF NOT EXISTS idx_blackboard_confidence ON blackboard_items(confidence);

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
);
CREATE INDEX IF NOT EXISTS idx_cognitive_map_archive ON cognitive_map_elites(archive_name);
CREATE INDEX IF NOT EXISTS idx_cognitive_map_score ON cognitive_map_elites(score);
CREATE INDEX IF NOT EXISTS idx_cognitive_map_status ON cognitive_map_elites(status);

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
);
CREATE INDEX IF NOT EXISTS idx_stigmergy_status ON stigmergy_markers(status);
CREATE INDEX IF NOT EXISTS idx_stigmergy_type ON stigmergy_markers(marker_type);
CREATE INDEX IF NOT EXISTS idx_stigmergy_intensity ON stigmergy_markers(intensity);

CREATE TABLE IF NOT EXISTS cognitive_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    snapshot_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cognitive_snapshots_created ON cognitive_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_snapshots_type ON cognitive_snapshots(snapshot_type);


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
);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_type ON cognitive_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_source ON cognitive_nodes(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_revisit ON cognitive_nodes(revisit_score);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_updated ON cognitive_nodes(updated_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_archived ON cognitive_nodes(archived);

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
);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_from ON cognitive_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_to ON cognitive_edges(to_node_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_type ON cognitive_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_weight ON cognitive_edges(weight);

CREATE TABLE IF NOT EXISTS cognitive_activations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    node_id INTEGER NOT NULL,
    context TEXT,
    reason TEXT,
    score REAL NOT NULL,
    components_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_cognitive_activations_created ON cognitive_activations(created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_activations_node ON cognitive_activations(node_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_activations_score ON cognitive_activations(score);

CREATE TABLE IF NOT EXISTS cognitive_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    trace_type TEXT NOT NULL,
    decision_id TEXT,
    root_node_id INTEGER,
    summary TEXT,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cognitive_traces_created ON cognitive_traces(created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_traces_type ON cognitive_traces(trace_type);
CREATE INDEX IF NOT EXISTS idx_cognitive_traces_decision ON cognitive_traces(decision_id);

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
);
CREATE INDEX IF NOT EXISTS idx_wake_signals_status ON wake_signals(status);
CREATE INDEX IF NOT EXISTS idx_wake_signals_type ON wake_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_wake_signals_priority ON wake_signals(priority);
CREATE INDEX IF NOT EXISTS idx_wake_signals_not_before ON wake_signals(not_before);
CREATE UNIQUE INDEX IF NOT EXISTS idx_wake_signals_pending_dedupe ON wake_signals(dedupe_key) WHERE status = 'pending' AND dedupe_key IS NOT NULL;


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
);
CREATE INDEX IF NOT EXISTS idx_memory_rollups_level ON memory_rollups(level);
CREATE INDEX IF NOT EXISTS idx_memory_rollups_score ON memory_rollups(score);
CREATE INDEX IF NOT EXISTS idx_memory_rollups_archived ON memory_rollups(archived);

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
);
CREATE INDEX IF NOT EXISTS idx_graph_summaries_score ON graph_community_summaries(score);
CREATE INDEX IF NOT EXISTS idx_graph_summaries_archived ON graph_community_summaries(archived);

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
);
CREATE INDEX IF NOT EXISTS idx_failure_cases_category ON failure_cases(category);
CREATE INDEX IF NOT EXISTS idx_failure_cases_task ON failure_cases(task_id);
CREATE INDEX IF NOT EXISTS idx_failure_cases_resolved ON failure_cases(resolved);

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
);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_status ON circuit_breakers(status);
CREATE INDEX IF NOT EXISTS idx_circuit_breakers_category ON circuit_breakers(category);


CREATE TABLE IF NOT EXISTS research_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    question_key TEXT NOT NULL UNIQUE, title TEXT NOT NULL, prompt TEXT NOT NULL, source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open', priority REAL DEFAULT 0.5, novelty REAL DEFAULT 0.5,
    utility REAL DEFAULT 0.5, risk_level TEXT DEFAULT 'low', evidence_json TEXT, metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_questions_status ON research_questions(status);
CREATE INDEX IF NOT EXISTS idx_research_questions_priority ON research_questions(priority);
CREATE INDEX IF NOT EXISTS idx_research_questions_source ON research_questions(source);
CREATE TABLE IF NOT EXISTS research_hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    question_id INTEGER, hypothesis_key TEXT NOT NULL UNIQUE, statement TEXT NOT NULL,
    rationale TEXT, expected_effect TEXT, falsification TEXT, confidence REAL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'open', evidence_json TEXT, metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_hypotheses_question ON research_hypotheses(question_id);
CREATE INDEX IF NOT EXISTS idx_research_hypotheses_status ON research_hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_research_hypotheses_confidence ON research_hypotheses(confidence);
CREATE TABLE IF NOT EXISTS research_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    question_id INTEGER, hypothesis_id INTEGER, experiment_key TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL, plan_json TEXT NOT NULL, variables_json TEXT, success_criteria_json TEXT,
    verification_commands_json TEXT, status TEXT NOT NULL DEFAULT 'planned', result_json TEXT,
    score REAL DEFAULT 0.0, risk_level TEXT DEFAULT 'low'
);
CREATE INDEX IF NOT EXISTS idx_research_experiments_question ON research_experiments(question_id);
CREATE INDEX IF NOT EXISTS idx_research_experiments_hypothesis ON research_experiments(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_research_experiments_status ON research_experiments(status);
CREATE TABLE IF NOT EXISTS research_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, evidence_type TEXT NOT NULL,
    source_type TEXT NOT NULL, source_id TEXT, title TEXT NOT NULL, summary TEXT, payload_json TEXT,
    confidence REAL DEFAULT 0.5, supports_type TEXT, supports_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_research_evidence_type ON research_evidence(evidence_type);
CREATE INDEX IF NOT EXISTS idx_research_evidence_source ON research_evidence(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_research_evidence_supports ON research_evidence(supports_type, supports_id);
CREATE TABLE IF NOT EXISTS research_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    question_id INTEGER, hypothesis_id INTEGER, experiment_id INTEGER, title TEXT NOT NULL,
    proposal_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'proposed', priority REAL DEFAULT 0.5,
    risk_level TEXT DEFAULT 'low', worker_prompt TEXT, success_criteria_json TEXT, evidence_ids_json TEXT,
    metadata_json TEXT, queued_task_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_research_proposals_status ON research_proposals(status);
CREATE INDEX IF NOT EXISTS idx_research_proposals_priority ON research_proposals(priority);
CREATE INDEX IF NOT EXISTS idx_research_proposals_experiment ON research_proposals(experiment_id);

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
INSERT OR REPLACE INTO schema_meta (key, value)
VALUES ('schema_version', '0.22.0-alpha');
