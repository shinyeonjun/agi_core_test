from __future__ import annotations

import sqlite3


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  created_by TEXT,
  source TEXT,
  goal TEXT NOT NULL,
  target TEXT NOT NULL,
  status TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  requires_approval INTEGER NOT NULL,
  task_spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_events (
  event_id TEXT PRIMARY KEY,
  idempotency_key TEXT UNIQUE,
  event_type TEXT NOT NULL,
  source TEXT NOT NULL,
  actor_id TEXT,
  work_id TEXT,
  job_id TEXT,
  task_id TEXT,
  proposal_id TEXT,
  correlation_id TEXT,
  causation_id TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'recorded',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  processed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_events_created_at ON agent_events(created_at);
CREATE INDEX IF NOT EXISTS idx_agent_events_work_id ON agent_events(work_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_job_id ON agent_events(job_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_type ON agent_events(event_type);
CREATE TABLE IF NOT EXISTS experiences (
  experience_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  phase TEXT NOT NULL,
  status TEXT NOT NULL,
  decision_policy TEXT NOT NULL,
  model_used INTEGER NOT NULL DEFAULT 0,
  model_unavailable_reason TEXT,
  before_state_json TEXT NOT NULL,
  after_state_json TEXT NOT NULL DEFAULT '{}',
  outcome_json TEXT NOT NULL DEFAULT '{}',
  learning_masks_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_experiences_task_id ON experiences(task_id);
CREATE INDEX IF NOT EXISTS idx_experiences_created_at ON experiences(created_at);
CREATE TABLE IF NOT EXISTS experience_candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experience_id TEXT NOT NULL REFERENCES experiences(experience_id) ON DELETE CASCADE,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  candidate_index INTEGER NOT NULL,
  action_id TEXT NOT NULL,
  params_json TEXT NOT NULL DEFAULT '{}',
  safety_decision_json TEXT NOT NULL DEFAULT '{}',
  model_score_json TEXT NOT NULL DEFAULT '{}',
  selected INTEGER NOT NULL DEFAULT 0,
  executed INTEGER NOT NULL DEFAULT 0,
  execution_result_known INTEGER NOT NULL DEFAULT 0,
  outcome_json TEXT NOT NULL DEFAULT '{}',
  target_mask_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_experience_candidates_experience_id ON experience_candidates(experience_id);
CREATE INDEX IF NOT EXISTS idx_experience_candidates_task_id ON experience_candidates(task_id);
CREATE TABLE IF NOT EXISTS action_decisions (
  decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  step INTEGER NOT NULL,
  candidate_actions_json TEXT NOT NULL,
  chosen_action_json TEXT,
  model_score_json TEXT NOT NULL DEFAULT '{}',
  safety_decision_json TEXT NOT NULL,
  gate_trace_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS execution_results (
  result_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  action_id TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL,
  success INTEGER NOT NULL,
  stdout_redacted TEXT NOT NULL DEFAULT '',
  stderr_redacted TEXT NOT NULL DEFAULT '',
  result_json TEXT NOT NULL,
  error_type TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  requested_by TEXT,
  approved_by TEXT,
  approved_at TEXT DEFAULT CURRENT_TIMESTAMP,
  decision TEXT NOT NULL,
  scope TEXT NOT NULL,
  reason TEXT
);
CREATE TABLE IF NOT EXISTS traces (
  trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  failure_bucket TEXT NOT NULL,
  state_json TEXT NOT NULL DEFAULT '{}',
  candidate_actions_json TEXT NOT NULL DEFAULT '[]',
  chosen_action_json TEXT NOT NULL DEFAULT '{}',
  expected_result_json TEXT NOT NULL DEFAULT '{}',
  actual_result_json TEXT NOT NULL DEFAULT '{}',
  analysis_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS user_preferences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  key TEXT NOT NULL,
  value_json TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'global',
  source TEXT NOT NULL DEFAULT 'explicit_user_request',
  confidence REAL NOT NULL DEFAULT 1.0,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
  expires_at TEXT,
  UNIQUE(user_id, key, scope)
);
CREATE TABLE IF NOT EXISTS conversation_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  channel_id TEXT,
  message_id TEXT,
  role TEXT NOT NULL,
  content_redacted TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  linked_task_id TEXT
);
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  channel_id TEXT,
  summary TEXT NOT NULL,
  covered_from INTEGER,
  covered_to INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS task_references (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  short_label TEXT NOT NULL,
  status TEXT NOT NULL,
  result_summary TEXT NOT NULL DEFAULT '',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS work_items (
  work_id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  goal TEXT NOT NULL,
  status TEXT NOT NULL,
  priority TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  owner_user_id TEXT,
  channel_id TEXT,
  source_message_id TEXT,
  parent_work_id TEXT,
  linked_entity_type TEXT,
  linked_entity_id TEXT,
  route_reason TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS work_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
  timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS work_notes (
  note_id INTEGER PRIMARY KEY AUTOINCREMENT,
  work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
  timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  actor TEXT NOT NULL,
  note_redacted TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS work_jobs (
  job_id TEXT PRIMARY KEY,
  work_id TEXT NOT NULL REFERENCES work_items(work_id) ON DELETE CASCADE,
  queue_name TEXT NOT NULL,
  status TEXT NOT NULL,
  priority TEXT NOT NULL DEFAULT 'medium',
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  worker_id TEXT,
  redis_message_id TEXT,
  heartbeat_at TEXT,
  last_error TEXT,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  enqueued_at TEXT,
  started_at TEXT,
  completed_at TEXT,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_work_jobs_work_id ON work_jobs(work_id);
CREATE INDEX IF NOT EXISTS idx_work_jobs_status ON work_jobs(status);
CREATE INDEX IF NOT EXISTS idx_work_jobs_queue_status ON work_jobs(queue_name, status);
CREATE TABLE IF NOT EXISTS capability_gaps (
  gap_id TEXT PRIMARY KEY,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  user_id TEXT,
  channel_id TEXT,
  request_text TEXT NOT NULL,
  normalized_request TEXT NOT NULL,
  gap_type TEXT NOT NULL,
  requested_capability TEXT NOT NULL,
  matched_actions_json TEXT NOT NULL DEFAULT '[]',
  confidence REAL NOT NULL DEFAULT 0.0,
  status TEXT NOT NULL,
  linked_task_id TEXT
);
CREATE TABLE IF NOT EXISTS capability_proposals (
  proposal_id TEXT PRIMARY KEY,
  gap_id TEXT NOT NULL REFERENCES capability_gaps(gap_id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  action_id TEXT NOT NULL,
  capability_name TEXT NOT NULL,
  purpose TEXT NOT NULL,
  target TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  side_effect INTEGER NOT NULL,
  requires_approval INTEGER NOT NULL,
  inputs_json TEXT NOT NULL,
  outputs_json TEXT NOT NULL,
  implementation_hint_json TEXT NOT NULL,
  test_plan_json TEXT NOT NULL,
  safety_notes_json TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.0,
  approval_required_for_implementation INTEGER NOT NULL,
  activation_requires_tests INTEGER NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS proposal_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  proposal_id TEXT NOT NULL REFERENCES capability_proposals(proposal_id) ON DELETE CASCADE,
  timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
  event_type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_json TEXT NOT NULL
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    migrate_schema(conn)
    conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    ensure_column(conn, "capability_proposals", "work_id", "TEXT")
    ensure_column(conn, "work_items", "queued_at", "TEXT")
    ensure_column(conn, "work_jobs", "heartbeat_at", "TEXT")


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    existing = {str(row["name"]) for row in rows}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
