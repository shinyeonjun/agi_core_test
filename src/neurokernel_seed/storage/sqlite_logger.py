from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from neurokernel_seed.core.schema import EpisodeEvent, EpisodeSummary


class SQLiteEpisodeLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def __enter__(self) -> "SQLiteEpisodeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
              id TEXT PRIMARY KEY,
              env_name TEXT NOT NULL,
              agent_name TEXT NOT NULL,
              seed INTEGER NOT NULL,
              env_metadata_json TEXT NOT NULL DEFAULT '{}',
              steps INTEGER DEFAULT 0,
              total_reward REAL DEFAULT 0,
              success INTEGER DEFAULT 0,
              done INTEGER DEFAULT 0,
              prediction_error REAL,
              created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              episode_id TEXT NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
              env_name TEXT NOT NULL,
              step_index INTEGER NOT NULL,
              action_json TEXT NOT NULL,
              prev_state_json TEXT NOT NULL,
              observation_json TEXT NOT NULL,
              next_state_json TEXT NOT NULL,
              reward REAL NOT NULL,
              done INTEGER NOT NULL,
              info_json TEXT NOT NULL,
              prediction_json TEXT,
              UNIQUE(episode_id, step_index)
            );
            """
        )
        self._ensure_column("episodes", "env_metadata_json", "TEXT NOT NULL DEFAULT '{}'")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def start_episode(self, episode_id: str, env_name: str, agent_name: str, seed: int, env_metadata: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO episodes(id, env_name, agent_name, seed, env_metadata_json) VALUES (?, ?, ?, ?, ?)",
            (episode_id, env_name, agent_name, seed, json.dumps(env_metadata or {}, ensure_ascii=False, sort_keys=True)),
        )
        self.conn.commit()

    def finish_episode(self, summary: EpisodeSummary) -> None:
        self.conn.execute(
            """
            UPDATE episodes
            SET steps=?, total_reward=?, success=?, done=?, prediction_error=?
            WHERE id=?
            """,
            (summary.steps, summary.total_reward, int(summary.success), int(summary.done), summary.prediction_error, summary.episode_id),
        )
        self.conn.commit()

    def log_event(self, event: EpisodeEvent) -> None:
        t = event.transition
        self.conn.execute(
            """
            INSERT INTO events(
              episode_id, env_name, step_index, action_json, prev_state_json, observation_json,
              next_state_json, reward, done, info_json, prediction_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.episode_id,
                event.env_name,
                event.step_index,
                json.dumps(_action(t.action), ensure_ascii=False, sort_keys=True),
                json.dumps(_state(t.prev_state), ensure_ascii=False, sort_keys=True),
                json.dumps(_observation(t.observation), ensure_ascii=False, sort_keys=True),
                json.dumps(_state(t.next_state), ensure_ascii=False, sort_keys=True),
                t.reward,
                int(t.done),
                json.dumps(t.info, ensure_ascii=False, sort_keys=True),
                json.dumps(_prediction(event.prediction), ensure_ascii=False, sort_keys=True) if event.prediction else None,
            ),
        )
        self.conn.commit()

    def episode_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def event_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])


def _action(action) -> dict[str, Any]:
    return {"name": action.name, "params": action.params, "source": action.source, "confidence": action.confidence}


def _state(state) -> dict[str, Any]:
    return {"env_name": state.env_name, "state_id": state.state_id, "vector": list(state.vector), "facts": state.facts, "terminal": state.terminal}


def _observation(obs) -> dict[str, Any]:
    return {"text": obs.text, "vector": list(obs.vector), "facts": obs.facts}


def _prediction(pred) -> dict[str, Any]:
    return {"next_state_vector": list(pred.next_state_vector), "reward": pred.reward, "terminal": pred.terminal, "confidence": pred.confidence, "reason": pred.reason}
