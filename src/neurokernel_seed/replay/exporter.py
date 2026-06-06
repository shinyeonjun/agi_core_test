from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

REPLAY_SCHEMA_VERSION = "neurokernel-replay-v2"


class ReplayExporter:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def export_jsonl(self, out_path: str | Path) -> dict[str, Any]:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT e.*, ep.agent_name, ep.seed, ep.env_metadata_json
            FROM events e JOIN episodes ep ON ep.id = e.episode_id
            ORDER BY e.episode_id, e.step_index
            """
        ).fetchall()
        envs: set[str] = set()
        splits: set[str] = set()
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                prev_state = json.loads(row["prev_state_json"])
                observation = json.loads(row["observation_json"])
                next_state = json.loads(row["next_state_json"])
                action = json.loads(row["action_json"])
                prediction = json.loads(row["prediction_json"]) if row["prediction_json"] else None
                env_metadata = json.loads(row["env_metadata_json"] or "{}")
                split = env_metadata.get("split") or prev_state.get("facts", {}).get("split")
                envs.add(row["env_name"])
                if split:
                    splits.add(str(split))
                record = {
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "episode_id": row["episode_id"],
                    "env_name": row["env_name"],
                    "split": split,
                    "agent_name": row["agent_name"],
                    "seed": row["seed"],
                    "step_index": row["step_index"],
                    "prev_state_id": prev_state["state_id"],
                    "next_state_id": next_state["state_id"],
                    "state_vector": prev_state["vector"],
                    "state_facts": prev_state["facts"],
                    "observation_text": observation["text"],
                    "observation_vector": observation["vector"],
                    "observation_facts": observation["facts"],
                    "action": action,
                    "predicted_next_state_vector": prediction["next_state_vector"] if prediction else None,
                    "actual_next_state_vector": next_state["vector"],
                    "next_state_facts": next_state["facts"],
                    "reward": row["reward"],
                    "done": bool(row["done"]),
                    "info": json.loads(row["info_json"]),
                    "env_metadata": env_metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        meta = {
            "source_db": str(self.db_path),
            "rows": len(rows),
            "envs": sorted(envs),
            "splits": sorted(splits),
            "format": REPLAY_SCHEMA_VERSION,
            "schema_version": REPLAY_SCHEMA_VERSION,
        }
        out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        conn.close()
        return meta
