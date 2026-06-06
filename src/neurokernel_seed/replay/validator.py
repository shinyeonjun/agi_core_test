from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPLAY_SCHEMA_VERSION = "neurokernel-replay-v2"


class ReplayValidationError(ValueError):
    pass


def validate_replay(path: str | Path) -> dict[str, Any]:
    replay_path = Path(path)
    meta_path = replay_path.with_suffix(replay_path.suffix + ".meta.json")
    if not replay_path.exists():
        raise ReplayValidationError(f"missing replay file: {replay_path}")
    if not meta_path.exists():
        raise ReplayValidationError(f"missing metadata file: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayValidationError(f"unsupported replay schema: {meta.get('schema_version')}")
    rows = 0
    envs: set[str] = set()
    splits: set[str] = set()
    with replay_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            _validate_record(record, line_no)
            rows += 1
            envs.add(record["env_name"])
            if record.get("split"):
                splits.add(record["split"])
    if rows != meta.get("rows"):
        raise ReplayValidationError(f"row count mismatch: data={rows}, meta={meta.get('rows')}")
    if sorted(envs) != meta.get("envs"):
        raise ReplayValidationError("env list mismatch")
    if sorted(splits) != meta.get("splits"):
        raise ReplayValidationError("split list mismatch")
    return {"rows": rows, "envs": sorted(envs), "splits": sorted(splits), "format": meta.get("format")}


def _validate_record(record: dict[str, Any], line_no: int) -> None:
    required = [
        "schema_version",
        "episode_id",
        "env_name",
        "split",
        "step_index",
        "prev_state_id",
        "next_state_id",
        "state_vector",
        "state_facts",
        "observation_text",
        "observation_vector",
        "observation_facts",
        "action",
        "actual_next_state_vector",
        "next_state_facts",
        "reward",
        "done",
        "env_metadata",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise ReplayValidationError(f"line {line_no}: missing keys {missing}")
    if record["schema_version"] != REPLAY_SCHEMA_VERSION:
        raise ReplayValidationError(f"line {line_no}: unsupported schema {record['schema_version']}")
    if len(record["state_vector"]) != len(record["actual_next_state_vector"]):
        raise ReplayValidationError(f"line {line_no}: vector length mismatch")
    if len(record["observation_vector"]) != len(record["actual_next_state_vector"]):
        raise ReplayValidationError(f"line {line_no}: observation vector length mismatch")
    predicted = record.get("predicted_next_state_vector")
    if predicted is not None and len(predicted) != len(record["actual_next_state_vector"]):
        raise ReplayValidationError(f"line {line_no}: predicted vector length mismatch")
    if not isinstance(record["action"], dict) or "name" not in record["action"]:
        raise ReplayValidationError(f"line {line_no}: invalid action")
    if not isinstance(record["env_metadata"], dict) or not record["env_metadata"].get("name"):
        raise ReplayValidationError(f"line {line_no}: invalid env metadata")
