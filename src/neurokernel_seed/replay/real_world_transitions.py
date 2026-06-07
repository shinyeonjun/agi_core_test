from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.harness.trace import redact_text


REAL_WORLD_TRANSITION_SCHEMA_VERSION = "neurokernel-real-world-transition-v1"


def export_real_world_transitions(db_path: str | Path, out_path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"harness db not found: {db}")
    rows = _extract_rows(db, limit=limit)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": REAL_WORLD_TRANSITION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(db),
        "out": str(out),
        "rows": len(rows),
        "actions": sorted({str(row["action"]["action_id"]) for row in rows}),
        "answer_quality_counts": _counts(row.get("interaction", {}).get("answer_quality") for row in rows),
        "status_counts": _counts(row.get("next_state", {}).get("task_status") for row in rows),
    }
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def _extract_rows(db: Path, *, limit: int | None) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        tasks = {str(row["task_id"]): _decode(row) for row in conn.execute("SELECT * FROM tasks")}
        experiences = [_decode(row) for row in conn.execute("SELECT * FROM experiences ORDER BY created_at, experience_id")]
        candidates = [_decode(row) for row in conn.execute("SELECT * FROM experience_candidates ORDER BY experience_id, candidate_index, id")]
        interactions = _load_interactions(conn)
    candidates_by_experience: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_experience[str(candidate["experience_id"])].append(candidate)
    interactions_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for interaction in interactions:
        task_id = interaction.get("task_id")
        if task_id:
            interactions_by_task[str(task_id)].append(interaction)

    rows: list[dict[str, Any]] = []
    for experience in experiences:
        task_id = str(experience.get("task_id") or "")
        task = tasks.get(task_id)
        if task is None:
            continue
        for candidate in candidates_by_experience.get(str(experience["experience_id"]), []):
            if not candidate.get("execution_result_known"):
                continue
            rows.append(_transition_row(task, experience, candidate, interactions_by_task.get(task_id, [])))
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def _transition_row(task: dict[str, Any], experience: dict[str, Any], candidate: dict[str, Any], interactions: list[dict[str, Any]]) -> dict[str, Any]:
    outcome = _dict(candidate.get("outcome_json"))
    before = _dict(experience.get("before_state_json"))
    after = _dict(experience.get("after_state_json"))
    params = _dict(candidate.get("params_json"))
    interaction = _interaction_payload(interactions)
    action_id = str(candidate.get("action_id") or "")
    return {
        "schema_version": REAL_WORLD_TRANSITION_SCHEMA_VERSION,
        "row_id": _row_id(task, experience, candidate),
        "task_id": task.get("task_id"),
        "experience_id": experience.get("experience_id"),
        "candidate_index": int(candidate.get("candidate_index") or 0),
        "source": {
            "task_source": task.get("source"),
            "experience_phase": experience.get("phase"),
            "decision_policy": experience.get("decision_policy"),
            "model_used": bool(experience.get("model_used")),
        },
        "state": {
            "target": task.get("target"),
            "risk_level": task.get("risk_level"),
            "mode": before.get("mode"),
            "task_status": before.get("task_status"),
            "candidate_actions": _list(before.get("candidate_actions")),
            "required_outputs": interaction.get("required_outputs", []),
            "answered_outputs": interaction.get("answered_outputs", []),
            "missing_outputs": interaction.get("missing_outputs", []),
        },
        "action": {
            "action_id": action_id,
            "params": params,
            "safety_decision": _dict(candidate.get("safety_decision_json")),
        },
        "next_state": {
            "task_status": after.get("task_status"),
            "action_id": after.get("action_id") or action_id,
            "success": outcome.get("success"),
            "answered_outputs": interaction.get("answered_outputs", []),
            "missing_outputs": interaction.get("missing_outputs", []),
        },
        "observation": {
            "success": outcome.get("success"),
            "reward": float(outcome.get("reward") or 0.0),
            "duration_seconds": float(outcome.get("duration_seconds") or 0.0),
            "failure_present": bool(outcome.get("failure_present")),
            "error_type": outcome.get("error_type"),
        },
        "interaction": interaction,
    }


def _load_interactions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interaction_outcomes'").fetchone()
    if exists is None:
        return []
    return [_decode(row) for row in conn.execute("SELECT * FROM interaction_outcomes ORDER BY created_at, id")]


def _interaction_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "interaction_aligned": False,
            "required_outputs": [],
            "answered_outputs": [],
            "missing_outputs": [],
            "answer_quality": "unknown",
        }
    latest = rows[-1]
    return {
        "interaction_aligned": True,
        "required_outputs": _list(latest.get("required_outputs_json")),
        "answered_outputs": _list(latest.get("answered_outputs_json")),
        "missing_outputs": _list(latest.get("missing_outputs_json")),
        "answer_quality": latest.get("answer_quality"),
        "request_text_redacted": redact_text(str(latest.get("request_text_redacted") or ""), max_chars=500),
    }


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key, value in list(data.items()):
        if key.endswith("_json") and isinstance(value, str):
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                data[key] = {}
    return data


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_id(task: dict[str, Any], experience: dict[str, Any], candidate: dict[str, Any]) -> str:
    raw = {
        "task_id": task.get("task_id"),
        "experience_id": experience.get("experience_id"),
        "candidate_index": candidate.get("candidate_index"),
        "action_id": candidate.get("action_id"),
    }
    digest = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"real_world_{digest}"


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
