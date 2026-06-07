from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.harness.trace import redact_text

RUNTIME_REPLAY_SCHEMA_VERSION = "neurokernel-runtime-action-v1"
FORBIDDEN_KEY_FRAGMENTS = ("token", "password", "secret", "api_key", "apikey")
FORBIDDEN_KEY_EXACT = {"env", "environ", "environment", "env_vars", "environment_variables"}


class RuntimeReplayEtlError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeReplayEtlConfig:
    db_path: str | Path
    out_path: str | Path
    limit: int | None = None
    min_rows: int = 1
    require_execution: bool = True


def run_runtime_replay_etl(config: RuntimeReplayEtlConfig) -> dict[str, Any]:
    db = Path(config.db_path)
    out = Path(config.out_path)
    if not db.exists():
        raise FileNotFoundError(f"harness db not found: {db}")
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()
    rows, extraction = _extract_runtime_rows(db, limit=config.limit)
    manifest = _manifest(db, out, rows, extraction=extraction, run_id=run_id)
    staging = out.with_name(f".{out.name}.{run_id}.staging")
    quarantine = out.with_name(f"{out.name}.failed.{run_id}")
    _write_jsonl(staging, rows)
    validation = validate_runtime_replay(staging)
    gates = check_runtime_replay_gates(staging, min_rows=config.min_rows, require_execution=config.require_execution)
    manifest = {
        **manifest,
        "validation": validation,
        "gates": gates,
        "status": "passed" if gates["passed"] else "failed",
    }
    if not gates["passed"]:
        staging.replace(quarantine)
        failed_manifest = Path(str(quarantine) + ".manifest.json")
        _write_json_atomic(failed_manifest, {**manifest, "out": str(quarantine), "quarantine": str(quarantine)})
        return {**manifest, "out": str(quarantine), "quarantine": str(quarantine), "ready_for_runtime_training": False}
    staging.replace(out)
    manifest = {**manifest, "out": str(out), "sha256": _sha256(out), "bytes": out.stat().st_size}
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path), "ready_for_runtime_training": True}


def export_runtime_replay(db_path: str | Path, out_path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"harness db not found: {db}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, extraction = _extract_runtime_rows(db, limit=limit)
    _write_jsonl_atomic(out, rows)
    manifest = _manifest(db, out, rows, extraction=extraction, run_id=_new_run_id())
    manifest = {**manifest, "sha256": _sha256(out), "bytes": out.stat().st_size}
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return manifest


def validate_runtime_replay(path: str | Path) -> dict[str, Any]:
    replay = Path(path)
    rows = _load_rows(replay)
    errors: list[dict[str, Any]] = []
    success_rows = 0
    failure_rows = 0
    row_ids: set[str] = set()
    duplicate_row_ids: list[str] = []
    for index, row in enumerate(rows):
        errors.extend(_validate_row(index, row))
        row_id = row.get("row_id") if isinstance(row, dict) else None
        if row_id:
            if str(row_id) in row_ids:
                duplicate_row_ids.append(str(row_id))
            row_ids.add(str(row_id))
        outcome = row.get("outcome") if isinstance(row, dict) else {}
        if isinstance(outcome, dict) and outcome.get("success") is True:
            success_rows += 1
        if isinstance(outcome, dict) and outcome.get("success") is False:
            failure_rows += 1
    if duplicate_row_ids:
        errors.append({"row": None, "error": f"duplicate row_id: {duplicate_row_ids[:5]}"})
    result = {
        "accepted": not errors,
        "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
        "path": str(replay),
        "rows": len(rows),
        "success_rows": success_rows,
        "failure_rows": failure_rows,
        "duplicate_row_ids": duplicate_row_ids[:20],
        "errors": errors[:50],
    }
    return result


def check_runtime_replay_gates(path: str | Path, *, min_rows: int = 1, require_execution: bool = True) -> dict[str, Any]:
    validation = validate_runtime_replay(path)
    rows = _load_rows(path) if validation["accepted"] else []
    executed_rows = sum(1 for row in rows if isinstance(row.get("execution"), dict) and row["execution"].get("action_id"))
    chosen_rows = sum(1 for row in rows if isinstance(row.get("decision"), dict) and row["decision"].get("chosen_action"))
    candidate_rows = sum(1 for row in rows if isinstance(row.get("decision"), dict) and row["decision"].get("candidate_actions"))
    gates = {
        "schema_valid": {"passed": bool(validation["accepted"])},
        "min_rows": {"passed": validation["rows"] >= min_rows, "actual": validation["rows"], "threshold": min_rows},
        "candidate_actions_present": {"passed": candidate_rows == validation["rows"] if validation["rows"] else False, "actual": candidate_rows},
        "chosen_action_present": {"passed": chosen_rows == validation["rows"] if validation["rows"] else False, "actual": chosen_rows},
        "execution_present": {"passed": (executed_rows > 0) if require_execution else True, "actual": executed_rows},
    }
    return {
        "passed": all(item["passed"] for item in gates.values()),
        "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
        "path": str(path),
        "rows": validation["rows"],
        "executed_rows": executed_rows,
        "chosen_rows": chosen_rows,
        "candidate_rows": candidate_rows,
        "validation": validation,
        "gates": gates,
    }


def _extract_runtime_rows(db: Path, *, limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        source_snapshot = _source_snapshot(conn, db)
        tasks = {str(row["task_id"]): _decode_row(row) for row in conn.execute("SELECT * FROM tasks")}
        decisions = [_decode_row(row) for row in conn.execute("SELECT * FROM action_decisions ORDER BY decision_id")]
        executions = [_decode_row(row) for row in conn.execute("SELECT * FROM execution_results ORDER BY result_id")]
        traces = [_decode_row(row) for row in conn.execute("SELECT * FROM traces ORDER BY trace_id")]
        experiences, experience_candidates = _load_experience_rows(conn)

    execution_by_task = _group_executions(executions)
    traces_by_task = defaultdict(list)
    for trace in traces:
        traces_by_task[str(trace["task_id"])].append(trace)
    experience_by_task = _group_experiences(experiences)
    candidates_by_experience = _group_experience_candidates(experience_candidates)

    rows = []
    skipped_missing_task = 0
    for decision in decisions:
        task_id = str(decision["task_id"])
        task = tasks.get(task_id)
        if task is None:
            skipped_missing_task += 1
            continue
        chosen = decision.get("chosen_action_json")
        action_id = str(chosen.get("action_id")) if isinstance(chosen, dict) and chosen.get("action_id") else None
        execution = _pick_execution(execution_by_task.get(task_id, []), action_id)
        task_traces = traces_by_task.get(task_id, [])
        experience = _pick_experience(experience_by_task.get(task_id, []), execution)
        rows.append(_runtime_row(task, decision, execution, task_traces, experience, candidates_by_experience))
        if limit is not None and len(rows) >= limit:
            break
    extraction = {
        "source_snapshot": source_snapshot,
        "decisions_seen": len(decisions),
        "experiences_seen": len(experiences),
        "experience_candidates_seen": len(experience_candidates),
        "rows_exported": len(rows),
        "skipped_missing_task": skipped_missing_task,
        "limit": limit,
    }
    return rows, extraction


def _runtime_row(
    task: dict[str, Any],
    decision: dict[str, Any],
    execution: dict[str, Any] | None,
    traces: list[dict[str, Any]],
    experience: dict[str, Any] | None = None,
    candidates_by_experience: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    task_spec = _dict_or_empty(task.get("task_spec_json"))
    chosen_action = _dict_or_none(decision.get("chosen_action_json"))
    candidate_actions = _list_or_empty(decision.get("candidate_actions_json"))
    safety_decision = _dict_or_empty(decision.get("safety_decision_json"))
    gate_trace = _dict_or_empty(decision.get("gate_trace_json"))
    model_score = _dict_or_empty(decision.get("model_score_json"))
    execution_payload = _execution_payload(execution)
    failure_trace = traces[-1] if traces else None
    success = bool(execution_payload.get("success")) if execution_payload else False
    row_id = _row_id(task, decision, execution)
    experience_payload = _experience_payload(experience)
    decision_status = _decision_time_status(task, experience_payload)
    candidate_outcomes = _candidate_outcomes(experience, candidates_by_experience or {})
    return {
        "row_id": row_id,
        "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
        "lineage": {
            "task_id": task["task_id"],
            "decision_id": decision.get("decision_id"),
            "result_id": execution.get("result_id") if execution else None,
            "experience_id": experience.get("experience_id") if experience else None,
            "trace_ids": [trace.get("trace_id") for trace in traces],
        },
        "experience": experience_payload,
        "task": {
            "task_id": task["task_id"],
            "created_at": task.get("created_at"),
            "source": task.get("source"),
            "target": task.get("target"),
            "status": decision_status,
            "final_status": task.get("status"),
            "risk_level": task.get("risk_level"),
            "requires_approval": bool(task.get("requires_approval")),
            "goal_redacted": redact_text(str(task.get("goal") or ""), max_chars=500),
            "allowed_actions": list(task_spec.get("allowed_actions") or []),
            "success_criteria": list(task_spec.get("success_criteria") or []),
        },
        "decision": {
            "step": int(decision.get("step") or 0),
            "candidate_actions": candidate_actions,
            "chosen_action": chosen_action,
            "safety_decision": safety_decision,
            "model_score": model_score,
            "gate_trace": gate_trace,
            "created_at": decision.get("created_at"),
        },
        "candidate_outcomes": candidate_outcomes,
        "execution": execution_payload,
        "outcome": {
            "success": success,
            "reward": 1.0 if success else -1.0,
            "failure_bucket": failure_trace.get("failure_bucket") if failure_trace else None,
            "failure_reason": _failure_reason(failure_trace, execution_payload),
        },
    }


def _load_experience_rows(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _table_exists(conn, "experiences") or not _table_exists(conn, "experience_candidates"):
        return [], []
    experiences = [_decode_row(row) for row in conn.execute("SELECT * FROM experiences ORDER BY created_at, experience_id")]
    candidates = [_decode_row(row) for row in conn.execute("SELECT * FROM experience_candidates ORDER BY experience_id, candidate_index, id")]
    return experiences, candidates


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def _group_experiences(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return grouped


def _group_experience_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experience_id"])].append(row)
    return grouped


def _pick_experience(rows: list[dict[str, Any]], execution: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rows:
        return None
    if execution is not None:
        for row in reversed(rows):
            if str(row.get("phase")) == "run" and str(row.get("status")) in {"completed", "failed"}:
                return row
    for row in reversed(rows):
        if str(row.get("phase")) == "run":
            return row
    return rows[-1]


def _experience_payload(experience: dict[str, Any] | None) -> dict[str, Any]:
    if not experience:
        return {
            "experience_aligned": False,
            "model_used": False,
            "model_unavailable_reason": "experience_log_missing",
        }
    return {
        "experience_aligned": True,
        "experience_id": experience.get("experience_id"),
        "phase": experience.get("phase"),
        "status": experience.get("status"),
        "decision_policy": experience.get("decision_policy"),
        "model_used": bool(experience.get("model_used")),
        "model_unavailable_reason": experience.get("model_unavailable_reason"),
        "before_state": _bounded_json(experience.get("before_state_json")),
        "after_state": _bounded_json(experience.get("after_state_json")),
        "learning_masks": _bounded_json(experience.get("learning_masks_json")),
    }


def _decision_time_status(task: dict[str, Any], experience_payload: dict[str, Any]) -> str:
    before = experience_payload.get("before_state") if isinstance(experience_payload.get("before_state"), dict) else {}
    status = before.get("task_status") or before.get("phase") or experience_payload.get("phase")
    return str(status or "deciding")


def _candidate_outcomes(experience: dict[str, Any] | None, grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not experience:
        return []
    rows = grouped.get(str(experience.get("experience_id")), [])
    result = []
    for row in rows:
        outcome = _dict_or_empty(row.get("outcome_json"))
        mask = _dict_or_empty(row.get("target_mask_json"))
        result.append(
            {
                "action_id": row.get("action_id"),
                "candidate_index": int(row.get("candidate_index") or 0),
                "params": _bounded_json(row.get("params_json")),
                "safety_decision": _dict_or_empty(row.get("safety_decision_json")),
                "model_score": _dict_or_empty(row.get("model_score_json")),
                "selected": bool(row.get("selected")),
                "executed": bool(row.get("executed")),
                "execution_result_known": bool(row.get("execution_result_known")),
                "outcome": outcome,
                "target_mask": mask,
            }
        )
    return result


def _execution_payload(execution: dict[str, Any] | None) -> dict[str, Any]:
    if execution is None:
        return {}
    return {
        "action_id": execution.get("action_id"),
        "started_at": execution.get("started_at"),
        "ended_at": execution.get("ended_at"),
        "success": bool(execution.get("success")),
        "stdout_redacted": redact_text(str(execution.get("stdout_redacted") or ""), max_chars=1_000),
        "stderr_redacted": redact_text(str(execution.get("stderr_redacted") or ""), max_chars=1_000),
        "result": _bounded_json(execution.get("result_json")),
        "error_type": execution.get("error_type"),
    }


def _failure_reason(trace: dict[str, Any] | None, execution: dict[str, Any]) -> str | None:
    if trace is not None:
        analysis = _dict_or_empty(trace.get("analysis_json"))
        reason = analysis.get("reason") or analysis.get("error") or trace.get("failure_bucket")
        return redact_text(str(reason), max_chars=500) if reason else None
    error_type = execution.get("error_type")
    return str(error_type) if error_type else None


def _manifest(db: Path, out: Path, rows: list[dict[str, Any]], *, extraction: dict[str, Any], run_id: str) -> dict[str, Any]:
    actions = sorted({str(row["execution"].get("action_id")) for row in rows if row.get("execution") and row["execution"].get("action_id")})
    statuses: dict[str, int] = defaultdict(int)
    for row in rows:
        statuses[str(row.get("task", {}).get("status") or "unknown")] += 1
    return {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
        "dataset_profile": "neurokernel-runtime-action-log-v1",
        "source_db": str(db),
        "out": str(out),
        "rows": len(rows),
        "actions": actions,
        "task_status_counts": dict(sorted(statuses.items())),
        "success_rows": sum(1 for row in rows if row["outcome"]["success"] is True),
        "failure_rows": sum(1 for row in rows if row["outcome"]["success"] is False),
        "extraction": extraction,
    }


def _validate_row(index: int, row: Any) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(row, dict):
        return [{"row": index, "error": "row must be object"}]
    if row.get("schema_version") != RUNTIME_REPLAY_SCHEMA_VERSION:
        errors.append({"row": index, "error": "schema_version mismatch"})
    for field in ("lineage", "task", "decision", "execution", "outcome"):
        if not isinstance(row.get(field), dict):
            errors.append({"row": index, "error": f"{field} must be object"})
    if not row.get("row_id"):
        errors.append({"row": index, "error": "row_id is required"})
    task = row.get("task") if isinstance(row.get("task"), dict) else {}
    decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    if not task.get("task_id"):
        errors.append({"row": index, "error": "task.task_id is required"})
    if not isinstance(decision.get("candidate_actions"), list):
        errors.append({"row": index, "error": "decision.candidate_actions must be array"})
    if "success" not in outcome:
        errors.append({"row": index, "error": "outcome.success is required"})
    secret_path = _find_forbidden_key(row)
    if secret_path:
        errors.append({"row": index, "error": f"forbidden key path: {secret_path}"})
    return errors


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key, value in list(data.items()):
        if key.endswith("_json") and isinstance(value, str):
            data[key] = _load_json(value)
    return data


def _group_executions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    return grouped


def _pick_execution(rows: list[dict[str, Any]], action_id: str | None) -> dict[str, Any] | None:
    if not rows:
        return None
    if action_id:
        for row in reversed(rows):
            if str(row.get("action_id")) == action_id:
                return row
    return rows[-1]


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    replay = Path(path)
    if not replay.exists():
        raise FileNotFoundError(f"runtime replay not found: {replay}")
    rows = []
    with replay.open("r", encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _source_snapshot(conn: sqlite3.Connection, db: Path) -> dict[str, Any]:
    tables = ("tasks", "action_decisions", "execution_results", "traces", "work_items", "work_jobs", "agent_events")
    counts = {}
    for table in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
        except sqlite3.Error:
            counts[table] = None
    stat = db.stat()
    return {
        "path": str(db),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "table_counts": counts,
    }


def _load_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bounded_json(value: Any) -> Any:
    text = json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)
    if len(text) <= 4_000:
        return value if value is not None else {}
    return {"truncated": True, "preview_redacted": redact_text(text, max_chars=4_000)}


def _row_id(task: dict[str, Any], decision: dict[str, Any], execution: dict[str, Any] | None) -> str:
    raw = {
        "task_id": task.get("task_id"),
        "decision_id": decision.get("decision_id"),
        "result_id": execution.get("result_id") if execution else None,
    }
    digest = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"runtime_{digest}"


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(f".{path.name}.{_new_run_id()}.tmp")
    _write_jsonl(tmp, rows)
    tmp.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{_new_run_id()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _find_forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_KEY_EXACT or any(fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS):
                return f"{path}.{key}"
            found = _find_forbidden_key(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None
