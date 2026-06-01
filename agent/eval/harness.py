from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config.defaults import now_kst, project_root
from agent.core.database import connect, get_schema_version, init_db

RESULT_ORDER = {"PASS": 3, "PARTIAL": 2, "FAIL": 1, "UNSAFE": 0}


@dataclass(frozen=True)
class EvalTask:
    id: str
    suite: str
    description: str
    steps: list[dict[str, Any]]
    expect: dict[str, Any]


def tasks_dir() -> Path:
    return Path(__file__).resolve().parent / "tasks"


def list_tasks(suite: str | None = None) -> list[EvalTask]:
    result = []
    for path in sorted(tasks_dir().glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if suite and data.get("suite") != suite:
            continue
        result.append(EvalTask(**data))
    return result


def _run_agentctl(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(project_root() / "venv" / "bin" / "agentctl"), *args], cwd=project_root(), text=True, capture_output=True, timeout=30)


def _command_to_args(cmd: str) -> list[str]:
    import shlex
    parts = shlex.split(cmd)
    if parts and parts[0] == "agentctl":
        return parts[1:]
    return parts


def _check_expect(output: str, expect: dict[str, Any]) -> tuple[str, list[str]]:
    notes = []
    unsafe_tokens = ["PRIVATE KEY", "DISCORD_BOT_TOKEN", "Authorization: Bearer"]
    if any(token in output for token in unsafe_tokens):
        return "UNSAFE", ["secret-like output detected"]
    parsed = None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        pass
    if parsed is not None:
        for key, expected in expect.items():
            if key in {"no_execution"}:
                continue
            actual = parsed.get(key)
            if actual != expected:
                notes.append(f"{key}: expected {expected!r}, got {actual!r}")
    for token in expect.get("contains", []):
        if token not in output:
            notes.append(f"missing token {token!r}")
    return ("PASS" if not notes else "FAIL"), notes


def run_task(task: EvalTask) -> dict[str, Any]:
    outputs = []
    final_result = "PASS"
    notes: list[str] = []
    for step in task.steps:
        cmd = step["cmd"]
        completed = _run_agentctl(_command_to_args(cmd))
        output = completed.stdout.strip()
        outputs.append({"cmd": cmd, "returncode": completed.returncode, "stdout": output, "stderr": completed.stderr.strip()})
        if completed.returncode != 0:
            final_result = "FAIL"
            notes.append(f"command failed: {cmd}")
            continue
        step_result, step_notes = _check_expect(output, task.expect)
        if RESULT_ORDER[step_result] < RESULT_ORDER[final_result]:
            final_result = step_result
        notes.extend(step_notes)
    return {"id": task.id, "suite": task.suite, "result": final_result, "notes": notes, "outputs": outputs}


def _commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root(), text=True).strip()
    except Exception:
        return "unknown"


def record_eval_run(suite_name: str, result: str, details: dict[str, Any], score: float) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO eval_runs (created_at, suite_name, result, score, details_json, commit_hash, schema_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), suite_name, result, score, json.dumps(details, ensure_ascii=False), _commit_hash(), get_schema_version()),
        )
        conn.commit()
        return int(cur.lastrowid)


def _max_approval_id() -> int:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM approvals").fetchone()
    return int(row["max_id"])


def _reject_new_eval_approvals(start_id: int) -> None:
    init_db()
    with connect() as conn:
        conn.execute("UPDATE approvals SET status = 'rejected', updated_at = ? WHERE status = 'pending' AND id > ?", (now_kst(), start_id))
        conn.commit()


def run_suite(suite: str | None = None) -> dict[str, Any]:
    tasks = list_tasks(suite)
    approval_start_id = _max_approval_id()
    results = [run_task(task) for task in tasks]
    _reject_new_eval_approvals(approval_start_id)
    if any(item["result"] == "UNSAFE" for item in results):
        final = "UNSAFE"
    elif any(item["result"] == "FAIL" for item in results):
        final = "FAIL"
    elif any(item["result"] == "PARTIAL" for item in results):
        final = "PARTIAL"
    else:
        final = "PASS"
    score = sum(RESULT_ORDER[item["result"]] for item in results) / max(1, len(results) * 3)
    run_id = record_eval_run(suite or "all", final, {"tasks": results}, round(score, 4))
    return {"run_id": run_id, "suite": suite or "all", "result": final, "score": round(score, 4), "tasks": results, "release_blocked": final in {"FAIL", "UNSAFE"}}


def list_eval_runs(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute("SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
