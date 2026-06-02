from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config.defaults import KST, env_path, now_kst, project_root, workspace_root
from agent.core.autonomy import get_autonomy_state
from agent.core.database import connect, get_schema_version, init_db
from agent.core.events import log_event

SELF_MAP_UNITS = (
    "agent-core-discord.service",
    "agent-core-tick.timer",
    "agent-core-summary.timer",
    "agent-core-daily-summary.timer",
    "agent-core-lab-tick.timer",
)

CONFIG_PRESENCE_KEYS = (
    "DISCORD_BOT_TOKEN",
    "DISCORD_CHAT_CHANNEL_ID",
    "DISCORD_APPROVAL_CHANNEL_ID",
    "DISCORD_SUMMARY_WEBHOOK_URL",
    "DISCORD_UPDATE_WEBHOOK_URL",
    "AGENT_CHAT_RENDERER",
    "AGENT_LANGUAGE_ENGINE",
    "AGENT_CODEX_LANGUAGE_MODEL",
    "AGENT_CODEX_RENDERER_MODEL",
    "AGENT_CODEX_WORK_MODEL",
)


def _run(args: list[str], *, cwd: Path | None = None, timeout: float = 2.0) -> dict[str, Any]:
    if not args or shutil.which(args[0]) is None:
        return {"available": False, "returncode": None, "stdout": "", "stderr": "command_not_found"}
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"available": True, "returncode": None, "stdout": "", "stderr": "timeout"}
    except OSError as exc:
        return {"available": False, "returncode": None, "stdout": "", "stderr": type(exc).__name__}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "").strip()[:1000],
        "stderr": (completed.stderr or "").strip()[:1000],
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"NAME", "VERSION_ID", "PRETTY_NAME", "ID"}:
                values[key.lower()] = value.strip().strip('"')[:120]
    except OSError:
        return {}
    return values


def _git_info(root: Path) -> dict[str, Any]:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    commit = _run(["git", "rev-parse", "--short", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root)
    if not branch["available"]:
        return {"available": False}
    return {
        "available": True,
        "branch": branch["stdout"] if branch["returncode"] == 0 else None,
        "commit": commit["stdout"] if commit["returncode"] == 0 else None,
        "dirty": bool(status["stdout"]) if status["returncode"] == 0 else None,
    }


def _systemd_units() -> dict[str, str]:
    if shutil.which("systemctl") is None:
        return {"systemd": "unavailable"}
    units: dict[str, str] = {}
    for unit in SELF_MAP_UNITS:
        result = _run(["systemctl", "is-active", unit], timeout=1.5)
        units[unit] = result["stdout"] if result["returncode"] in {0, 3} and result["stdout"] else "unknown"
    return units


def _latest_eval(conn) -> dict[str, Any] | None:
    row = conn.execute("SELECT id, created_at, result, score FROM eval_runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def _row_counts(conn) -> dict[str, int]:
    tables = ("events", "memories", "goals", "reflections", "action_runs", "renderer_runs", "decisions")
    counts: dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        counts[table] = int(row["count"] if row else 0)
    return counts


def collect_self_map() -> dict[str, Any]:
    init_db()
    root = project_root()
    workspace = workspace_root()
    with connect() as conn:
        latest_eval = _latest_eval(conn)
        row_counts = _row_counts(conn)
    snapshot = {
        "created_at": now_kst(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "os_release": _read_os_release(),
        },
        "paths": {
            "project_root": str(root),
            "workspace_root": str(workspace),
            "env_file_exists": env_path().exists(),
        },
        "git": _git_info(root),
        "codex_cli": _run(["codex", "--version"], timeout=2.0),
        "services": _systemd_units(),
        "autonomy": get_autonomy_state(),
        "config_presence": {key: bool(os.environ.get(key)) for key in CONFIG_PRESENCE_KEYS},
        "database": {
            "schema_version": get_schema_version(),
            "row_counts": row_counts,
            "latest_eval": latest_eval,
        },
    }
    fingerprint_source = {
        "host": snapshot["host"],
        "paths": snapshot["paths"],
        "git": snapshot["git"],
        "services": snapshot["services"],
        "autonomy_profile": snapshot["autonomy"].get("autonomy_profile"),
        "config_presence": snapshot["config_presence"],
        "schema_version": snapshot["database"]["schema_version"],
    }
    snapshot["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    snapshot["summary"] = summarize_snapshot(snapshot)
    return snapshot


def summarize_snapshot(snapshot: dict[str, Any]) -> str:
    os_release = snapshot.get("host", {}).get("os_release") or {}
    os_name = os_release.get("pretty_name") or snapshot.get("host", {}).get("platform") or "unknown"
    git = snapshot.get("git") or {}
    services = snapshot.get("services") or {}
    active_units = [name for name, state in services.items() if state == "active"]
    eval_row = (snapshot.get("database") or {}).get("latest_eval") or {}
    profile = (snapshot.get("autonomy") or {}).get("autonomy_profile", "unknown")
    commit = git.get("commit") or "unknown"
    dirty = "dirty" if git.get("dirty") else "clean"
    eval_text = f"{eval_row.get('result')} / {eval_row.get('score')}" if eval_row else "none"
    return (
        f"{os_name}; profile={profile}; git={commit} {dirty}; "
        f"active_services={len(active_units)}; latest_eval={eval_text}"
    )


def _latest_row(conn) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM self_maps ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def refresh_self_map(*, record_event: bool = True) -> dict[str, Any]:
    snapshot = collect_self_map()
    summary = str(snapshot["summary"])
    fingerprint = str(snapshot["fingerprint"])
    with connect() as conn:
        previous = _latest_row(conn)
        changed = previous is None or previous.get("fingerprint") != fingerprint
        cur = conn.execute(
            """
            INSERT INTO self_maps (created_at, summary, snapshot_json, fingerprint, changed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (snapshot["created_at"], summary, json.dumps(snapshot, ensure_ascii=False), fingerprint, 1 if changed else 0),
        )
        conn.commit()
        row_id = int(cur.lastrowid)
    result = {"id": row_id, "created_at": snapshot["created_at"], "changed": changed, "fingerprint": fingerprint, "summary": summary, "snapshot": snapshot}
    if record_event:
        event_type = "self_map_changed" if changed else "self_map_refreshed"
        log_event("self_map", event_type, summary, {"self_map_id": row_id, "changed": changed}, 0.72 if changed else 0.35)
    return result


def latest_self_map() -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = _latest_row(conn)
    if not row:
        return None
    snapshot = json.loads(str(row["snapshot_json"]))
    return {**row, "snapshot": snapshot}


def _active_service_count(item: dict[str, Any] | None) -> int:
    if not item:
        return 0
    services = ((item.get("snapshot") or {}).get("services") or {})
    return len([state for state in services.values() if state == "active"])


def _age_seconds(item: dict[str, Any] | None) -> float | None:
    if not item or not item.get("created_at"):
        return None
    try:
        created = datetime.fromisoformat(str(item["created_at"]))
    except ValueError:
        return None
    return max(0.0, (datetime.now(created.tzinfo or KST) - created).total_seconds())


def latest_self_map_fresh(*, max_age_seconds: int = 300, refresh_if_stale: bool = True, record_event_on_refresh: bool = True) -> dict[str, Any] | None:
    latest = latest_self_map()
    age = _age_seconds(latest)
    stale = latest is None or age is None or age > max(1, int(max_age_seconds))
    suspicious_age = max(60, int(max_age_seconds) // 2)
    suspicious = latest is not None and age is not None and age > suspicious_age and _active_service_count(latest) == 0 and shutil.which("systemctl") is not None
    if refresh_if_stale and (stale or suspicious):
        return refresh_self_map(record_event=record_event_on_refresh and (stale or suspicious))
    return latest


def self_map_brief(*, max_age_seconds: int | None = None, refresh_if_stale: bool = False, record_event_on_refresh: bool = True) -> dict[str, Any] | None:
    latest = (
        latest_self_map_fresh(max_age_seconds=max_age_seconds, refresh_if_stale=refresh_if_stale, record_event_on_refresh=record_event_on_refresh)
        if max_age_seconds is not None
        else latest_self_map()
    )
    if not latest:
        return None
    snapshot = latest.get("snapshot") or latest
    host = snapshot.get("host") or {}
    os_release = host.get("os_release") or {}
    return {
        "id": latest.get("id"),
        "created_at": latest.get("created_at"),
        "summary": latest.get("summary"),
        "os": os_release.get("pretty_name") or host.get("platform"),
        "hostname": host.get("hostname"),
        "architecture": host.get("machine"),
        "project_root": (snapshot.get("paths") or {}).get("project_root"),
        "workspace_root": (snapshot.get("paths") or {}).get("workspace_root"),
        "git": snapshot.get("git"),
        "services": snapshot.get("services"),
        "autonomy_profile": (snapshot.get("autonomy") or {}).get("autonomy_profile"),
        "latest_eval": (snapshot.get("database") or {}).get("latest_eval"),
    }
