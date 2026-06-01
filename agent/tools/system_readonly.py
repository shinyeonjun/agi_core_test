from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from agent.config.defaults import now_kst, project_root
from agent.core.database import connect, init_db
from agent.core.events import log_event

MAX_RAW_OUTPUT = 100000


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: list[str]
    risk_level: str = "medium"
    requires_approval: bool = False
    timeout_seconds: int = 15
    capture_stdout: bool = True
    redact_patterns: list[str] = field(default_factory=list)


READ_ONLY_COMMANDS = {
    "disk": CommandSpec("disk", ["df", "-h", "/"]),
    "memory": CommandSpec("memory", ["free", "-h"]),
    "swap": CommandSpec("swap", ["swapon", "--show"]),
    "zram": CommandSpec("zram", ["zramctl"]),
    "uptime": CommandSpec("uptime", ["uptime"]),
    "failed_services": CommandSpec("failed_services", ["systemctl", "--failed", "--no-pager"]),
    "journal_usage": CommandSpec("journal_usage", ["journalctl", "--disk-usage"]),
    "git_status": CommandSpec("git_status", ["git", "status", "--short"]),
    "git_log": CommandSpec("git_log", ["git", "log", "--oneline", "-5"]),
}


def redact_output(text: str) -> str:
    text = re.sub(r"(?i)(token|api[_-]?key|authorization:\s*bearer)\s*[:=]\s*[^\s]+", r"\1=<redacted>", text)
    if "PRIVATE KEY" in text:
        log_event("tool", "unsafe_output_blocked", "private key pattern detected", {}, 1.0)
        return "<unsafe output blocked>"
    return text[:MAX_RAW_OUTPUT]


def run_readonly(name: str) -> dict[str, Any]:
    if name not in READ_ONLY_COMMANDS:
        raise ValueError(f"not allowed read-only command: {name}")
    spec = READ_ONLY_COMMANDS[name]
    try:
        completed = subprocess.run(spec.argv, cwd=project_root(), text=True, capture_output=True, timeout=spec.timeout_seconds, check=False)
        stdout = redact_output(completed.stdout)
        stderr = redact_output(completed.stderr)
        returncode = completed.returncode
        error = None
    except FileNotFoundError as exc:
        stdout = ""
        stderr = redact_output(str(exc))
        returncode = 127
        error = "command_not_found"
    except subprocess.TimeoutExpired as exc:
        stdout = redact_output(exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = redact_output(exc.stderr or "") if isinstance(exc.stderr, str) else ""
        returncode = 124
        error = "timeout"
    result = {"name": name, "argv": spec.argv, "returncode": returncode, "stdout": stdout, "stderr": stderr, "risk_level": spec.risk_level, "requires_approval": spec.requires_approval, "error": error}
    init_db()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO tool_runs (ts, tool_name, action, risk_level, approved, success, result_summary, raw_output, metadata_json)
            VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
            """,
            (now_kst(), "system_readonly", name, spec.risk_level, 1 if returncode == 0 else 0, f"rc={returncode}", stdout, json.dumps({"stderr": stderr, "error": error}, ensure_ascii=False)),
        )
        conn.commit()
    return result


def system_snapshot() -> dict[str, Any]:
    outputs = {name: run_readonly(name) for name in ["disk", "memory", "swap", "uptime", "failed_services"]}
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO system_snapshots (ts, raw_json) VALUES (?, ?)",
            (now_kst(), json.dumps(outputs, ensure_ascii=False)),
        )
        conn.commit()
    return {"snapshot_id": int(cur.lastrowid), "outputs": outputs}
