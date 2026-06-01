from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9), "KST")


def now_kst() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def project_root() -> Path:
    return Path(os.environ.get("AGENT_CORE_HOME", "/home/ubuntu/agent_core")).expanduser().resolve()


def data_dir() -> Path:
    return project_root() / "data"


def logs_dir() -> Path:
    return project_root() / "logs"


def db_path() -> Path:
    return Path(os.environ.get("AGENT_CORE_DB_PATH", data_dir() / "agent.db")).expanduser().resolve()


def state_path() -> Path:
    return Path(os.environ.get("AGENT_CORE_STATE_PATH", data_dir() / "state.json")).expanduser().resolve()


def ensure_runtime_dirs() -> None:
    for path in [data_dir(), logs_dir(), project_root() / "renderer_workspace", data_dir() / "backups"]:
        path.mkdir(parents=True, exist_ok=True)
