from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
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


def backup_dir() -> Path:
    return data_dir() / "backups"


def renderer_workspace() -> Path:
    return project_root() / "renderer_workspace"


def workspace_root() -> Path:
    return Path(os.environ.get("AGENT_WORKSPACE_ROOT", "/home/ubuntu/agent_workspace")).expanduser().resolve()


def workspace_dirs() -> list[Path]:
    root = workspace_root()
    return [root / name for name in ["scratch", "outputs", "tasks", "reports", "projects", "experiments"]]


def env_path() -> Path:
    return Path(os.environ.get("AGENT_CORE_ENV_PATH", project_root() / ".env")).expanduser().resolve()


def db_path() -> Path:
    return Path(os.environ.get("AGENT_CORE_DB_PATH", data_dir() / "agent.db")).expanduser().resolve()


def state_path() -> Path:
    return Path(os.environ.get("AGENT_CORE_STATE_PATH", data_dir() / "state.json")).expanduser().resolve()


def ensure_runtime_dirs() -> None:
    for path in [data_dir(), logs_dir(), renderer_workspace(), backup_dir()]:
        path.mkdir(parents=True, exist_ok=True)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_csv(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}
