from __future__ import annotations

import shutil
from pathlib import Path

from agent.config.defaults import backup_dir, db_path, now_kst, state_path


def create_backup(label: str = "manual") -> dict[str, str]:
    backup_dir().mkdir(parents=True, exist_ok=True)
    stamp = now_kst().replace(":", "").replace("+", "_")
    result: dict[str, str] = {}
    if db_path().exists():
        target = backup_dir() / f"agent_{label}_{stamp}.db"
        shutil.copy2(db_path(), target)
        result["db"] = str(target)
    if state_path().exists():
        target = backup_dir() / f"state_{label}_{stamp}.json"
        shutil.copy2(state_path(), target)
        result["state"] = str(target)
    return result
