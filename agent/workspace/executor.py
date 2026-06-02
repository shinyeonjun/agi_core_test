from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.config.defaults import now_kst, workspace_dirs, workspace_root
from agent.core.events import log_event
from agent.workspace.store import record_project_spec, record_workspace_artifact

ALLOWED_SECTIONS = {"scratch", "outputs", "tasks", "reports", "projects", "experiments"}
MAX_TEXT_BYTES = 200_000
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
    re.compile(r"(?im)^\s*[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|API[_-]?KEY)[A-Z0-9_]*\s*="),
)


def ensure_workspace() -> dict[str, Any]:
    root = workspace_root()
    for path in workspace_dirs():
        path.mkdir(parents=True, exist_ok=True)
    return {"root": str(root), "dirs": [str(path) for path in workspace_dirs()]}


def _slug(text: str) -> str:
    value = re.sub(r"[^0-9A-Za-z?-?_.-]+", "-", text.strip()).strip("-._").lower()
    return value[:64] or "artifact"


def _assert_safe_content(content: str) -> None:
    if len(content.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError("workspace content too large")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        raise ValueError("workspace content contains secret-like text")


def safe_workspace_path(section: str, relative_path: str) -> Path:
    if section not in ALLOWED_SECTIONS:
        raise ValueError(f"workspace section not allowed: {section}")
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("workspace path must be relative")
    candidate = (workspace_root() / section / relative_path).resolve()
    base = (workspace_root() / section).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError("workspace path escapes allowed section")
    return candidate


def write_text_artifact(section: str, relative_path: str, content: str, artifact_type: str = "file", title: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_workspace()
    _assert_safe_content(content)
    path = safe_workspace_path(section, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    rel = path.relative_to(workspace_root()).as_posix()
    size = path.stat().st_size
    artifact_id = record_workspace_artifact(artifact_type, title or path.name, rel, size, metadata)
    log_event("workspace", "workspace_artifact_created", rel, {"artifact_id": artifact_id, "artifact_type": artifact_type}, 0.5)
    return {"id": artifact_id, "path": str(path), "relative_path": rel, "bytes": size, "artifact_type": artifact_type}


def create_status_report(title: str = "Workspace status report", metrics: dict[str, Any] | None = None, drives: dict[str, Any] | None = None) -> dict[str, Any]:
    stamp = now_kst().replace(":", "").replace("+", "_")
    filename = f"{stamp}-{_slug(title)}.md"
    content = [f"# {title}", "", f"created_at: {now_kst()}", "", "## Metrics", ""]
    for key, value in (metrics or {}).items():
        content.append(f"- {key}: {value}")
    content.extend(["", "## Drives", ""] )
    for key, value in (drives or {}).items():
        content.append(f"- {key}: {value}")
    content.append("")
    return write_text_artifact("reports", filename, "\n".join(content), "report", title, {"source": "workspace_executor"})


def create_project_spec(title: str, objective: str, notes: list[str] | None = None) -> dict[str, Any]:
    stamp = now_kst().replace(":", "").replace("+", "_")
    slug = _slug(title)
    spec = {
        "version": "0.11",
        "title": title,
        "objective": objective,
        "status": "draft",
        "created_at": now_kst(),
        "notes": notes or [],
        "safety": {
            "workspace_only": True,
            "os_changes": False,
            "agent_core_direct_patch": False,
            "secret_export": False,
        },
    }
    artifact = write_text_artifact(
        "projects",
        f"{stamp}-{slug}/project_spec.json",
        json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        "project_spec",
        title,
        {"source": "project_builder"},
    )
    project_id = record_project_spec(title, objective, int(artifact["id"]), spec)
    artifact["project_id"] = project_id
    return artifact
