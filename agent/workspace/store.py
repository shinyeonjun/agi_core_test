from __future__ import annotations

import json
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db


def record_workspace_artifact(artifact_type: str, title: str, relative_path: str, size_bytes: int, metadata: dict[str, Any] | None = None) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO workspace_artifacts (created_at, artifact_type, title, relative_path, bytes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now_kst(), artifact_type, title, relative_path, size_bytes, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_workspace_artifacts(limit: int = 20, artifact_type: str | None = None) -> list[dict[str, Any]]:
    init_db()
    if artifact_type:
        query = "SELECT * FROM workspace_artifacts WHERE artifact_type = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (artifact_type, limit)
    else:
        query = "SELECT * FROM workspace_artifacts ORDER BY id DESC LIMIT ?"
        params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def record_project_spec(title: str, objective: str, artifact_id: int, spec: dict[str, Any], status: str = "draft") -> int:
    init_db()
    ts = now_kst()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO project_specs (created_at, updated_at, title, objective, status, artifact_id, spec_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, ts, title, objective, status, artifact_id, json.dumps(spec, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_project_specs(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    if status:
        query = "SELECT * FROM project_specs WHERE status = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (status, limit)
    else:
        query = "SELECT * FROM project_specs ORDER BY id DESC LIMIT ?"
        params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def count_workspace_artifacts() -> int:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS count FROM workspace_artifacts").fetchone()
    return int(row["count"] if row else 0)
