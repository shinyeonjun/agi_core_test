from __future__ import annotations

import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.autonomy import current_profile
from agent.core.database import connect, init_db
from agent.core.drives import compute_drives
from agent.core.events import list_events, log_event
from agent.core.goals import create_goal, is_noise_goal_record, list_goals, similar
from agent.core.learner import list_reflections
from agent.core.objective_registry import ALLOWED_GENERATED_GOAL_TYPES, DEFAULT_OBJECTIVES, TEMPLATES
from agent.core.task_queue import enqueue_task
from agent.tools.action_log import list_action_runs

OPEN_STATUSES = {"proposed", "active", "waiting_approval", "blocked"}


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata_json")
    try:
        row["metadata"] = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        row["metadata"] = {"decode_error": True}
    return row


def seed_default_objectives() -> list[dict[str, Any]]:
    init_db()
    rows_out: list[dict[str, Any]] = []
    with connect() as conn:
        ts = now_kst()
        for item in DEFAULT_OBJECTIVES:
            row = conn.execute("SELECT id FROM root_objectives WHERE objective_type = ?", (item["objective_type"],)).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE root_objectives
                    SET title = ?, description = ?, priority = ?, cooldown_seconds = ?, updated_at = ?
                    WHERE objective_type = ?
                    """,
                    (item["title"], item["description"], item["priority"], item["cooldown_seconds"], ts, item["objective_type"]),
                )
                objective_id = int(row["id"])
            else:
                cur = conn.execute(
                    """
                    INSERT INTO root_objectives (
                        created_at, updated_at, title, description, objective_type,
                        priority, enabled, cooldown_seconds, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (ts, ts, item["title"], item["description"], item["objective_type"], item["priority"], item["cooldown_seconds"], _json({"seed": True})),
                )
                objective_id = int(cur.lastrowid)
            rows_out.append({"id": objective_id, "objective_type": item["objective_type"], "title": item["title"]})
        conn.commit()
    log_event("goal", "root_objectives_seeded", "default root objectives seeded", {"count": len(rows_out)}, 0.55)
    return rows_out


def add_root_objective(title: str, description: str, objective_type: str, priority: float = 0.5, cooldown_seconds: int = 21600) -> int:
    init_db()
    priority = max(0.0, min(1.0, float(priority)))
    with connect() as conn:
        ts = now_kst()
        conn.execute(
            """
            INSERT INTO root_objectives (
                created_at, updated_at, title, description, objective_type,
                priority, enabled, cooldown_seconds, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(objective_type) DO UPDATE SET
                title = excluded.title,
                description = excluded.description,
                priority = excluded.priority,
                cooldown_seconds = excluded.cooldown_seconds,
                updated_at = excluded.updated_at
            """,
            (ts, ts, title, description, objective_type, priority, cooldown_seconds, _json({"manual": True})),
        )
        row = conn.execute("SELECT id FROM root_objectives WHERE objective_type = ?", (objective_type,)).fetchone()
        conn.commit()
    return int(row["id"])


def list_root_objectives(include_disabled: bool = False, limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM root_objectives"
    if not include_disabled:
        query += " WHERE enabled = 1"
    query += " ORDER BY enabled DESC, priority DESC, id ASC LIMIT ?"
    with connect() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [_decode(dict(row)) for row in rows]


def set_objective_enabled(objective_id: int, enabled: bool) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute("UPDATE root_objectives SET enabled = ?, updated_at = ? WHERE id = ?", (1 if enabled else 0, now_kst(), objective_id))
        conn.commit()
        return cur.rowcount > 0


def list_goal_candidates(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    if status:
        query = "SELECT * FROM generated_goal_candidates WHERE status = ? ORDER BY id DESC LIMIT ?"
        params: tuple[Any, ...] = (status, limit)
    else:
        query = "SELECT * FROM generated_goal_candidates ORDER BY id DESC LIMIT ?"
        params = (limit,)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_decode(dict(row)) for row in rows]


def is_noise_goal(goal: dict[str, Any]) -> bool:
    return is_noise_goal_record(goal)


def meaningful_open_goals(limit: int = 200) -> list[dict[str, Any]]:
    init_db()
    placeholders = ",".join("?" for _ in OPEN_STATUSES)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM goals
            WHERE status IN ({placeholders})
            ORDER BY
                goal_type = 'user_directed' DESC,
                priority DESC,
                id DESC
            LIMIT ?
            """,
            (*OPEN_STATUSES, limit),
        ).fetchall()
    return [dict(row) for row in rows if not is_noise_goal(dict(row))]


def _recent_candidate_duplicate(title: str, goal_type: str, within_hours: int = 24, *, include_dry_run: bool = False) -> dict[str, Any] | None:
    cutoff = (datetime.now(KST) - timedelta(hours=within_hours)).isoformat(timespec="seconds")
    statuses = ("selected", "proposed", "dry_run") if include_dry_run else ("selected", "proposed")
    placeholders = ",".join("?" for _ in statuses)
    init_db()
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM generated_goal_candidates
            WHERE goal_type = ? AND created_at >= ? AND status IN ({placeholders})
            ORDER BY id DESC
            """,
            (goal_type, cutoff, *statuses),
        ).fetchall()
    for row in rows:
        item = dict(row)
        if similar(str(item["title"]), title) >= 0.88:
            return item
    return None


def _open_goal_duplicate(title: str, goal_type: str) -> dict[str, Any] | None:
    for goal in list_goals(limit=80):
        if goal.get("status") not in OPEN_STATUSES or str(goal.get("goal_type")) != goal_type:
            continue
        if SequenceMatcher(None, str(goal.get("title") or "").lower(), title.lower()).ratio() >= 0.86:
            return goal
    return None


def _cooldown_active(objective: dict[str, Any]) -> bool:
    last_used = objective.get("last_used_at")
    if not last_used:
        return False
    try:
        last = datetime.fromisoformat(str(last_used))
    except ValueError:
        return False
    cooldown = int(objective.get("cooldown_seconds") or 0)
    return cooldown > 0 and datetime.now(last.tzinfo or KST) < last + timedelta(seconds=cooldown)


def _record_candidate(candidate: dict[str, Any]) -> int:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO generated_goal_candidates (
                created_at, root_objective_id, title, description, goal_type,
                novelty_score, utility_score, risk_level, score, status,
                rejection_reason, generated_goal_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst(), candidate.get("root_objective_id"), candidate["title"], candidate.get("description"),
                candidate["goal_type"], candidate.get("novelty_score", 0.0), candidate.get("utility_score", 0.0),
                candidate.get("risk_level", "low"), candidate.get("score", 0.0), candidate.get("status", "candidate"),
                candidate.get("rejection_reason"), candidate.get("generated_goal_id"), _json(candidate.get("metadata")),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def _update_candidate(candidate_id: int, *, status: str, generated_goal_id: int | None = None, rejection_reason: str | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE generated_goal_candidates SET status = ?, generated_goal_id = COALESCE(?, generated_goal_id), rejection_reason = COALESCE(?, rejection_reason) WHERE id = ?",
            (status, generated_goal_id, rejection_reason, candidate_id),
        )
        conn.commit()


def _mark_objective_used(objective_id: int) -> None:
    init_db()
    with connect() as conn:
        ts = now_kst()
        conn.execute("UPDATE root_objectives SET last_used_at = ?, updated_at = ? WHERE id = ?", (ts, ts, objective_id))
        conn.commit()


def _candidate_for_objective(objective: dict[str, Any], drives: dict[str, float], context: dict[str, Any], *, include_dry_run_duplicates: bool = False) -> dict[str, Any]:
    template = TEMPLATES.get(str(objective.get("objective_type")), TEMPLATES["research_loop"])
    drive_name = str(template.get("drive") or "curiosity")
    drive_match = float(drives.get(drive_name, 0.0))
    novelty = float(template.get("novelty", 0.5))
    utility = float(template.get("utility", 0.5))
    risk_level = str(template.get("risk", "low"))
    risk_penalty = 0.12 if risk_level == "medium" else 0.0
    repetition_penalty = 0.20 if _recent_candidate_duplicate(str(template["title"]), str(template["goal_type"]), include_dry_run=include_dry_run_duplicates) else 0.0
    score = (
        float(objective.get("priority") or 0.5) * 0.30
        + drive_match * 0.25
        + novelty * 0.20
        + utility * 0.15
        + float(drives.get("user_alignment", 0.0)) * 0.10
        - risk_penalty
        - repetition_penalty
    )
    return {
        "root_objective_id": objective.get("id"),
        "objective_type": objective.get("objective_type"),
        "title": template["title"],
        "description": template["description"],
        "goal_type": template["goal_type"],
        "novelty_score": round(novelty, 4),
        "utility_score": round(utility, 4),
        "risk_level": risk_level,
        "score": round(max(0.0, min(1.0, score)), 4),
        "metadata": {
            "source": "goal_generator",
            "drive_match": drive_name,
            "drive_value": drive_match,
            "current_profile": context.get("profile"),
            "recent_events": context.get("recent_events"),
            "recent_actions": context.get("recent_actions"),
            "recent_reflections": context.get("recent_reflections"),
            "execution": "not_executed",
        },
    }


def generate_goal_candidates(*, dry_run: bool = False, max_candidates: int = 3) -> dict[str, Any]:
    seed_default_objectives()
    open_goals = [goal for goal in meaningful_open_goals() if goal.get("goal_type") != "user_directed"]
    if open_goals:
        result = {"created_goal_id": None, "generated": False, "reason": "meaningful_open_goal_exists", "open_goal_id": open_goals[0].get("id"), "candidates": []}
        log_event("goal", "goal_generation_skipped", result["reason"], result, 0.55)
        return result

    objectives = list_root_objectives(include_disabled=False, limit=20)
    drives = compute_drives()
    context = {"profile": current_profile(), "recent_events": len(list_events(10)), "recent_actions": len(list_action_runs(10)), "recent_reflections": len(list_reflections(10))}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for objective in objectives:
        if _cooldown_active(objective):
            rejected.append({"objective_id": objective.get("id"), "objective_type": objective.get("objective_type"), "reason": "cooldown"})
            continue
        candidate = _candidate_for_objective(objective, drives, context, include_dry_run_duplicates=dry_run)
        if candidate["goal_type"] not in ALLOWED_GENERATED_GOAL_TYPES:
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "goal_type_not_allowed"
            candidate["id"] = _record_candidate(candidate)
            rejected.append(candidate)
            continue
        duplicate = _open_goal_duplicate(candidate["title"], candidate["goal_type"])
        if duplicate:
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "duplicate_open_goal"
            candidate["metadata"]["duplicate_goal_id"] = duplicate.get("id")
            candidate["id"] = _record_candidate(candidate)
            rejected.append(candidate)
            continue
        recent_duplicate = _recent_candidate_duplicate(candidate["title"], candidate["goal_type"], include_dry_run=dry_run)
        if recent_duplicate:
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "duplicate_recent_candidate"
            candidate["metadata"]["duplicate_candidate_id"] = recent_duplicate.get("id")
            candidate["id"] = _record_candidate(candidate)
            rejected.append(candidate)
            continue
        candidates.append(candidate)

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[: max(1, min(3, max_candidates))]
    if not candidates:
        result = {"created_goal_id": None, "generated": False, "reason": "no_candidate", "candidates": [], "rejected": rejected[:5]}
        log_event("goal", "goal_generation_skipped", result["reason"], result, 0.55)
        return result

    status = "dry_run" if dry_run else "candidate"
    for candidate in candidates:
        candidate["status"] = status
        candidate["id"] = _record_candidate(candidate)

    selected = candidates[0]
    created_goal_id: int | None = None
    if not dry_run:
        created_goal_id = create_goal(
            selected["title"],
            selected["description"],
            goal_type=selected["goal_type"],
            status="proposed",
            priority=float(selected["score"]),
            risk_level=selected["risk_level"],
            requires_approval=False,
            metadata={"generated_by": "goal_generator", "root_objective_id": selected.get("root_objective_id"), "candidate_id": selected.get("id"), "execution": "not_executed_same_tick"},
            dedupe=True,
        )
        task_id = enqueue_task(
            "autonomous",
            goal_id=created_goal_id,
            task_kind=str(selected["goal_type"]),
            title=str(selected["title"]),
            source="goal_generator",
            priority=float(selected["score"]),
            payload={"candidate_id": selected.get("id"), "root_objective_id": selected.get("root_objective_id")},
        )
        _update_candidate(int(selected["id"]), status="proposed", generated_goal_id=created_goal_id)
        _mark_objective_used(int(selected["root_objective_id"]))
        selected["status"] = "proposed"
        selected["generated_goal_id"] = created_goal_id
        selected["task_id"] = task_id
        for candidate in candidates[1:]:
            _update_candidate(int(candidate["id"]), status="rejected", rejection_reason="lower_score")
            candidate["status"] = "rejected"
            candidate["rejection_reason"] = "lower_score"

    result = {"created_goal_id": created_goal_id, "generated": not dry_run, "reason": "dry_run" if dry_run else "created_proposed_goal", "candidates": candidates, "rejected": rejected[:5], "execution": "not_executed"}
    log_event("goal", "goal_candidates_generated", result["reason"], {"created_goal_id": created_goal_id, "candidate_count": len(candidates), "dry_run": dry_run}, 0.7)
    return result
