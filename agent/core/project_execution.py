from __future__ import annotations

import json
import re
from typing import Any

from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.failure import classify_failure, recovery_hint

BROAD_GOAL_TOKENS = (
    "싹", "전부", "완벽", "최대한", "프로젝트", "구현", "개선", "디벨롭",
    "리팩토링", "리팩터링", "분석", "조사", "최종", "운영체제", "agi",
)

PROJECT_PLAN_TASK_KINDS = {"code_change", "project_spec", "improvement_plan", "workspace_experiment"}


def _json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _decode_json(value: object, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _decode_plan(row: dict[str, Any]) -> dict[str, Any]:
    row["plan"] = _decode_json(row.get("plan_json"), {})
    row["result"] = _decode_json(row.get("result_json"), {})
    return row


def _decode_step(row: dict[str, Any]) -> dict[str, Any]:
    row["completion_criteria"] = _decode_json(row.get("completion_criteria_json"), [])
    row["verification"] = _decode_json(row.get("verification_json"), [])
    row["result"] = _decode_json(row.get("result_json"), {})
    return row


def needs_project_plan(text: str, task_kind: str, interpretation: dict[str, Any] | None = None) -> bool:
    lowered = text.lower()
    if task_kind in {"code_change", "project_spec"}:
        return True
    if task_kind in PROJECT_PLAN_TASK_KINDS and (len(text.strip()) >= 36 or any(token in lowered for token in BROAD_GOAL_TOKENS)):
        return True
    target = str((interpretation or {}).get("target") or "")
    return target in PROJECT_PLAN_TASK_KINDS and any(token in lowered for token in BROAD_GOAL_TOKENS)


def classify_failure_reason(value: object) -> str:
    return classify_failure(value)


def _verification_for(task_kind: str) -> list[dict[str, str]]:
    if task_kind == "code_change":
        return [
            {"type": "tests", "command": "python -m agent.cli.agentctl test run fast --json"},
            {"type": "audit", "command": "python -m agent.cli.agentctl audit"},
        ]
    if task_kind == "project_spec":
        return [{"type": "artifact", "expect": "project spec artifact exists"}]
    if task_kind == "improvement_plan":
        return [{"type": "artifact", "expect": "improvement plan artifact exists"}]
    return [{"type": "artifact", "expect": "task result artifact or report exists"}]


def build_project_plan(text: str, task_kind: str) -> dict[str, Any]:
    objective = re.sub(r"\s+", " ", text.strip())[:500]
    if task_kind == "code_change":
        steps = [
            ("요청 해석", "사용자 목표, 위험 경계, 완료 조건을 정리한다.", "planning"),
            ("구현", "Codex work worker가 repo 안에서 좁은 범위로 수정한다.", "implementation"),
            ("검증", "가능한 테스트와 audit/eval 기준으로 결과를 확인한다.", "verification"),
            ("보고", "변경점, 검증 결과, 남은 위험을 사용자에게 보고한다.", "reporting"),
        ]
    elif task_kind == "project_spec":
        steps = [
            ("요구 정리", "목표와 사용자 맥락을 구조화한다.", "planning"),
            ("설계 초안", "기능, 제약, 산출물을 문서화한다.", "implementation"),
            ("검토", "누락된 위험과 다음 구현 단위를 확인한다.", "verification"),
            ("보고", "생성된 산출물과 다음 액션을 보고한다.", "reporting"),
        ]
    else:
        steps = [
            ("문제 정리", "목표와 개선 대상을 명확히 한다.", "planning"),
            ("작업 수행", "현재 안전 경계 안에서 산출물 또는 구현을 만든다.", "implementation"),
            ("결과 확인", "완료 조건과 실패 원인을 점검한다.", "verification"),
            ("보고", "결과와 다음 액션을 짧게 정리한다.", "reporting"),
        ]
    verification = _verification_for(task_kind)
    return {
        "objective": objective,
        "task_kind": task_kind,
        "steps": [
            {
                "index": index,
                "title": title,
                "description": description,
                "task_kind": kind,
                "completion_criteria": [
                    f"{title} 단계가 기록됨",
                    "정책/위험 경계를 우회하지 않음",
                ],
                "verification": verification if kind == "verification" else [],
            }
            for index, (title, description, kind) in enumerate(steps, start=1)
        ],
    }


def create_project_execution_plan(
    *,
    goal_id: int | None,
    task_id: int | None = None,
    source: str,
    owner: str,
    title: str,
    objective: str,
    task_kind: str,
    priority: float = 0.5,
) -> dict[str, Any]:
    init_db()
    ts = now_kst()
    plan = build_project_plan(objective, task_kind)
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO project_execution_plans (
                created_at, updated_at, goal_id, task_id, source, owner, title,
                objective, status, priority, current_step_index, plan_json, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, 0, ?, '{}')
            """,
            (ts, ts, goal_id, task_id, source, owner, title[:160], objective[:1000], max(0.0, min(1.0, priority)), _json(plan)),
        )
        plan_id = int(cur.lastrowid)
        for step in plan["steps"]:
            conn.execute(
                """
                INSERT INTO project_execution_steps (
                    created_at, updated_at, plan_id, step_index, title, description,
                    task_kind, status, completion_criteria_json, verification_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, '{}')
                """,
                (
                    ts, ts, plan_id, step["index"], step["title"], step["description"],
                    step["task_kind"], _json(step["completion_criteria"]), _json(step["verification"]),
                ),
            )
        conn.commit()
    result = get_project_plan(plan_id) or {"id": plan_id}
    log_event("project_execution", "plan_created", title[:160], {"plan_id": plan_id, "goal_id": goal_id, "task_kind": task_kind}, 0.78)
    return result


def link_plan_task(plan_id: int, task_id: int) -> bool:
    init_db()
    with connect() as conn:
        cur = conn.execute(
            "UPDATE project_execution_plans SET task_id = ?, updated_at = ? WHERE id = ?",
            (int(task_id), now_kst(), int(plan_id)),
        )
        conn.execute("UPDATE project_execution_steps SET queued_task_id = ? WHERE plan_id = ?", (int(task_id), int(plan_id)))
        conn.commit()
    return cur.rowcount > 0


def get_project_plan(plan_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        plan_row = conn.execute("SELECT * FROM project_execution_plans WHERE id = ?", (int(plan_id),)).fetchone()
        step_rows = conn.execute("SELECT * FROM project_execution_steps WHERE plan_id = ? ORDER BY step_index ASC", (int(plan_id),)).fetchall()
    if not plan_row:
        return None
    plan = _decode_plan(dict(plan_row))
    plan["steps"] = [_decode_step(dict(row)) for row in step_rows]
    return plan


def get_project_plan_for_goal(goal_id: int) -> dict[str, Any] | None:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT id FROM project_execution_plans WHERE goal_id = ? ORDER BY id DESC LIMIT 1", (int(goal_id),)).fetchone()
    return get_project_plan(int(row["id"])) if row else None


def list_project_plans(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM project_execution_plans"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY status IN ('running','planned') DESC, priority DESC, id DESC LIMIT ?"
    params.append(int(limit))
    with connect() as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    return [_decode_plan(dict(row)) for row in rows]


def mark_plan_running(plan_id: int, *, task_id: int | None = None) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE project_execution_plans SET status = 'running', task_id = COALESCE(?, task_id), updated_at = ? WHERE id = ?",
            (task_id, now_kst(), int(plan_id)),
        )
        conn.execute(
            "UPDATE project_execution_steps SET status = 'running', updated_at = ? WHERE plan_id = ? AND step_index = 1 AND status = 'pending'",
            (now_kst(), int(plan_id)),
        )
        conn.commit()


def mark_project_step(plan_id: int, step_kind: str, status: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    if status not in {"pending", "running", "done", "blocked", "skipped"}:
        raise ValueError(f"invalid project step status: {status}")
    init_db()
    ts = now_kst()
    failure_category = None if status in {"pending", "running", "done", "skipped"} else classify_failure_reason(result)
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, step_index
            FROM project_execution_steps
            WHERE plan_id = ? AND task_kind = ?
            ORDER BY step_index ASC
            LIMIT 1
            """,
            (int(plan_id), step_kind),
        ).fetchone()
        if not row:
            raise ValueError(f"project step not found: {step_kind}")
        step_index = int(row["step_index"])
        conn.execute(
            """
            UPDATE project_execution_steps
            SET status = ?, updated_at = ?, failure_category = ?, result_json = ?
            WHERE id = ?
            """,
            (status, ts, failure_category, _json(result or {}), int(row["id"])),
        )
        plan_status = "blocked" if status == "blocked" else "running"
        conn.execute(
            """
            UPDATE project_execution_plans
            SET status = ?, current_step_index = ?, updated_at = ?
            WHERE id = ?
            """,
            (plan_status, step_index, ts, int(plan_id)),
        )
        conn.commit()
    plan = get_project_plan(plan_id) or {"id": plan_id, "status": status}
    log_event("project_execution", f"step_{status}", step_kind, {"plan_id": plan_id, "step_kind": step_kind, "failure_category": failure_category}, 0.74)
    return plan


def mark_project_step_done(plan_id: int, step_kind: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    return mark_project_step(plan_id, step_kind, "done", result)


def block_project_plan(plan_id: int, result: dict[str, Any]) -> dict[str, Any]:
    failure_category = classify_failure_reason(result)
    report = {"status": result.get("status"), "failure_category": failure_category, "recovery_hint": recovery_hint(failure_category), "task_result": result}
    ts = now_kst()
    init_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE project_execution_steps
            SET status = CASE WHEN status = 'done' THEN 'done' ELSE 'blocked' END,
                updated_at = ?,
                failure_category = CASE WHEN status = 'done' THEN failure_category ELSE ? END,
                result_json = CASE WHEN status = 'done' THEN result_json ELSE ? END
            WHERE plan_id = ?
            """,
            (ts, failure_category, _json(result), int(plan_id)),
        )
        conn.execute(
            """
            UPDATE project_execution_plans
            SET status = 'blocked', updated_at = ?, result_json = ?
            WHERE id = ?
            """,
            (ts, _json(report), int(plan_id)),
        )
        conn.commit()
    plan = get_project_plan(plan_id) or {"id": plan_id, "status": "blocked"}
    log_event("project_execution", "plan_blocked", str(plan_id), {"plan_id": plan_id, "failure_category": failure_category}, 0.78)
    return plan


def complete_project_plan(plan_id: int, result: dict[str, Any]) -> dict[str, Any]:
    init_db()
    status = str(result.get("status") or "")
    success = status in {"user_goal_completed", "artifact_created", "completed", "done", "codex_work_completed"}
    failure_category = None if success else classify_failure_reason(result)
    recovery = None if success else recovery_hint(failure_category or "unknown")
    final_status = "done" if success else "blocked"
    ts = now_kst()
    with connect() as conn:
        rows = conn.execute("SELECT id, step_index FROM project_execution_steps WHERE plan_id = ? ORDER BY step_index ASC", (int(plan_id),)).fetchall()
        for row in rows:
            conn.execute(
                """
                UPDATE project_execution_steps
                SET status = CASE WHEN status = 'done' THEN 'done' ELSE ? END,
                    updated_at = ?,
                    failure_category = CASE WHEN status = 'done' THEN failure_category ELSE ? END,
                    result_json = CASE WHEN status = 'done' THEN result_json ELSE ? END
                WHERE id = ?
                """,
                (final_status, ts, failure_category, _json(result), int(row["id"])),
            )
        conn.execute(
            """
            UPDATE project_execution_plans
            SET status = ?, current_step_index = ?, updated_at = ?, result_json = ?
            WHERE id = ?
            """,
            (final_status, len(rows), ts, _json({"status": status, "failure_category": failure_category, "recovery_hint": recovery, "task_result": result}), int(plan_id)),
        )
        conn.commit()
    plan = get_project_plan(plan_id) or {"id": plan_id, "status": final_status}
    log_event("project_execution", f"plan_{final_status}", str(plan_id), {"plan_id": plan_id, "failure_category": failure_category}, 0.82 if success else 0.74)
    return plan


def project_plan_brief(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not plan:
        return {}
    steps = plan.get("steps") or []
    done = len([step for step in steps if step.get("status") == "done"])
    blocked = len([step for step in steps if step.get("status") == "blocked"])
    return {
        "id": plan.get("id"),
        "status": plan.get("status"),
        "title": plan.get("title"),
        "steps_total": len(steps),
        "steps_done": done,
        "steps_blocked": blocked,
        "failure_category": (plan.get("result") or {}).get("failure_category"),
        "recovery_hint": (plan.get("result") or {}).get("recovery_hint"),
    }
