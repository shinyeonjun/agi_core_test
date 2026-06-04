from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

from agent.config.defaults import KST, now_kst
from agent.core.database import connect, init_db

NODE_TYPES = {
    "memory", "goal", "task", "skill", "failure_case", "policy_decision",
    "device_state", "user_preference", "research_idea",
}
EDGE_TYPES = {"supports", "caused", "blocked_by", "similar_to", "depends_on", "improves", "conflicts_with", "learned_from", "suggests"}
RISK_SCORE = {"low": 0.08, "medium": 0.35, "high": 0.72, "critical": 1.0}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode(value: object, fallback: Any) -> Any:
    try:
        return json.loads(str(value)) if value is not None else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _terms(text: str) -> list[str]:
    out = []
    for term in re.findall(r"[0-9A-Za-z_\uac00-\ud7a3]+", text.lower()):
        if len(term) >= 2 and term not in out:
            out.append(term)
    return out[:24]


def _overlap(a: str, b: str) -> float:
    aa, bb = set(_terms(a)), set(_terms(b))
    return len(aa & bb) / len(aa | bb) if aa and bb else 0.0


def _fresh(*values: object) -> float:
    raw = next((value for value in values if value), None)
    if not raw:
        return 0.25
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return 0.25
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    days = max(0.0, (datetime.now(KST) - dt).total_seconds() / 86400)
    return _clamp(math.exp(-days / 21.0))


def _risk(value: object) -> float:
    return RISK_SCORE.get(str(value or "low"), 0.2)


def _success_rate(success: object, failure: object) -> float | None:
    s, f = max(0, int(success or 0)), max(0, int(failure or 0))
    return round(s / (s + f), 4) if s + f else None


def _revisit(importance: float, confidence: float, freshness: float, risk: float, success_rate: float | None) -> float:
    success = 0.5 if success_rate is None else success_rate
    score = importance * 0.32 + freshness * 0.20 + confidence * 0.16 + (1 - confidence) * 0.12 + (1 - risk) * 0.10 + success * 0.10
    return round(_clamp(score), 4)


def upsert_node(node_type: str, source_type: str, source_id: object, title: str, *, summary: str = "", importance: float = 0.5, confidence: float = 0.7, freshness: float = 0.5, risk: float = 0.0, success_rate: float | None = None, metadata: dict[str, Any] | None = None) -> int:
    if node_type not in NODE_TYPES:
        raise ValueError(f"unknown cognitive node type: {node_type}")
    init_db()
    ts = now_kst()
    importance, confidence, freshness, risk = map(lambda v: _clamp(float(v)), (importance, confidence, freshness, risk))
    success_rate = None if success_rate is None else _clamp(float(success_rate))
    revisit = _revisit(importance, confidence, freshness, risk, success_rate)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO cognitive_nodes (created_at, updated_at, node_type, source_type, source_id, title, summary, importance, confidence, freshness, risk, success_rate, revisit_score, metadata_json, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(node_type, source_type, source_id) DO UPDATE SET
                updated_at = excluded.updated_at, title = excluded.title, summary = excluded.summary,
                importance = excluded.importance, confidence = excluded.confidence, freshness = excluded.freshness,
                risk = excluded.risk, success_rate = excluded.success_rate, revisit_score = excluded.revisit_score,
                metadata_json = excluded.metadata_json, archived = 0
            """,
            (ts, ts, node_type, source_type, str(source_id), title[:220], summary[:2000], importance, confidence, freshness, risk, success_rate, revisit, _json(metadata or {})),
        )
        row = conn.execute("SELECT id FROM cognitive_nodes WHERE node_type=? AND source_type=? AND source_id=?", (node_type, source_type, str(source_id))).fetchone()
        conn.commit()
    return int(row["id"])


def upsert_edge(from_node_id: int, to_node_id: int, edge_type: str, *, weight: float = 0.5, confidence: float = 0.7, evidence: dict[str, Any] | None = None) -> int:
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"unknown cognitive edge type: {edge_type}")
    if int(from_node_id) == int(to_node_id):
        raise ValueError("cognitive edge cannot point to itself")
    init_db()
    ts = now_kst()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO cognitive_edges (created_at, updated_at, from_node_id, to_node_id, edge_type, weight, confidence, evidence_json, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(from_node_id, to_node_id, edge_type) DO UPDATE SET
                updated_at=excluded.updated_at, weight=excluded.weight, confidence=excluded.confidence, evidence_json=excluded.evidence_json, archived=0
            """,
            (ts, ts, int(from_node_id), int(to_node_id), edge_type, _clamp(float(weight)), _clamp(float(confidence)), _json(evidence or {})),
        )
        row = conn.execute("SELECT id FROM cognitive_edges WHERE from_node_id=? AND to_node_id=? AND edge_type=?", (int(from_node_id), int(to_node_id), edge_type)).fetchone()
        conn.commit()
    return int(row["id"])


def _counts() -> dict[str, int]:
    init_db()
    with connect() as conn:
        return {
            "nodes": int(conn.execute("SELECT COUNT(*) c FROM cognitive_nodes WHERE archived=0").fetchone()["c"]),
            "edges": int(conn.execute("SELECT COUNT(*) c FROM cognitive_edges WHERE archived=0").fetchone()["c"]),
            "activations": int(conn.execute("SELECT COUNT(*) c FROM cognitive_activations").fetchone()["c"]),
            "traces": int(conn.execute("SELECT COUNT(*) c FROM cognitive_traces").fetchone()["c"]),
        }


def sync_graph(limit: int = 80) -> dict[str, Any]:
    init_db()
    before = _counts()
    with connect() as conn:
        goals = {}
        for r in conn.execute("SELECT * FROM goals WHERE status!='archived' ORDER BY status='active' DESC, priority DESC, id DESC LIMIT ?", (limit,)).fetchall():
            d = dict(r)
            goals[int(d["id"])] = upsert_node("goal", "goal", d["id"], d["title"], summary=d.get("description") or "", importance=float(d.get("priority") or 0.5), confidence=0.78 if d.get("status") == "active" else 0.58, freshness=_fresh(d.get("updated_at"), d.get("created_at")), risk=_risk(d.get("risk_level")), metadata={"status": d.get("status"), "goal_type": d.get("goal_type")})
        tasks = {}
        for r in conn.execute("SELECT * FROM task_queue ORDER BY status IN ('queued','running','waiting_approval') DESC, priority DESC, id DESC LIMIT ?", (limit,)).fetchall():
            d = dict(r); status = str(d.get("status") or "queued")
            sr = 1.0 if status in {"done", "completed"} else 0.0 if status in {"blocked", "failed", "timeout"} else None
            tasks[int(d["id"])] = upsert_node("task", "task_queue", d["id"], d["title"], summary=f"{d.get('queue_type')} / {d.get('task_kind')} / {status}", importance=float(d.get("priority") or 0.5), confidence=0.82 if status in {"running", "done"} else 0.62, freshness=_fresh(d.get("updated_at"), d.get("created_at")), risk=0.25 if d.get("approval_id") else 0.08, success_rate=sr, metadata={"goal_id": d.get("goal_id"), "status": status, "queue_type": d.get("queue_type")})
            if d.get("goal_id") and int(d["goal_id"]) in goals:
                upsert_edge(tasks[int(d["id"])], goals[int(d["goal_id"])], "depends_on", weight=0.86, confidence=0.9, evidence={"source": "task.goal_id"})
        for r in conn.execute("SELECT * FROM memories WHERE archived=0 ORDER BY importance DESC, updated_at DESC, id DESC LIMIT ?", (limit,)).fetchall():
            d = dict(r); mt = str(d.get("memory_type") or "fact")
            nt = "failure_case" if mt in {"failure_summary", "negative_feedback"} else "user_preference" if mt in {"preference", "preference_summary"} else "research_idea" if "research" in mt else "memory"
            nid = upsert_node(nt, "memory", d["id"], d["title"], summary=d.get("content") or "", importance=float(d.get("importance") or 0.5), confidence=float(d.get("confidence") or 0.7), freshness=_fresh(d.get("last_used_at"), d.get("updated_at"), d.get("created_at")), risk=0.05, metadata={"memory_type": mt, "tags": _decode(d.get("tags_json"), [])})
            _link_best_goal(conn, nid, nt, f"{d.get('title','')} {d.get('content','')}", goals)
        for r in conn.execute("SELECT * FROM skills WHERE archived=0 ORDER BY confidence DESC, id DESC LIMIT ?", (max(20, limit // 2),)).fetchall():
            d = dict(r)
            nid = upsert_node("skill", "skill", d["id"], d["name"], summary=d.get("trigger_description") or "", importance=float(d.get("confidence") or 0.5), confidence=float(d.get("confidence") or 0.5), freshness=_fresh(d.get("last_used_at"), d.get("updated_at"), d.get("created_at")), risk=0.08, success_rate=_success_rate(d.get("success_count"), d.get("failure_count")), metadata={"tags": _decode(d.get("tags_json"), [])})
            _link_best_goal(conn, nid, "skill", f"{d.get('name','')} {d.get('trigger_description','')}", goals)
        for r in conn.execute("SELECT * FROM policy_decisions ORDER BY id DESC LIMIT ?", (max(20, limit // 2),)).fetchall():
            d = dict(r)
            upsert_node("policy_decision", "policy_decision", d["id"], f"{d.get('action_type')} / {d.get('risk_level')}", summary=d.get("reason") or d.get("normalized_text") or "", importance=0.86 if d.get("denied") or d.get("requires_approval") else 0.48, confidence=0.82, freshness=_fresh(d.get("created_at")), risk=_risk(d.get("risk_level")), success_rate=0.0 if d.get("denied") else None, metadata={"requires_approval": bool(d.get("requires_approval")), "denied": bool(d.get("denied"))})
        for r in conn.execute("SELECT * FROM style_feedback ORDER BY id DESC LIMIT ?", (max(20, limit // 2),)).fetchall():
            d = dict(r)
            upsert_node("user_preference", "style_feedback", d["id"], d.get("feedback_type") or "style feedback", summary=d.get("feedback_text") or d.get("source_text") or "", importance=0.7, confidence=0.72, freshness=_fresh(d.get("created_at")), risk=0.03, metadata={"preference": _decode(d.get("extracted_preference_json"), {})})
        for r in conn.execute("SELECT * FROM blackboard_items WHERE topic LIKE '%research%' OR topic LIKE '%paper%' OR tags_json LIKE '%research%' ORDER BY confidence DESC, id DESC LIMIT ?", (max(20, limit // 2),)).fetchall():
            d = dict(r)
            nid = upsert_node("research_idea", "blackboard", d["id"], d.get("topic") or "research idea", summary=d.get("content") or "", importance=float(d.get("confidence") or 0.5), confidence=float(d.get("confidence") or 0.5), freshness=_fresh(d.get("updated_at"), d.get("created_at")), risk=0.1, metadata={"source": d.get("source"), "status": d.get("status")})
            _link_best_goal(conn, nid, "research_idea", f"{d.get('topic','')} {d.get('content','')}", goals)
        for r in conn.execute("SELECT * FROM task_lifecycle_events WHERE status IN ('blocked','failed') ORDER BY id DESC LIMIT ?", (max(20, limit // 2),)).fetchall():
            d = dict(r)
            upsert_node("failure_case", "task_lifecycle", d["id"], f"Task failure/block #{d.get('task_id')}", summary=d.get("summary") or "", importance=0.76, confidence=0.76, freshness=_fresh(d.get("created_at")), risk=0.52, success_rate=0.0, metadata={"task_id": d.get("task_id"), "phase": d.get("phase"), "status": d.get("status")})
        row = conn.execute("SELECT * FROM self_maps ORDER BY id DESC LIMIT 1").fetchone()
        if row:
            d = dict(row)
            device = upsert_node("device_state", "self_map", d["id"], "Current device state", summary=d.get("summary") or "", importance=0.68, confidence=0.82, freshness=_fresh(d.get("created_at")), risk=0.12, metadata={"changed": bool(d.get("changed")), "fingerprint": d.get("fingerprint")})
            for gid, node in list(goals.items())[:8]:
                upsert_edge(device, node, "supports", weight=0.42, confidence=0.58, evidence={"source": "latest_self_map"})
    after = _counts()
    return {"synced_at": now_kst(), "before": before, "after": after, "delta": {k: after[k] - before.get(k, 0) for k in after}}


def _link_best_goal(conn: Any, node_id: int, node_type: str, text: str, goals: dict[int, int]) -> None:
    if not goals:
        return
    rows = conn.execute("SELECT id, title, COALESCE(description,'') description FROM goals WHERE status!='archived' ORDER BY priority DESC LIMIT 80").fetchall()
    best = None; score = 0.0
    for r in rows:
        if int(r["id"]) not in goals:
            continue
        s = _overlap(text, f"{r['title']} {r['description']}")
        if s > score:
            score = s; best = int(r["id"])
    if best and score >= 0.08:
        et = "learned_from" if node_type == "failure_case" else "suggests" if node_type in {"skill", "research_idea"} else "supports"
        upsert_edge(node_id, goals[best], et, weight=min(0.95, 0.45 + score), confidence=0.62, evidence={"lexical_overlap": round(score, 4)})


def activate_graph(query: str, *, limit: int = 8, sync: bool = False) -> dict[str, Any]:
    if sync:
        sync_graph(limit=max(40, limit * 8))
    init_db(); terms = _terms(query)
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM cognitive_nodes WHERE archived=0 ORDER BY revisit_score DESC, updated_at DESC, id DESC LIMIT 400").fetchall()]
        scored = []
        for d in rows:
            lexical = _overlap(query, f"{d.get('title','')} {d.get('summary','')} {d.get('node_type','')}") if terms else 0.0
            success = 0.5 if d.get("success_rate") is None else float(d.get("success_rate") or 0.0)
            components = {"lexical": lexical, "importance": float(d.get("importance") or 0), "confidence": float(d.get("confidence") or 0), "freshness": float(d.get("freshness") or 0), "revisit": float(d.get("revisit_score") or 0), "success": success, "risk_penalty": float(d.get("risk") or 0) * 0.10}
            d["activation_score"] = round(_clamp(components["lexical"] * 0.36 + components["importance"] * 0.18 + components["confidence"] * 0.14 + components["freshness"] * 0.12 + components["revisit"] * 0.12 + components["success"] * 0.08 - components["risk_penalty"]), 4)
            d["components"] = {k: round(v, 4) for k, v in components.items()}
            scored.append(d)
        scored.sort(key=lambda x: x["activation_score"], reverse=True)
        picked = scored[:limit]
        ts = now_kst()
        for d in picked:
            conn.execute("INSERT INTO cognitive_activations (created_at, node_id, context, reason, score, components_json) VALUES (?, ?, ?, ?, ?, ?)", (ts, d["id"], query[:500], "query_activation", d["activation_score"], _json(d["components"])))
        conn.commit()
    return {"query": query, "terms": terms, "items": [_public_node(d) for d in picked]}


def trace_decision(trace_type: str, summary: str, *, query: str | None = None, decision_id: str | None = None, activated: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    activated = activated if activated is not None else (activate_graph(query, limit=6)["items"] if query else [])
    payload = {"query": query, "summary": summary, "activated": activated}
    root = activated[0].get("id") if activated else None
    init_db()
    with connect() as conn:
        cur = conn.execute("INSERT INTO cognitive_traces (created_at, trace_type, decision_id, root_node_id, summary, trace_json) VALUES (?, ?, ?, ?, ?, ?)", (now_kst(), trace_type, decision_id, root, summary[:500], _json(payload)))
        conn.commit()
    return {"trace_id": int(cur.lastrowid), "trace_type": trace_type, "summary": summary, "root_node_id": root, "activated": activated}


def graph_snapshot(*, limit: int = 10) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        node_types = {r["node_type"]: int(r["c"]) for r in conn.execute("SELECT node_type, COUNT(*) c FROM cognitive_nodes WHERE archived=0 GROUP BY node_type")}
        edge_types = {r["edge_type"]: int(r["c"]) for r in conn.execute("SELECT edge_type, COUNT(*) c FROM cognitive_edges WHERE archived=0 GROUP BY edge_type")}
        top = [_public_node(dict(r)) for r in conn.execute("SELECT * FROM cognitive_nodes WHERE archived=0 ORDER BY revisit_score DESC, updated_at DESC LIMIT ?", (limit,)).fetchall()]
        recent = [dict(r) for r in conn.execute("SELECT id, created_at, trace_type, decision_id, summary, root_node_id FROM cognitive_traces ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
    return {"created_at": now_kst(), "counts": _counts(), "node_types": node_types, "edge_types": edge_types, "top_nodes": top, "recent_traces": recent}


def _public_node(d: dict[str, Any]) -> dict[str, Any]:
    return {"id": int(d["id"]), "type": d.get("node_type"), "source": f"{d.get('source_type')}:{d.get('source_id')}", "title": d.get("title"), "summary": d.get("summary"), "scores": {"activation": d.get("activation_score"), "importance": d.get("importance"), "confidence": d.get("confidence"), "freshness": d.get("freshness"), "risk": d.get("risk"), "success_rate": d.get("success_rate"), "revisit": d.get("revisit_score")}, "components": d.get("components")}
