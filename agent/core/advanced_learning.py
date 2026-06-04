from __future__ import annotations

import hashlib, json, re
from collections import Counter, defaultdict
from typing import Any

from agent.config.defaults import now_kst
from agent.core.cognitive_engine import failure_strategy, htn_plan_for_goal
from agent.core.cognitive_graph import graph_snapshot, sync_graph
from agent.core.database import connect, init_db
from agent.core.failure import failure_report
from agent.core.goals import get_goal, list_goals
from agent.core.learner import create_reflection


def _json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def _decode(v: object, fallback: Any) -> Any:
    if v is None:
        return fallback
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(str(v))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _terms(text: object, limit: int = 12) -> list[str]:
    out = []
    for term in re.findall(r"[0-9A-Za-z_\uac00-\ud7a3]+", str(text or "").lower()):
        if len(term) >= 2 and term not in out:
            out.append(term)
    return out[:limit]


def _fp(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p or "") for p in parts).encode()).hexdigest()[:16]


def _snippet(row: dict[str, Any], max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(row.get("content") or row.get("summary") or row.get("title") or "")).strip()
    return text[:max_chars]


def _cluster_key(row: dict[str, Any]) -> str:
    tags = sorted(str(t).lower() for t in _decode(row.get("tags_json"), []) if str(t).strip())[:4]
    return "|".join([str(row.get("memory_type") or "fact"), ",".join(tags), ",".join(_terms(row.get("title"), 4))])


def _persist_rollup(level: int, key: str, title: str, summary: str, source_ids: list[int], score: float, meta: dict[str, Any]) -> int:
    ts = now_kst()
    with connect() as conn:
        conn.execute("""
            INSERT INTO memory_rollups (created_at, updated_at, level, cluster_key, title, summary, source_memory_ids_json, score, metadata_json, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(level, cluster_key) DO UPDATE SET updated_at=excluded.updated_at, title=excluded.title, summary=excluded.summary, source_memory_ids_json=excluded.source_memory_ids_json, score=excluded.score, metadata_json=excluded.metadata_json, archived=0
        """, (ts, ts, level, key, title[:220], summary[:2400], _json(source_ids), max(0, min(1, score)), _json(meta)))
        row = conn.execute("SELECT id FROM memory_rollups WHERE level=? AND cluster_key=?", (level, key)).fetchone()
        conn.commit()
    return int(row["id"])


def build_memory_hierarchy(*, limit: int = 80, levels: int = 2, persist: bool = False) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("""
            SELECT * FROM memories WHERE archived=0
            ORDER BY importance DESC, confidence DESC, updated_at DESC, id DESC LIMIT ?
        """, (max(1, int(limit)),)).fetchall()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_cluster_key(row)].append(row)
    clusters = []
    for key, items in groups.items():
        source_ids = [int(r["id"]) for r in items]
        summary = " / ".join(dict.fromkeys(_snippet(r) for r in items if _snippet(r)))[:1400] or "요약할 기억이 부족함"
        score = min(1.0, len(items) / 8 * 0.35 + max(float(r.get("importance") or 0) for r in items) * 0.45 + max(float(r.get("confidence") or 0) for r in items) * 0.20)
        item = {"level": 1, "cluster_key": _fp("memory", key), "title": f"기억 묶음: {items[0].get('memory_type') or 'fact'} / {items[0].get('title') or 'untitled'}"[:220], "summary": summary, "source_memory_ids": source_ids, "count": len(items), "score": round(score, 4)}
        if persist:
            item["rollup_id"] = _persist_rollup(1, item["cluster_key"], item["title"], summary, source_ids, score, {"count": len(items)})
        clusters.append(item)
    clusters.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    roots = []
    if levels >= 2 and clusters:
        top = clusters[: min(12, len(clusters))]
        source_ids = sorted({mid for c in top for mid in c["source_memory_ids"]})
        score = sum(float(c["score"]) for c in top) / len(top)
        root = {"level": 2, "cluster_key": _fp("memory-root", len(clusters), source_ids[:40]), "title": "전체 기억 상위 요약", "summary": " / ".join(c["summary"][:180] for c in top[:6])[:1800], "source_memory_ids": source_ids, "count": len(source_ids), "score": round(score, 4)}
        if persist:
            root["rollup_id"] = _persist_rollup(2, root["cluster_key"], root["title"], root["summary"], source_ids, score, {"cluster_count": len(top)})
        roots.append(root)
    if persist and (clusters or roots):
        create_reflection("기억 계층 요약을 갱신함", learned={"clusters": len(clusters), "roots": len(roots)}, confidence=0.78)
    return {"persisted": persist, "source_count": len(rows), "clusters": clusters, "roots": roots}


def summarize_cognitive_graph(*, limit: int = 120, persist: bool = False, sync: bool = True) -> dict[str, Any]:
    init_db()
    if sync:
        sync_graph(limit=max(40, min(200, int(limit))))
    with connect() as conn:
        nodes = [dict(r) for r in conn.execute("SELECT * FROM cognitive_nodes WHERE archived=0 ORDER BY revisit_score DESC, updated_at DESC, id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()]
        edges = [dict(r) for r in conn.execute("""
            SELECT e.*, a.node_type from_type, b.node_type to_type FROM cognitive_edges e
            JOIN cognitive_nodes a ON a.id=e.from_node_id JOIN cognitive_nodes b ON b.id=e.to_node_id
            WHERE e.archived=0 ORDER BY e.weight DESC, e.updated_at DESC LIMIT ?
        """, (max(1, int(limit * 2)),)).fetchall()]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_type[str(node.get("node_type") or "unknown")].append(node)
    communities = []
    for node_type, items in by_type.items():
        top = sorted(items, key=lambda r: float(r.get("revisit_score") or 0), reverse=True)[:8]
        ids = {int(r["id"]) for r in items}
        linked = [e for e in edges if int(e["from_node_id"]) in ids or int(e["to_node_id"]) in ids]
        summary = " / ".join(_snippet(r) for r in top if _snippet(r))[:1400] or "그래프 요약 재료가 부족함"
        score = sum(float(r.get("revisit_score") or 0) for r in top) / max(1, len(top))
        item = {"community_key": _fp("graph", node_type), "node_type": node_type, "title": f"인지 그래프 요약: {node_type}", "summary": summary, "node_ids": [int(r["id"]) for r in top], "edge_count": len(linked), "dominant_edges": dict(Counter(str(e.get("edge_type") or "unknown") for e in linked).most_common(4)), "score": round(score, 4)}
        if persist:
            ts = now_kst()
            with connect() as conn:
                conn.execute("""
                    INSERT INTO graph_community_summaries (created_at, updated_at, community_key, title, summary, node_ids_json, edge_ids_json, score, metadata_json, archived)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(community_key) DO UPDATE SET updated_at=excluded.updated_at, title=excluded.title, summary=excluded.summary, node_ids_json=excluded.node_ids_json, edge_ids_json=excluded.edge_ids_json, score=excluded.score, metadata_json=excluded.metadata_json, archived=0
                """, (ts, ts, item["community_key"], item["title"], summary, _json(item["node_ids"]), _json([int(e["id"]) for e in linked[:50]]), score, _json({"node_type": node_type, "dominant_edges": item["dominant_edges"]})))
                row = conn.execute("SELECT id FROM graph_community_summaries WHERE community_key=?", (item["community_key"],)).fetchone(); conn.commit()
            item["summary_id"] = int(row["id"])
        communities.append(item)
    communities.sort(key=lambda x: x["score"], reverse=True)
    return {"persisted": persist, "graph": graph_snapshot(limit=5), "communities": communities}

def index_failure_cases(*, limit: int = 80, persist: bool = False) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        lifecycle = [dict(r) for r in conn.execute("""
            SELECT * FROM task_lifecycle_events
            WHERE status IN ('blocked','failed','timeout') OR summary LIKE '%실패%' OR summary LIKE '%차단%'
            ORDER BY id DESC LIMIT ?
        """, (max(1, int(limit)),)).fetchall()]
        tasks = [dict(r) for r in conn.execute("SELECT * FROM task_queue WHERE status IN ('blocked','skipped') ORDER BY id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()]
    cases = []
    for r in lifecycle:
        evidence = {"summary": r.get("summary"), "status": r.get("status"), "metadata": _decode(r.get("metadata_json"), {})}
        rep = failure_report(evidence)
        cases.append({"source_type": "task_lifecycle", "source_id": str(r.get("id")), "task_id": r.get("task_id"), "title": f"작업 #{r.get('task_id')} {r.get('phase')} 실패/차단", "category": rep["category"], "summary": str(r.get("summary") or "")[:600], "evidence": evidence})
    for r in tasks:
        evidence = {"title": r.get("title"), "status": r.get("status"), "result": _decode(r.get("result_json"), {})}
        rep = failure_report(evidence)
        cases.append({"source_type": "task_queue", "source_id": str(r.get("id")), "task_id": r.get("id"), "title": f"큐 작업 #{r.get('id')} 실패/차단", "category": rep["category"], "summary": str(r.get("title") or "")[:600], "evidence": evidence})
    items = list({(c["source_type"], c["source_id"]): c for c in cases}.values())[: max(1, int(limit))]
    if persist:
        ts = now_kst()
        with connect() as conn:
            for c in items:
                conn.execute("""
                    INSERT INTO failure_cases (created_at, updated_at, source_type, source_id, task_id, title, category, summary, evidence_json, strategy_json, resolved, archived)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
                    ON CONFLICT(source_type, source_id) DO UPDATE SET updated_at=excluded.updated_at, task_id=excluded.task_id, title=excluded.title, category=excluded.category, summary=excluded.summary, evidence_json=excluded.evidence_json, strategy_json=excluded.strategy_json, archived=0
                """, (ts, ts, c["source_type"], c["source_id"], c.get("task_id"), c["title"][:220], c["category"], c["summary"], _json(c["evidence"]), _json(failure_strategy(c["category"]))))
            conn.commit()
    return {"persisted": persist, "count": len(items), "categories": dict(Counter(c["category"] for c in items)), "items": items}


def _failure_group_key(row: dict[str, Any]) -> str:
    text = f"{row.get('category') or ''} {row.get('summary') or ''} {row.get('title') or ''}".lower()
    text = re.sub(r"#?\d+", " ", text)
    text = re.sub(r"\b(executing|verifying|reporting|planning|queued|blocked|failed|timeout)\b", " ", text)
    text = re.sub(r"[^0-9a-z_\uac00-\ud7a3]+", " ", text)
    return _fp(row.get("category"), ",".join(_terms(text, 10)))


def circuit_breaker_snapshot(*, limit: int = 80, threshold: int = 3, cooldown_seconds: int = 1800, persist: bool = False) -> dict[str, Any]:
    init_db()
    if persist:
        index_failure_cases(limit=limit, persist=True)
    with connect() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM failure_cases WHERE archived=0 AND resolved=0 ORDER BY updated_at DESC, id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_failure_group_key(r)].append(r)
    items = []
    for key, group in groups.items():
        cat = str(group[0].get("category") or "unknown")
        status = "open" if len(group) >= max(1, int(threshold)) and cat != "success" else "closed"
        item = {"breaker_key": key, "status": status, "category": cat, "failure_count": len(group), "threshold": max(1, int(threshold)), "cooldown_seconds": max(60, int(cooldown_seconds)), "sample_titles": [str(r.get("title") or "")[:120] for r in group[:4]], "recommendation": failure_strategy(cat)["recovery_hint"]}
        if persist:
            ts = now_kst()
            with connect() as conn:
                conn.execute("""
                    INSERT INTO circuit_breakers (created_at, updated_at, breaker_key, status, category, failure_count, threshold_count, cooldown_seconds, last_failure_at, opened_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CASE WHEN ?='open' THEN ? ELSE NULL END, ?)
                    ON CONFLICT(breaker_key) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status, category=excluded.category, failure_count=excluded.failure_count, threshold_count=excluded.threshold_count, cooldown_seconds=excluded.cooldown_seconds, last_failure_at=excluded.last_failure_at, opened_at=CASE WHEN excluded.status='open' THEN COALESCE(circuit_breakers.opened_at, excluded.opened_at) ELSE NULL END, metadata_json=excluded.metadata_json
                """, (ts, ts, key, status, cat, len(group), max(1, int(threshold)), max(60, int(cooldown_seconds)), group[0].get("updated_at"), status, ts, _json({"sample_titles": item["sample_titles"], "recommendation": item["recommendation"]})))
                conn.commit()
        items.append(item)
    items.sort(key=lambda x: (x["status"] == "open", x["failure_count"]), reverse=True)
    return {"persisted": persist, "threshold": max(1, int(threshold)), "open_count": sum(1 for x in items if x["status"] == "open"), "items": items}


def behavior_tree_for_goal(*, goal_id: int | None = None, query: str | None = None) -> dict[str, Any]:
    init_db()
    goal = get_goal(int(goal_id)) if goal_id else None
    if goal is None and query:
        goal = {"id": None, "title": query, "description": query, "goal_type": "ad_hoc", "metadata_json": _json({"priority_owner": "user"})}
    if goal is None:
        goals = [g for g in list_goals(limit=20) if str(g.get("status")) in {"active", "proposed", "blocked", "waiting_approval"}]
        goal = goals[0] if goals else {"id": None, "title": "ad hoc goal", "description": "decompose a goal into safe executable steps", "goal_type": "ad_hoc", "metadata_json": _json({"priority_owner": "core"})}
    plan = htn_plan_for_goal(goal)
    nodes = [{"id": f"{i:02d}_{s['phase']}", "type": "sequence_step", "phase": s["phase"], "task": s["task"], "output": s["output"], "guard": "정책/권한/검증 조건 통과", "on_failure": plan["fallback_policy"]["return_to_phase"] if s["phase"] in {"route", "verify"} else "observe"} for i, s in enumerate(plan["steps"], 1)]
    return {"goal_id": goal.get("id"), "title": goal.get("title"), "objective": plan["objective"], "tree_type": "safe_htn_behavior_tree", "root": {"type": "sequence", "name": "목표를 안전하게 끝까지 처리"}, "nodes": nodes, "fallback_policy": plan["fallback_policy"]}


def advanced_learning_snapshot(*, persist: bool = False) -> dict[str, Any]:
    return {"persisted": persist, "memory_hierarchy": build_memory_hierarchy(persist=persist), "graph_summary": summarize_cognitive_graph(persist=persist), "failure_cases": index_failure_cases(persist=persist), "circuit_breakers": circuit_breaker_snapshot(persist=persist), "behavior_tree": behavior_tree_for_goal()}
