from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any

from agent.core.database import connect, init_db
from agent.core.events import log_event
from agent.core.learner import create_reflection, upsert_skill
from agent.core.operating_intelligence import skill_candidates
from agent.memory.store import add_memory, archive_memories, rebuild_memory_fts


def _decode_json(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _cluster_key(row: dict[str, Any]) -> str:
    title = _norm_text(row.get("title"))
    memory_type = str(row.get("memory_type") or "fact")
    tags = sorted(str(tag).lower() for tag in _decode_json(row.get("tags_json"), []) if str(tag).strip())
    tag_hint = ",".join(tags[:4])
    return f"{memory_type}|{title}|{tag_hint}"


def _summary_title(rows: list[dict[str, Any]]) -> str:
    title = str(rows[0].get("title") or "memory cluster").strip()
    return f"summary: {title[:120]}"


def _summary_type(rows: list[dict[str, Any]]) -> str:
    memory_type = str(rows[0].get("memory_type") or "fact")
    if memory_type == "failure":
        return "failure_summary"
    if any("style" in str(tag).lower() or "talk" in str(tag).lower() for row in rows for tag in _decode_json(row.get("tags_json"), [])):
        return "preference_summary"
    return "summary"


def _summary_tags(rows: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = ["memory_summary", "compacted"]
    for row in rows:
        for tag in _decode_json(row.get("tags_json"), []):
            tag_text = str(tag).strip()
            if tag_text and tag_text not in tags:
                tags.append(tag_text)
    return tags[:12]


def _summary_content(rows: list[dict[str, Any]]) -> str:
    ids = [int(row["id"]) for row in rows]
    counter = Counter(_norm_text(row.get("content")) for row in rows)
    examples = [text for text, _count in counter.most_common(5) if text]
    lines = [
        f"Compacted {len(rows)} related memories.",
        f"source_memory_ids: {ids}",
        "",
        "핵심 요약:",
    ]
    if examples:
        for example in examples:
            lines.append(f"- {example[:220]}")
    else:
        lines.append("- 같은 제목/태그를 가진 반복 기억을 하나의 검색용 요약 기억으로 압축함.")
    return "\n".join(lines)


def duplicate_memory_clusters(*, min_size: int = 3, limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(
            """
            SELECT * FROM memories
            WHERE archived = 0
              AND memory_type NOT IN ('summary', 'preference_summary', 'failure_summary')
            ORDER BY id DESC
            LIMIT 1000
            """
        ).fetchall()]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_cluster_key(row)].append(row)
    clusters: list[dict[str, Any]] = []
    for key, items in groups.items():
        if len(items) < min_size:
            continue
        score = min(1.0, len(items) / 10)
        clusters.append({
            "key": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
            "title": _summary_title(items),
            "memory_type": _summary_type(items),
            "source_ids": [int(row["id"]) for row in items],
            "count": len(items),
            "score": round(score, 4),
            "tags": _summary_tags(items),
        })
    return sorted(clusters, key=lambda row: (row["score"], row["count"]), reverse=True)[:limit]


def compact_memories(*, min_size: int = 3, limit: int = 5, dry_run: bool = False) -> dict[str, Any]:
    clusters = duplicate_memory_clusters(min_size=min_size, limit=limit)
    if dry_run:
        return {"dry_run": True, "clusters": clusters, "created": [], "archived_count": 0}
    created: list[dict[str, Any]] = []
    archived_count = 0
    init_db()
    with connect() as conn:
        by_id = {
            int(row["id"]): dict(row)
            for row in conn.execute(
                "SELECT * FROM memories WHERE archived = 0 AND id IN (%s)" % ",".join("?" for cluster in clusters for _ in cluster["source_ids"]),
                tuple(memory_id for cluster in clusters for memory_id in cluster["source_ids"]),
            ).fetchall()
        } if clusters else {}
    for cluster in clusters:
        rows = [by_id[memory_id] for memory_id in cluster["source_ids"] if memory_id in by_id]
        if len(rows) < min_size:
            continue
        memory_id = add_memory(
            cluster["title"],
            _summary_content(rows),
            memory_type=str(cluster["memory_type"]),
            tags=list(cluster["tags"]),
            importance=max(0.78, min(1.0, max(float(row.get("importance") or 0.5) for row in rows) + 0.08)),
            confidence=max(0.72, min(1.0, sum(float(row.get("confidence") or 0.7) for row in rows) / len(rows))),
        )
        archived_count += archive_memories(cluster["source_ids"], reason="compacted")
        created.append({**cluster, "memory_id": memory_id})
    if created:
        rebuild_memory_fts()
        reflection_id = create_reflection(
            "Compacted duplicate raw memories into summary memories.",
            learned={"created": len(created), "archived_count": archived_count},
            confidence=0.82,
        )
        log_event("memory", "memory_compacted", f"created={len(created)} archived={archived_count}", {"created": created, "reflection_id": reflection_id}, 0.78)
    return {"dry_run": False, "clusters": clusters, "created": created, "archived_count": archived_count}


def promote_skill_candidates(*, limit: int = 5, dry_run: bool = False, min_evidence: int = 3) -> dict[str, Any]:
    candidates = [item for item in skill_candidates(limit=limit * 2) if int(item.get("evidence_count") or 0) >= min_evidence]
    selected = candidates[:limit]
    if dry_run:
        return {"dry_run": True, "candidates": selected, "promoted": []}
    promoted: list[dict[str, Any]] = []
    for item in selected:
        skill_id = upsert_skill(
            str(item["name"]),
            str(item.get("trigger") or item["name"]),
            [str(step) for step in item.get("suggested_procedure") or []],
            tags=["auto_promoted", "reflection_pattern"],
        )
        promoted.append({**item, "skill_id": skill_id})
    if promoted:
        reflection_id = create_reflection(
            "Promoted repeated reflection patterns into reusable skills.",
            learned={"promoted": [item["name"] for item in promoted]},
            confidence=0.84,
        )
        log_event("skill", "skills_promoted", f"promoted={len(promoted)}", {"promoted": promoted, "reflection_id": reflection_id}, 0.8)
    return {"dry_run": False, "candidates": selected, "promoted": promoted}


def run_memory_intelligence(*, dry_run: bool = False) -> dict[str, Any]:
    memory = compact_memories(dry_run=dry_run)
    skills = promote_skill_candidates(dry_run=dry_run)
    return {"dry_run": dry_run, "memory": memory, "skills": skills}
