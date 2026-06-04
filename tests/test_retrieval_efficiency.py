import json

from agent.cli.agentctl import main
from agent.memory.retrieval import adaptive_chunks, mmr_rerank, reciprocal_rank_fusion
from agent.memory.store import add_memory, build_memory_context, search_memories


def test_adaptive_chunks_keeps_chunks_inside_budget():
    text = "\n\n".join([f"section {idx} explains memory retrieval and context packing." * 8 for idx in range(6)])

    chunks = adaptive_chunks(text, target_chars=260, max_chars=420, min_chars=80)

    assert len(chunks) >= 3
    assert all(chunk["quality"]["chars"] <= 420 for chunk in chunks)
    assert all("size_compliance" in chunk["quality"] for chunk in chunks)


def test_reciprocal_rank_fusion_rewards_cross_source_hits():
    fused = reciprocal_rank_fusion([
        [{"id": 1}, {"id": 2}],
        [{"id": 2}, {"id": 3}],
        [{"id": 2}, {"id": 1}],
    ])

    assert fused[2] > fused[1] > fused[3]


def test_mmr_rerank_keeps_relevant_but_diverse_items():
    candidates = [
        {"id": 1, "title": "renderer fallback", "content": "renderer fallback internal field leak", "score": 1.0},
        {"id": 2, "title": "renderer fallback duplicate", "content": "renderer fallback internal field leak duplicate", "score": 0.99},
        {"id": 3, "title": "memory context pack", "content": "memory retrieval context budget", "score": 0.82},
    ]

    picked = mmr_rerank(candidates, query="renderer fallback context", limit=2, diversity=0.45)

    assert picked[0]["id"] == 1
    assert {item["id"] for item in picked} == {1, 3}


def test_memory_search_exposes_rrf_sources_and_uses_diverse_ranking():
    renderer_id = add_memory("renderer fallback", "fallback renderer should hide internal fields", tags=["renderer", "core"], importance=0.8)
    context_id = add_memory("context budget", "context packing should keep diverse memory evidence", tags=["memory", "core"], importance=0.75)
    add_memory("renderer fallback duplicate", "fallback renderer should hide internal fields", tags=["renderer", "core"], importance=0.79)

    results = search_memories("renderer fallback context", limit=3)
    ids = [row["id"] for row in results]

    assert renderer_id in ids
    assert context_id in ids
    assert any(row.get("retrieval_sources") for row in results)
    assert any("rrf" in row.get("score_components", {}) for row in results)


def test_memory_context_pack_respects_budget_and_cli(capsys):
    add_memory("adaptive chunking", "Adaptive chunking picks chunk boundaries by document shape. " * 20, tags=["rag", "memory"], importance=0.9)
    add_memory("graph retrieval", "Graph retrieval connects goals, failures, skills, and memories. " * 20, tags=["graph", "memory"], importance=0.85)

    pack = build_memory_context("memory retrieval", limit=2, max_chars=900, per_item_chars=350)

    assert pack["item_count"] <= 2
    assert pack["used_chars"] <= 900
    assert all(len(item["snippet"]) <= 350 for item in pack["items"])

    assert main(["memory", "context", "memory", "retrieval", "--limit", "2", "--max-chars", "900"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["items"]

    assert main(["memory", "chunks", "? ?????.", "?? ?????.", "--target-chars", "20", "--max-chars", "60"]) == 0
    chunks = json.loads(capsys.readouterr().out)
    assert chunks["chunks"]
