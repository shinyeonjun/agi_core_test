import json

from agent.cli.agentctl import main
from agent.core.cognitive_engine import list_blackboard_items
from agent.core.goals import get_goal
from agent.core.learner import list_reflections
from agent.core.research_ingestion import ingest_research_papers, list_research_paper_seeds
from agent.memory.store import list_memories, search_memories


def test_research_paper_seeds_expose_core_connections():
    papers = list_research_paper_seeds()

    assert len(papers) >= 6
    assert {paper["key"] for paper in papers} >= {"react", "reflexion", "memgpt"}
    assert all(paper["url"].startswith("https://arxiv.org/abs/") for paper in papers)
    assert all(paper["core_connection"] for paper in papers)
    assert all(paper["development_hook"] for paper in papers)


def test_ingest_research_papers_links_memory_reflection_blackboard_and_goal():
    result = ingest_research_papers(limit=2)

    assert result["dry_run"] is False
    assert len(result["created"]) == 2
    assert len(result["reflection_ids"]) == 2
    assert len(result["blackboard_ids"]) == 2

    goal = get_goal(result["goal_id"])
    assert goal is not None
    assert goal["goal_type"] == "self_improvement_proposal"
    assert "research_ingestion" in goal["metadata_json"]

    memories = list_memories(limit=10)
    assert sum(1 for row in memories if row["memory_type"] == "research_paper") == 2
    assert search_memories("verbal reflection core learning", limit=5)
    assert any("Linked research paper to Core learning" in row["summary"] for row in list_reflections(limit=10))
    assert len(list_blackboard_items(limit=10)) == 2


def test_ingest_research_papers_skips_existing_memory():
    first = ingest_research_papers(limit=1)
    second = ingest_research_papers(limit=1)

    assert len(first["created"]) == 1
    assert second["created"] == []
    assert len(second["skipped"]) == 1
    assert second["skipped"][0]["reason"] == "already_ingested"


def test_research_cli_lists_and_dry_runs(capsys):
    assert main(["research", "papers", "--limit", "1"]) == 0
    papers_payload = json.loads(capsys.readouterr().out)
    assert len(papers_payload["items"]) == 1
    assert papers_payload["items"][0]["key"] == "react"

    assert main(["research", "ingest", "--limit", "2", "--dry-run"]) == 0
    ingest_payload = json.loads(capsys.readouterr().out)
    assert ingest_payload["dry_run"] is True
    assert ingest_payload["planned_memory_count"] == 2
    assert list_memories(limit=10) == []
