import json

from agent.cli.agentctl import main
from agent.core.database import connect, init_db
from agent.core.learner import create_reflection, list_skills
from agent.core.memory_intelligence import compact_memories, promote_skill_candidates, run_memory_intelligence
from agent.memory.store import add_memory, count_archived_memories, search_memories


def test_compact_memories_promotes_repeated_rows_to_summary():
    init_db()
    source_ids = [
        add_memory("self map observation", f"body state sample {idx}", tags=["core", "self_map"], importance=0.54)
        for idx in range(3)
    ]

    dry = compact_memories(min_size=3, limit=5, dry_run=True)
    result = compact_memories(min_size=3, limit=5)
    found = search_memories("self map body state", limit=5)

    assert dry["clusters"]
    assert result["created"]
    assert result["archived_count"] == len(source_ids)
    assert count_archived_memories() == len(source_ids)
    assert any(row["memory_type"] == "summary" for row in found)
    assert all(row["id"] not in source_ids for row in found)


def test_promote_skill_candidates_from_repeated_reflections():
    init_db()
    for _ in range(3):
        create_reflection("Task worker executed one approved local action and recorded the result.")

    dry = promote_skill_candidates(dry_run=True)
    result = promote_skill_candidates()
    skills = list_skills(limit=10)

    assert dry["candidates"]
    assert result["promoted"]
    assert any(skill["name"] == "task_worker_execution_review" for skill in skills)


def test_memory_intelligence_cli_dry_runs(capsys):
    init_db()
    for _ in range(3):
        add_memory("duplicate preference", "user prefers short blunt Korean replies", tags=["talk", "style"])
        create_reflection("Task worker executed one approved local action and recorded the result.")

    assert main(["memory", "compact", "--dry-run"]) == 0
    compact_payload = json.loads(capsys.readouterr().out)
    assert compact_payload["dry_run"] is True

    assert main(["skill", "promote", "--dry-run"]) == 0
    skill_payload = json.loads(capsys.readouterr().out)
    assert skill_payload["dry_run"] is True

    assert main(["intelligence", "maintain", "--dry-run"]) == 0
    maintain_payload = json.loads(capsys.readouterr().out)
    assert maintain_payload["dry_run"] is True


def test_memory_intelligence_maintain_can_run_without_candidates():
    init_db()

    result = run_memory_intelligence(dry_run=False)

    assert result["dry_run"] is False
    assert result["memory"]["created"] == []
    assert result["skills"]["promoted"] == []

