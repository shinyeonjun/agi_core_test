import json

from agent.cli.agentctl import main
from agent.core.advanced_learning import behavior_tree_for_goal, build_memory_hierarchy, circuit_breaker_snapshot, index_failure_cases, summarize_cognitive_graph
from agent.core.database import connect, init_db
from agent.core.goals import create_goal
from agent.core.task_lifecycle import record_task_phase
from agent.core.task_queue import enqueue_task, finish_task
from agent.memory.store import add_memory


def test_memory_hierarchy_persists_rollups():
    init_db()
    for idx in range(4):
        add_memory("renderer fallback", f"fallback should hide internal field {idx}", memory_type="fact", tags=["renderer", "fallback"], importance=0.8)

    result = build_memory_hierarchy(limit=20, persist=True)

    assert result["clusters"]
    assert result["roots"]
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM memory_rollups").fetchone()["c"]
    assert count >= 2


def test_graph_summary_persists_communities():
    init_db()
    goal_id = create_goal("Improve renderer fallback", "Keep fallback replies natural.", goal_type="self_improvement", priority=0.8, dedupe=False)
    add_memory("renderer fallback", "fallback reply should not expose selected_goal_id", tags=["renderer", "fallback"], importance=0.85)
    enqueue_task("user", goal_id=goal_id, task_kind="codex_work", title="Improve renderer fallback", source="test", priority=0.8, dedupe_goal=False)

    result = summarize_cognitive_graph(limit=40, persist=True)

    assert result["communities"]
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS c FROM graph_community_summaries").fetchone()["c"]
    assert count >= 1


def test_failure_index_and_circuit_breaker():
    init_db()
    goal_id = create_goal("Repair blocked worker", "Blocked task should be explained.", goal_type="self_improvement", priority=0.8, dedupe=False)
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="codex_work", title="Repair blocked worker", source="test", priority=0.8, dedupe_goal=False)
    record_task_phase(task_id, "reporting", "blocked", "pytest verification_failed after code change", queue_type="user", metadata={"reason": "verification_failed"})
    finish_task(task_id, "blocked", {"status": "blocked", "reason": "verification_failed", "queue_type": "user"})

    failures = index_failure_cases(limit=20, persist=True)
    circuit = circuit_breaker_snapshot(limit=20, threshold=1, persist=True)

    assert failures["count"] >= 1
    assert circuit["open_count"] >= 1
    with connect() as conn:
        stored = conn.execute("SELECT COUNT(*) AS c FROM circuit_breakers WHERE status='open'").fetchone()["c"]
    assert stored >= 1


def test_behavior_tree_and_cli(capsys):
    init_db()
    goal_id = create_goal("Build memory hierarchy", "Create rollups and verify them.", goal_type="self_improvement", priority=0.8, dedupe=False)

    tree = behavior_tree_for_goal(goal_id=goal_id)
    assert tree["nodes"][0]["phase"] == "observe"
    assert tree["fallback_policy"]

    assert main(["intelligence", "htn", "--goal-id", str(goal_id)]) == 0
    htn_payload = json.loads(capsys.readouterr().out)
    assert htn_payload["goal_id"] == goal_id

    add_memory("graph retrieval", "memory hierarchy helps retrieval", tags=["memory"], importance=0.7)
    assert main(["intelligence", "hierarchy", "--limit", "10", "--persist"]) == 0
    hierarchy_payload = json.loads(capsys.readouterr().out)
    assert hierarchy_payload["persisted"] is True

    assert main(["intelligence", "advanced"]) == 0
    advanced_payload = json.loads(capsys.readouterr().out)
    assert "memory_hierarchy" in advanced_payload