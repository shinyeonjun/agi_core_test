import json

from agent.cli.agentctl import main
from agent.core.cognitive_graph import activate_graph, graph_snapshot, sync_graph, trace_decision
from agent.core.database import connect, init_db
from agent.core.goals import create_goal
from agent.core.learner import upsert_skill
from agent.core.policy import PolicyEngine
from agent.core.self_map import refresh_self_map
from agent.core.task_queue import enqueue_task
from agent.memory.store import add_memory


def _seed_graph_sources():
    init_db()
    memory_id = add_memory(
        "renderer fallback repair",
        "fallback replies should not expose internal fields.",
        memory_type="fact",
        tags=["renderer", "core"],
        importance=0.9,
    )
    goal_id = create_goal(
        "Improve renderer fallback quality",
        "Reduce internal-field leakage in fallback replies.",
        goal_type="self_improvement",
        priority=0.85,
        dedupe=False,
    )
    task_id = enqueue_task(
        "user",
        goal_id=goal_id,
        task_kind="codex_work",
        title="Improve renderer fallback quality",
        source="test",
        priority=0.9,
        dedupe_goal=False,
    )
    skill_id = upsert_skill("renderer_repair", "fallback renderer repair", ["inspect", "patch", "test"], tags=["renderer"])
    engine = PolicyEngine(); engine.record_decision(engine.classify_decision("sudo apt install nginx"))
    refresh_self_map(record_event=False)
    return {"memory_id": memory_id, "goal_id": goal_id, "task_id": task_id, "skill_id": skill_id}


def test_cognitive_graph_sync_creates_core_node_types_and_edges():
    ids = _seed_graph_sources()

    result = sync_graph(limit=30)

    assert result["after"]["nodes"] >= 5
    with connect() as conn:
        node_types = {row["node_type"] for row in conn.execute("SELECT DISTINCT node_type FROM cognitive_nodes").fetchall()}
        task_edge = conn.execute(
            """
            SELECT e.* FROM cognitive_edges e
            JOIN cognitive_nodes a ON a.id = e.from_node_id
            JOIN cognitive_nodes b ON b.id = e.to_node_id
            WHERE a.node_type = 'task' AND b.node_type = 'goal' AND e.edge_type = 'depends_on'
            """
        ).fetchone()
        goal_node = conn.execute("SELECT * FROM cognitive_nodes WHERE node_type = 'goal' AND source_id = ?", (str(ids["goal_id"]),)).fetchone()
    assert {"memory", "goal", "task", "skill", "policy_decision", "device_state"}.issubset(node_types)
    assert task_edge is not None
    assert goal_node is not None


def test_cognitive_graph_activation_and_trace_persist_reason_path():
    _seed_graph_sources()
    sync_graph(limit=30)

    activated = activate_graph("renderer fallback repair", limit=5)
    trace = trace_decision("reply", "why renderer repair was selected", query="renderer fallback repair")

    assert activated["items"]
    assert activated["items"][0]["scores"]["activation"] is not None
    assert trace["trace_id"] > 0
    with connect() as conn:
        activation_count = conn.execute("SELECT COUNT(*) AS count FROM cognitive_activations").fetchone()["count"]
        trace_row = conn.execute("SELECT trace_json FROM cognitive_traces WHERE id = ?", (trace["trace_id"],)).fetchone()
    assert activation_count >= 5
    payload = json.loads(trace_row["trace_json"])
    assert payload["activated"]


def test_cognitive_graph_cli_sync_snapshot_activate_and_trace(capsys):
    _seed_graph_sources()

    assert main(["intelligence", "graph", "sync", "--limit", "30"]) == 0
    sync_payload = json.loads(capsys.readouterr().out)
    assert sync_payload["after"]["nodes"] > 0

    assert main(["intelligence", "graph", "snapshot", "--limit", "3"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["counts"]["nodes"] > 0

    assert main(["intelligence", "graph", "activate", "renderer fallback", "--limit", "3"]) == 0
    activation = json.loads(capsys.readouterr().out)
    assert activation["items"]

    assert main(["intelligence", "graph", "trace", "why this task was selected", "--query", "renderer fallback", "--trace-type", "goal_selection"]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["trace_type"] == "goal_selection"
