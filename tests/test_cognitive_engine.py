import json

from agent.cli.agentctl import main
from agent.core.cognitive_engine import (
    add_blackboard_item,
    add_stigmergy_marker,
    bayesian_update,
    build_map_elites,
    cognitive_growth_snapshot,
    failure_strategy,
    htn_plan_for_goal,
    list_blackboard_items,
    list_stigmergy_markers,
    outcome_patterns,
)
from agent.core.database import connect, init_db
from agent.core.goals import create_goal
from agent.memory.store import add_memory


def test_bayesian_update_uses_success_failure_evidence():
    strong = bayesian_update(9, 1)
    weak = bayesian_update(1, 9)

    assert strong["expected_success"] > weak["expected_success"]
    assert strong["confidence"] == weak["confidence"]


def test_failure_strategy_maps_categories_to_recovery_routes():
    timeout = failure_strategy("timeout")
    env = failure_strategy("environment_issue")

    assert timeout["route"] == "split"
    assert timeout["retry_recommended"] is True
    assert env["route"] == "dependency_doctor"
    assert env["next_phase"] == "observe"


def test_htn_plan_uses_failure_fallback_from_goal_metadata():
    plan = htn_plan_for_goal({
        "id": 10,
        "title": "Fix tests",
        "description": "pytest 실패 복구",
        "goal_type": "user_directed",
        "metadata_json": '{"failure_category": "verification_failed", "priority_owner": "user"}',
    })

    assert plan["owner"] == "user"
    assert plan["fallback_policy"]["route"] == "repair"
    assert plan["fallback_policy"]["return_to_phase"] == "verify"


def test_map_elites_keeps_best_candidate_per_cell():
    items = [
        {"id": 1, "title": "low score", "score": 0.3, "risk_level": "low", "components": {"novelty": 0.7, "utility": 0.8}},
        {"id": 2, "title": "high score", "score": 0.9, "risk_level": "low", "components": {"novelty": 0.7, "utility": 0.8}},
        {"id": 3, "title": "other cell", "score": 0.5, "risk_level": "high", "components": {"novelty": 0.2, "utility": 0.4}},
    ]

    elites = build_map_elites(items)

    assert len(elites) == 2
    assert elites[0]["candidate"]["id"] == 2


def test_blackboard_and_stigmergy_are_persistent():
    init_db()

    board_id = add_blackboard_item("test", "memory", "check shared workspace", confidence=0.8, tags=["growth"])
    marker_id = add_stigmergy_marker("attention", "goal", 7, "follow this signal", intensity=0.7)

    board = list_blackboard_items()
    markers = list_stigmergy_markers()

    assert board[0]["id"] == board_id
    assert markers[0]["id"] == marker_id


def test_cognitive_growth_snapshot_exposes_nine_algorithms_and_persists():
    init_db()
    add_memory("Core architecture", "Core separates user queue and autonomous loop.", tags=["core"], importance=0.8)
    create_goal("Improve Core reliability", "Make the growth loop observable.", goal_type="self_improvement_proposal", status="active", priority=0.7, dedupe=False)

    result = cognitive_growth_snapshot(persist=True)

    assert len(result["algorithms"]) == 9
    assert result["curiosity"]
    assert result["utility_scoring"]
    assert result["htn_plan"]["steps"]
    assert result["htn_plan"]["fallback_policy"]
    assert "action_execution" in result["bayesian_update"]
    assert "failure_learning" in result
    assert result["active_inference"]["recommended_strategy"]
    assert "snapshot_id" in result
    with connect() as conn:
        row = conn.execute("SELECT * FROM cognitive_snapshots WHERE id = ?", (result["snapshot_id"],)).fetchone()
    assert row is not None


def test_intelligence_growth_cli(capsys):
    init_db()

    assert main(["intelligence", "growth", "--persist", "--limit", "4"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["algorithms"]) == 9
    assert "blackboard" in payload
    assert "stigmergy" in payload


def test_outcome_patterns_exposes_strategy_shape():
    init_db()

    result = outcome_patterns()

    assert "dominant_strategy" in result
    assert result["dominant_strategy"]["route"]
