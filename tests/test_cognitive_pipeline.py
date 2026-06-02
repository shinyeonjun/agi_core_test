from agent.core.cognitive_pipeline import growth_task_decision, maybe_enqueue_growth_task
from agent.core.database import init_db
from agent.core.goals import create_goal, get_goal
from agent.core.task_queue import enqueue_task, list_tasks
from agent.scheduler.idle_policy import run_idle_policy


def _snapshot(mode: str = "explore", topic: str = "memory_hygiene", pressure: float = 0.8) -> dict:
    return {
        "active_inference": {"mode": mode, "free_energy": 0.42, "pressures": {"exploration": pressure}},
        "curiosity": [{"topic": topic, "pressure": pressure, "question": "what should Core inspect?", "evidence": {}}],
    }


def test_growth_pipeline_enqueues_autonomous_task_from_pressure():
    init_db()

    result = maybe_enqueue_growth_task(_snapshot())

    assert result["created_goal_id"]
    assert result["task_id"]
    goal = get_goal(result["created_goal_id"])
    tasks = list_tasks(limit=10, queue_type="autonomous")
    assert goal["goal_type"] == "memory_cleanup"
    assert tasks[0]["id"] == result["task_id"]
    assert tasks[0]["source"] == "cognitive_pipeline"


def test_growth_pipeline_defers_when_user_queue_has_work():
    init_db()
    goal_id = create_goal("User task", "do this first", goal_type="user_directed", metadata={"priority_owner": "user"}, dedupe=False)
    enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="User task", source="test", priority=0.9)

    decision = growth_task_decision(_snapshot())

    assert decision["should_enqueue"] is False
    assert decision["reason"] == "user_queue_has_priority"


def test_idle_policy_records_cognitive_growth_result():
    init_db()

    result = run_idle_policy()

    assert "cognitive_growth" in result
    assert result["cognitive_growth"] is not None
    assert "mode" in result["cognitive_growth"]
