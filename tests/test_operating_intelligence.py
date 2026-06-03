import json

from agent.bridge.reports import build_activity_summary
from agent.cli.agentctl import main
from agent.core.autonomy import set_autonomy_profile
from agent.core.database import connect, init_db
from agent.core.goals import create_goal, get_goal
from agent.core.learner import create_reflection
from agent.core.operating_intelligence import action_critics, memory_hygiene_candidates, operating_snapshot, ranked_goals, refresh_goal_priorities, skill_candidates
from agent.memory.store import add_memory
from agent.tools.full_device import run_action


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def test_user_goal_priority_beats_autonomous_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    user_goal = create_goal("User important task", "do this", goal_type="user_directed", status="active", priority=0.5, metadata={"priority_owner": "user"}, dedupe=False)
    auto_goal = create_goal("Autonomous note", "note", goal_type="research_note", status="active", priority=0.8, dedupe=False)

    result = refresh_goal_priorities()
    ranked = ranked_goals()

    assert result["changed"] >= 1
    assert ranked[0]["id"] == user_goal
    assert get_goal(user_goal)["priority"] > get_goal(auto_goal)["priority"]


def test_action_critic_classifies_profile_block(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")
    run_action("printf blocked", cwd=str(tmp_path))

    critics = action_critics(limit=5)

    assert critics[0]["category"] == "profile_block"
    assert "권한" in critics[0]["recommendation"] or "승인" in critics[0]["recommendation"]


def test_memory_hygiene_finds_duplicate_titles(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    add_memory("same memory", "first", tags=["core"], importance=0.5)
    add_memory("same memory", "second", tags=["core"], importance=0.5)

    items = memory_hygiene_candidates()

    assert items
    assert items[0]["action"] == "merge"


def test_skill_candidates_from_repeated_reflections(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    for _ in range(3):
        create_reflection("Task worker executed one approved local action and recorded the result.", learned={"status": "completed"})

    items = skill_candidates()

    assert items
    assert items[0]["name"] == "task_worker_execution_review"


def test_operating_snapshot_can_persist_review(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = operating_snapshot(persist=True)

    assert "review_id" in result
    with connect() as conn:
        row = conn.execute("SELECT * FROM operating_reviews WHERE id = ?", (result["review_id"],)).fetchone()
    assert row is not None


def test_operating_snapshot_is_read_only_by_default(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("User readable task", "observe only", goal_type="user_directed", status="active", priority=0.4, metadata={"priority_owner": "user"}, dedupe=False)

    result = operating_snapshot()

    assert result["goal_priorities"]["refreshed"] is False
    assert get_goal(goal_id)["priority"] == 0.4


def test_intelligence_cli_and_summary(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    assert main(["intelligence", "snapshot", "--persist"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "goal_priorities" in payload
    assert "action_critics" in payload

    assert main(["intelligence", "priorities", "--refresh"]) == 0
    priorities = json.loads(capsys.readouterr().out)
    assert "items" in priorities

    text = build_activity_summary()
    assert "운영 판단" in text
    assert "자율 개선 방향" in text
