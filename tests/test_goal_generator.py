import json

from agent.cli.agentctl import main
from agent.core.database import init_db
from agent.core.goal_generator import add_root_objective, generate_goal_candidates, list_goal_candidates, list_root_objectives, meaningful_open_goals, seed_default_objectives
from agent.core.goals import cleanup_noise_goals, create_goal, list_goals


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    init_db()


def test_objective_seed_is_idempotent(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    first = seed_default_objectives()
    second = seed_default_objectives()
    rows = list_root_objectives(include_disabled=True)
    assert len(first) == 7
    assert len(second) == 7
    assert len(rows) == 7
    assert {row["objective_type"] for row in rows} >= {"self_maintenance", "system_observation", "workspace_experiment"}


def test_goal_generate_dry_run_creates_candidates_but_no_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = generate_goal_candidates(dry_run=True)
    assert result["created_goal_id"] is None
    assert result["reason"] == "dry_run"
    assert 1 <= len(result["candidates"]) <= 3
    assert list_goals(limit=10) == []
    assert list_goal_candidates(limit=10)


def test_goal_generate_creates_at_most_one_proposed_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = generate_goal_candidates()
    assert result["created_goal_id"] is not None
    goals = list_goals(limit=10)
    proposed = [goal for goal in goals if goal["status"] == "proposed"]
    assert len(proposed) == 1
    assert proposed[0]["id"] == result["created_goal_id"]


def test_no_generation_if_meaningful_open_goal_exists(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    create_goal("Write a real project report", "meaningful", goal_type="reporting", status="active", dedupe=False)
    result = generate_goal_candidates()
    assert result["created_goal_id"] is None
    assert result["reason"] == "meaningful_open_goal_exists"


def test_noise_goals_do_not_block_generation(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    create_goal("Answer user input", "noise", goal_type="answer_user", status="active", dedupe=False)
    assert meaningful_open_goals() == []
    result = generate_goal_candidates()
    assert result["created_goal_id"] is not None


def test_duplicate_candidate_is_suppressed(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    dry_run = generate_goal_candidates(dry_run=True)
    title = dry_run["candidates"][0]["title"]
    goal_type = dry_run["candidates"][0]["goal_type"]
    add_root_objective("Duplicate source", "duplicate", "duplicate_source", priority=1.0, cooldown_seconds=0)
    result = generate_goal_candidates(dry_run=True)
    titles = [candidate["title"] for candidate in result["candidates"]]
    assert title not in titles or all(candidate["goal_type"] != goal_type for candidate in result["candidates"] if candidate["title"] == title)
    assert any(row.get("rejection_reason") == "duplicate_recent_candidate" for row in list_goal_candidates(limit=20))


def test_cooldown_suppresses_repeated_objective(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    first = generate_goal_candidates()
    assert first["created_goal_id"] is not None
    assert main(["goal", "done", str(first["created_goal_id"])]) == 0
    second = generate_goal_candidates(dry_run=True)
    used_objective = first["candidates"][0]["root_objective_id"]
    assert all(candidate["root_objective_id"] != used_objective for candidate in second["candidates"])


def test_goal_generator_cli(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    assert main(["objective", "seed"]) == 0
    seeded = json.loads(capsys.readouterr().out)
    assert len(seeded) == 7
    assert main(["objective", "list"]) == 0
    objectives = json.loads(capsys.readouterr().out)
    assert objectives
    assert main(["goal", "generate", "--dry-run"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert generated["reason"] == "dry_run"
    assert main(["goal", "candidates", "--limit", "3"]) == 0
    candidates = json.loads(capsys.readouterr().out)
    assert candidates


def test_meaningful_open_goal_filter_sees_past_many_noise_goals(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    for index in range(80):
        create_goal(f"secret goal summary marker {index}", "noise", goal_type="test", status="active", dedupe=False)
    meaningful_id = create_goal("Generate a real observation report", "meaningful", goal_type="system_observation", status="proposed", dedupe=False)
    open_goals = meaningful_open_goals()
    assert any(goal["id"] == meaningful_id for goal in open_goals)
    result = generate_goal_candidates()
    assert result["created_goal_id"] is None
    assert result["reason"] == "meaningful_open_goal_exists"


def test_cleanup_noise_goals_marks_noise_done_and_redacts(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    noise_id = create_goal(
        "secret goal summary marker",
        "description contains DISCORD_BOT_TOKEN=abc123",
        goal_type="test",
        status="active",
        metadata={"token": "abc123"},
        dedupe=False,
    )
    real_id = create_goal("Generate a real report", "meaningful", goal_type="reporting", status="active", dedupe=False)

    result = cleanup_noise_goals()

    goals = {goal["id"]: goal for goal in list_goals(limit=10, include_archived=True)}
    assert result["marked_done"] == 1
    assert noise_id in result["noise_goal_ids"]
    assert goals[noise_id]["status"] == "done"
    assert goals[noise_id]["description"] == "redacted test/noise goal"
    assert goals[noise_id]["metadata_json"] == "{}"
    assert goals[real_id]["status"] == "active"


def test_cleanup_noise_cli(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    create_goal("secret goal summary marker", "description contains DISCORD_BOT_TOKEN=abc123", goal_type="test", status="active", metadata={"token": "abc123"}, dedupe=False)
    assert main(["goal", "cleanup-noise"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["marked_done"] == 1
