import json

from agent.bridge.formatter import format_action_update
from agent.bridge.reports import build_activity_summary
from agent.cli.agentctl import main
from agent.core.autonomy import set_autonomy_profile
from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.goals import create_goal
from agent.core.observability import action_failure_breakdown, action_observation, task_observation
from agent.core.task_queue import enqueue_task, list_tasks
from agent.renderer.codex_renderer import sanitize_decision_for_renderer
from agent.tools.action_log import get_action_run, list_action_runs
from agent.tools.full_device import run_action


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    init_db()


def test_action_observation_classifies_policy_block(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")

    result = run_action("printf hello", cwd=str(tmp_path))
    observed = action_observation(get_action_run(result["id"]))

    assert observed["category"] == "profile_block"
    assert observed["label"] == "모드 제한"
    assert "full_device_lab" in observed["next_step"]


def test_action_observation_classifies_missing_command(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")

    result = run_action("definitely_missing_agent_core_command", cwd=str(tmp_path))
    observed = action_observation(get_action_run(result["id"]))

    assert result["status"] == "failed"
    assert observed["category"] == "command_not_found"
    assert observed["label"] == "명령 없음"


def test_action_update_includes_classification_and_next_step(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = format_action_update({
        "id": 7,
        "status": "timeout",
        "profile": "full_device_lab",
        "risk_level": "medium",
        "command_json": json.dumps(["sleep", "99"]),
        "result_summary": "timeout",
        "returncode": 124,
    })

    assert "분류: 시간 초과" in text
    assert "다음:" in text
    assert "timeout" in text or "시간" in text


def test_task_observation_explains_queue_waiting(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Autonomous check", "check", goal_type="autonomous", status="active", priority=0.5, dedupe=False)
    task_id = enqueue_task("autonomous", goal_id=goal_id, task_kind="system_check", title="Autonomous check", source="test")
    task = next(row for row in list_tasks(limit=5) if row["id"] == task_id)
    observed = task_observation(task)

    assert "자율 스케줄러" in observed["waiting_reason"]
    assert "lab tick" in observed["next_step"]


def test_decision_trace_is_recorded_and_renderer_safe(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    decision = build_talk_decision("지금 상태 분석해줘")
    sanitized = sanitize_decision_for_renderer(decision)

    assert "decision_trace" in decision
    assert decision["decision_trace"]["facts"]
    assert "decision_trace" in sanitized
    assert "민감정보" in " ".join(decision["decision_trace"]["guards"])


def test_observe_cli_outputs_actions_tasks_and_snapshot(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")
    run_action("printf hello", cwd=str(tmp_path))
    goal_id = create_goal("CLI observe task", "observe", goal_type="user_directed", status="active", priority=0.9, dedupe=False)
    enqueue_task("user", goal_id=goal_id, task_kind="task_note", title="CLI observe task", source="test")

    assert main(["observe", "actions", "--limit", "5"]) == 0
    actions = json.loads(capsys.readouterr().out)
    assert actions["items"][0]["category"] == "profile_block"

    assert main(["observe", "tasks", "--limit", "5"]) == 0
    tasks = json.loads(capsys.readouterr().out)
    assert "waiting_reason" in tasks["items"][0]

    assert main(["observe", "snapshot", "--limit", "5"]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert "action_failure_breakdown" in snapshot


def test_activity_summary_contains_observability_sections(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")
    run_action("printf hello", cwd=str(tmp_path))

    text = build_activity_summary()

    assert "실패/차단 원인" in text
    assert "작업 큐 상태" in text
    assert "profile_not_full_device_lab" not in text
