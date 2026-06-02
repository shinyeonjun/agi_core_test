import json

from agent.cli.agentctl import main
from agent.config.defaults import now_kst
from agent.core.database import connect, init_db
from agent.core.metrics import collect_metrics


METRIC_KEYS = {
    "events_count",
    "memories_count",
    "open_goals_count",
    "meaningful_open_goals_count",
    "noise_open_goals_count",
    "pending_approvals_count",
    "reflections_count",
    "skills_count",
    "last_eval_result",
    "last_eval_score",
    "renderer_success_rate",
    "renderer_fallback_rate",
    "policy_critical_count_24h",
    "discord_messages_24h",
    "tick_count_24h",
    "workspace_artifact_count",
    "action_runs_count",
    "action_proposals_count",
    "repeated_action_suppressed_count",
    "self_map_count",
    "queued_user_tasks_count",
    "queued_autonomous_tasks_count",
    "action_success_rate_24h",
    "action_execution_success_rate_24h",
    "action_planned_block_rate_24h",
    "action_executed_count_24h",
    "action_successful_count_24h",
    "action_timeout_count_24h",
    "action_blocked_count_24h",
    "full_device_lab_enabled",
    "current_autonomy_profile",
    "last_action_at",
    "last_action_status",
    "catastrophic_actions_count",
}


def test_collect_metrics_shape():
    metrics = collect_metrics()
    assert set(metrics) == METRIC_KEYS
    numeric_or_bool = METRIC_KEYS - {"last_eval_result", "last_eval_score", "current_autonomy_profile", "last_action_at", "last_action_status"}
    for key in numeric_or_bool:
        assert isinstance(metrics[key], (int, float, bool))
    assert metrics["current_autonomy_profile"] in {"safe", "workspace", "full_device_lab"}


def test_metrics_cli_json_shape(capsys):
    assert main(["metrics", "--json"]) == 0
    output = capsys.readouterr().out
    metrics = json.loads(output)
    assert set(metrics) == METRIC_KEYS


def test_metrics_cli_text_shape(capsys):
    assert main(["metrics"]) == 0
    output = capsys.readouterr().out
    assert "events_count:" in output
    assert "tick_count_24h:" in output
    assert "workspace_artifact_count:" in output
    assert "action_success_rate_24h:" in output
    assert "action_execution_success_rate_24h:" in output


def test_action_rates_separate_planned_blocks_from_execution_failures():
    init_db()
    created_at = now_kst()
    rows = [
        ("completed", 0, "rc=0"),
        ("completed", 0, "rc=0"),
        ("timeout", None, "timeout"),
        ("blocked", None, "profile_not_full_device_lab"),
    ]
    with connect() as conn:
        conn.executemany(
            """
            INSERT INTO action_runs (
                created_at, completed_at, action_type, command_json, cwd, profile,
                risk_level, status, returncode, result_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (created_at, created_at, "local_command", "[\"true\"]", None, "full_device_lab", "low", status, returncode, summary)
                for status, returncode, summary in rows
            ],
        )
        conn.commit()

    metrics = collect_metrics()

    assert metrics["action_success_rate_24h"] == 0.5
    assert metrics["action_execution_success_rate_24h"] == 0.6667
    assert metrics["action_planned_block_rate_24h"] == 0.25
    assert metrics["action_executed_count_24h"] == 3
    assert metrics["action_successful_count_24h"] == 2
