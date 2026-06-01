import json

from agent.cli.agentctl import main
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
    "action_success_rate_24h",
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
