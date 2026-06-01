import json

from agent.cli.agentctl import main
from agent.core.metrics import collect_metrics


METRIC_KEYS = {
    "events_count",
    "memories_count",
    "open_goals_count",
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
}


def test_collect_metrics_shape():
    metrics = collect_metrics()
    assert set(metrics) == METRIC_KEYS
    for key in METRIC_KEYS - {"last_eval_result", "last_eval_score"}:
        assert isinstance(metrics[key], (int, float))


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
