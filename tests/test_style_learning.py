import json

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.cli.agentctl import main
from agent.core.database import init_db
from agent.core.goals import list_goals
from agent.core.policy import PolicyEngine
from agent.core.style import apply_style_feedback, get_active_style_profile, list_style_feedback
from agent.core.user_goals import is_user_goal_request


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    init_db()


def control_config():
    return DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10", "20"},
        chat_channel_id="10",
        approval_channel_id="20",
        user_cooldown_seconds=0,
    )


def test_style_feedback_updates_active_profile(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = apply_style_feedback("더 짧게 말해줘")

    assert result is not None
    assert result["feedback_type"] == "shorter"
    profile = get_active_style_profile()["profile"]
    assert profile["detail_level"] == "shorter"
    rows = list_style_feedback(5)
    assert rows
    assert rows[0]["feedback_type"] == "shorter"


def test_style_feedback_does_not_become_user_directed_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    assert is_user_goal_request("더 짧게 말해줘") is False
    assert is_user_goal_request("너무 AI같음. 좀 덜 정중하게.") is False


def test_discord_style_feedback_reply_is_conversational(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m-style", False, False, "더 짧게 말해줘")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "말투 피드백" in output
    assert "실행/정책 판단" in output
    assert "fallback renderer" not in output
    assert not any(goal["goal_type"] == "user_directed" for goal in list_goals(limit=10, include_archived=True))


def test_style_cli_show_and_feedback(capsys, monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["style", "feedback", "냉정하게 팩트 위주로 말해줘"]) == 0
    feedback = json.loads(capsys.readouterr().out)
    assert feedback["feedback_type"] == "colder"

    assert main(["style", "show"]) == 0
    profile = json.loads(capsys.readouterr().out)
    assert profile["profile"]["structure"] == "findings_first"
    assert any("policy" in item for item in profile["directives"])


def test_style_learning_does_not_relax_policy(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    apply_style_feedback("더 과감하고 직설적으로 말해줘")

    proposal = PolicyEngine().classify_text("rm -rf /")

    assert proposal.denied_reason is not None
    assert proposal.requires_approval is True
