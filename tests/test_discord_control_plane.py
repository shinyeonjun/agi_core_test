import json

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.formatter import format_approval_card, redact_discord_content
from agent.bridge.notifier import post_webhook
from agent.bridge.reports import build_daily_summary, notify_test_summary, notify_test_update
from agent.bridge.router import DiscordEvent, channel_role, route_discord_event
from agent.cli.agentctl import main
from agent.core.approvals import ApprovalStore
from agent.core.database import init_db
from agent.core.policy import PolicyEngine


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.delenv("DISCORD_SUMMARY_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_UPDATE_WEBHOOK_URL", raising=False)
    init_db()


def control_config():
    return DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10", "20"},
        chat_channel_id="10",
        approval_channel_id="20",
        summary_channel_id="30",
        update_channel_id="40",
        user_cooldown_seconds=0,
    )


def test_channel_roles_are_explicit():
    config = control_config()
    assert channel_role("10", config) == "chat"
    assert channel_role("20", config) == "approval"
    assert channel_role("30", config) == "summary"
    assert channel_role("40", config) == "update"
    assert channel_role("99", config) == "other"


def test_chat_channel_routes_to_core(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m1", False, False, "\uc9c0\uae08 \uc0c1\ud0dc \uc54c\ub824\uc918")
    chunks = route_discord_event(event, control_config())
    assert chunks
    assert "Core" in chunks[0] or "core" in chunks[0].lower()


def test_approval_channel_blocks_normal_chat(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "20", "1", "m2", False, False, "\uadf8\ub0e5 \ub300\ud654\ud558\uc790")
    chunks = route_discord_event(event, control_config())
    assert chunks == ["\uc5ec\uae30\ub294 \uc2b9\uc778 \uc804\uc6a9 \ucc44\ub110\uc774\uc57c. `!approvals`, `!approve <id>`, `!reject <id>`\ub9cc \uc0ac\uc6a9\ud560 \uc218 \uc788\uc5b4."]


def test_approval_channel_allows_approval_commands(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    proposal = PolicyEngine().classify_text("apt-get install nginx")
    approval_id = ApprovalStore().create_approval(proposal)
    event = DiscordEvent(None, "20", "1", "m3", False, False, "!approvals")
    output = "\n".join(route_discord_event(event, control_config()))
    assert f"\uc2b9\uc778 \ud544\uc694 #{approval_id}" in output
    assert "proposed_payload_json" not in output
    assert "payload" not in output


def test_webhook_only_channels_do_not_chat(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    config = DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"30"}, summary_channel_id="30", user_cooldown_seconds=0)
    event = DiscordEvent(None, "30", "1", "m4", False, False, "hello")
    output = route_discord_event(event, config)
    assert output
    assert "\uc6f9\ud6c5" in output[0]


def test_summary_and_update_redact_secrets(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    secret = "https://discord.com/api/webhooks/123/abcdef TOKEN=abc123 Authorization: Bearer xyz"
    assert "abcdef" not in redact_discord_content(secret)
    assert "abc123" not in redact_discord_content(secret)
    assert "xyz" not in redact_discord_content(secret)


def test_approval_summary_hides_raw_payload(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    proposal = PolicyEngine().classify_text("apt-get install nginx")
    approval_id = ApprovalStore().create_approval(proposal)
    row = ApprovalStore().list_pending()[0]
    card = format_approval_card(row)
    assert f"#{approval_id}" in card
    assert "proposed_payload_json" not in card
    assert "matched_rules" not in card


def test_missing_webhook_url_does_not_crash(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = post_webhook("summary", "hello")
    assert result["sent"] is False
    assert result["reason"] == "missing_webhook_url"


def test_notify_dry_run_cli(capsys, monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    assert main(["notify", "test-summary", "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reason"] == "missing_webhook_url"
    assert main(["notify", "test-update", "--dry-run"]) == 0
    update = json.loads(capsys.readouterr().out)
    assert update["reason"] == "missing_webhook_url"


def test_daily_summary_is_human_readable(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = build_daily_summary()
    assert "**Core \uc694\uc57d**" in text
    assert "{" not in text
    assert "proposed_payload_json" not in text
