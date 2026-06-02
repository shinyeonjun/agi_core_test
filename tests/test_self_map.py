import json
from datetime import datetime, timedelta

from agent.bridge.reports import build_activity_summary
from agent.cli.agentctl import main
from agent.config.defaults import KST
from agent.core.database import connect, init_db
from agent.core.decision import build_talk_decision
from agent.core.self_map import latest_self_map, refresh_self_map, self_map_brief
from agent.scheduler.idle_policy import run_idle_policy


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    init_db()


def test_self_map_refresh_stores_safe_snapshot(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "secret-token-value")

    result = refresh_self_map()
    latest = latest_self_map()
    brief = self_map_brief()
    raw = json.dumps(result, ensure_ascii=False)

    assert result["id"] > 0
    assert latest is not None
    assert brief is not None
    assert brief["summary"] == result["summary"]
    assert "secret-token-value" not in raw
    assert "DISCORD_BOT_TOKEN" not in json.dumps(brief, ensure_ascii=False)
    assert result["snapshot"]["config_presence"]["DISCORD_BOT_TOKEN"] is True


def test_self_map_cli_show_and_refresh(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["self-map", "show"]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["available"] is False

    assert main(["self-map", "refresh"]) == 0
    refreshed = json.loads(capsys.readouterr().out)
    assert refreshed["id"] > 0
    assert "summary" in refreshed

    assert main(["self-map", "show"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["id"] == refreshed["id"]


def test_idle_policy_refreshes_self_map_with_cooldown(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    first = run_idle_policy()
    second = run_idle_policy()

    assert first["self_map_id"] is not None
    assert second["self_map_id"] is None
    assert "self_map_refresh" in str(second["skipped_reason"]) or second["skipped_reason"] is not None


def test_decision_and_summary_use_latest_self_map(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    refresh_self_map()

    decision = build_talk_decision("너 OS 뭐야?")
    summary = build_activity_summary()

    assert decision["runtime_self_map"] is not None
    assert decision["runtime_self_map"]["summary"]
    assert "self-map" in summary
    assert "DISCORD_BOT_TOKEN" not in summary


def test_self_map_brief_refreshes_stale_snapshot(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    old_created = (datetime.now(KST) - timedelta(hours=2)).isoformat(timespec="seconds")
    stale_snapshot = {
        "created_at": old_created,
        "summary": "stale self-map",
        "host": {"os_release": {"pretty_name": "old"}, "hostname": "old", "machine": "old"},
        "paths": {},
        "git": {},
        "services": {},
        "autonomy": {},
        "database": {},
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO self_maps (created_at, summary, snapshot_json, fingerprint, changed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (old_created, "stale self-map", json.dumps(stale_snapshot), "stale", 1),
        )
        conn.commit()

    brief = self_map_brief(max_age_seconds=60, refresh_if_stale=True)

    assert brief is not None
    assert brief["id"] == 2
    assert brief["summary"] != "stale self-map"
