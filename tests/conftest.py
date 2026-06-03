from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_DISCORD_LOCK_PATH", str(tmp_path / "discord_bot.pid"))
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.delenv("DISCORD_SUMMARY_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_UPDATE_WEBHOOK_URL", raising=False)
