import pytest


@pytest.fixture(autouse=True)
def default_test_renderer(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
