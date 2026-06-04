from agent.core.database import init_db
from agent.core.talk_decision_builder import assemble_talk_decision, collect_talk_decision_context, selected_chat_renderer_name


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", " fallback ")
    init_db()


def test_selected_chat_renderer_name_normalizes_env(monkeypatch):
    monkeypatch.setenv("AGENT_CHAT_RENDERER", " Fallback ")

    assert selected_chat_renderer_name() == "fallback"


def test_talk_decision_context_can_be_assembled(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    context = collect_talk_decision_context("Core status?", source_event_id=7)
    decision = assemble_talk_decision(context)

    assert context.renderer_name == "fallback"
    assert context.source_event_id == 7
    assert decision["kind"] == "talk_response"
    assert decision["source_event_id"] == 7
    assert decision["renderer"] == "fallback"
    assert decision["decision_schema"]["kind"] == "talk_response"
    assert decision["decision_trace"]["guards"]
    assert decision["routing"]["memory"]["strategy"] == "fts_sparse_hybrid"
