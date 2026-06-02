from agent.core.database import init_db
from agent.core.pipeline import run_talk


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "core"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    init_db()


def test_talk_decision_has_pipeline_schema_and_routes(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = run_talk("지금 Core 상태 알려줘")
    decision = result["decision"]
    trace = decision["pipeline_trace"]
    schema = decision["decision_schema"]
    routing = decision["routing"]

    assert trace["version"] == "0.17"
    assert trace["missing"] == []
    assert [phase["name"] for phase in trace["phases"]] == trace["phase_order"]
    assert schema["version"] == "0.17"
    assert schema["kind"] == "talk_response"
    assert schema["tool_plan"]
    assert routing["memory"]["strategy"] == "fts_sparse_hybrid"
    assert routing["skills"]["strategy"] == "trigger_similarity"
    assert any(route["name"] == "policy_engine" for route in routing["tools"])
