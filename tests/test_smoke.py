from agent.core.database import get_schema_version, init_db
from agent.core.pipeline import run_talk
from agent.core.state import load_state
from agent.memory.store import add_memory, search_memories
from agent.scheduler.tick import run_tick


def test_state_and_memory_smoke():
    init_db()
    assert get_schema_version() == "0.16.0-alpha"
    state = load_state()
    assert state["version"] == "0.16"
    memory_id = add_memory("test memory", "Core smoke test memory", tags=["test", "core"])
    assert memory_id > 0
    results = search_memories("smoke")
    assert results
    assert "score" in results[0]


def test_talk_pipeline_creates_v013_output(monkeypatch):
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    result = run_talk("Core next step?")
    assert "v0.16" in result["text"]
    assert result["decision"]["version"] == "0.16"
    assert result["validation"]["ok"] is True


def test_tick_creates_reflection():
    result = run_tick()
    assert "reflection_id" in result
    assert result["reflection_id"] > 0


def test_korean_memory_search_uses_unicode_terms():
    memory_id = add_memory("digital agi korean", "Core remembers \ub514\uc9c0\ud138 AGI context.", tags=["digital_agi", "core"])
    results = search_memories("\ub514\uc9c0\ud138 AGI")
    assert any(row["id"] == memory_id for row in results)


def test_memory_fts_can_update_usage_after_search():
    init_db()
    memory_id = add_memory("fts usage", "digital agi core memory", tags=["digital_agi", "core"])

    first = search_memories("digital agi core", limit=5)
    second = search_memories("digital agi core", limit=5)

    assert any(row["id"] == memory_id for row in first)
    assert any(row["id"] == memory_id for row in second)
