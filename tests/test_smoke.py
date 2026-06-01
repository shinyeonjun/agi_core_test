from agent.core.database import init_db
from agent.core.state import load_state
from agent.memory.store import add_memory, search_memories


def test_state_and_memory_smoke():
    init_db()
    state = load_state()
    assert state["version"] == "0.1"
    memory_id = add_memory("\ud14c\uc2a4\ud2b8 \uae30\uc5b5", "Core smoke test memory", tags=["test"])
    assert memory_id > 0
    results = search_memories("smoke")
    assert results
