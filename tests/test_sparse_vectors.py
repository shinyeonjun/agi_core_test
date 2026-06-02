import json

from agent.cli.agentctl import main
from agent.core.database import connect, init_db
from agent.memory.sparse_vector import backfill_memory_vectors, sparse_vector, vector_status
from agent.memory.store import add_memory, search_memories


def test_sparse_vector_is_deterministic_and_normalized():
    first = sparse_vector("오렌지파이 몸 상태 확인")
    second = sparse_vector("오렌지파이 몸 상태 확인")

    assert first == second
    assert first
    assert abs(sum(value * value for value in first.values()) - 1.0) < 0.001


def test_add_memory_creates_vector_and_search_uses_sparse_score():
    memory_id = add_memory("오렌지파이 몸 상태", "보드의 메모리와 디스크 상태를 주기적으로 확인한다.", tags=["core", "self_map"], importance=0.7)

    with connect() as conn:
        row = conn.execute("SELECT * FROM memory_vectors WHERE memory_id = ?", (memory_id,)).fetchone()
    results = search_memories("오렌지파이 몸상태 점검", limit=5)

    assert row is not None
    assert any(item["id"] == memory_id for item in results)
    matched = next(item for item in results if item["id"] == memory_id)
    assert matched["score_components"]["sparse"] > 0


def test_vector_backfill_updates_existing_rows(monkeypatch):
    monkeypatch.setenv("AGENT_SPARSE_VECTOR_ENABLED", "false")
    memory_id = add_memory("대화 말투 선호", "사용자는 짧고 직설적인 반말을 선호한다.", tags=["style", "talk"])
    monkeypatch.setenv("AGENT_SPARSE_VECTOR_ENABLED", "true")

    before = vector_status()
    result = backfill_memory_vectors(limit=10)
    after = vector_status()

    assert result["updated"] >= 1
    assert before["vectorized_memories"] == 0
    assert after["vectorized_memories"] >= 1
    with connect() as conn:
        row = conn.execute("SELECT * FROM memory_vectors WHERE memory_id = ?", (memory_id,)).fetchone()
    assert row is not None


def test_vector_cli_status_backfill_and_search(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_SPARSE_VECTOR_ENABLED", "false")
    add_memory("작업 큐 분리", "사용자 작업과 자율 스케줄러 작업은 분리해서 실행한다.", tags=["task_queue", "core"])
    monkeypatch.setenv("AGENT_SPARSE_VECTOR_ENABLED", "true")

    assert main(["vector", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["enabled"] is True

    assert main(["vector", "backfill", "--limit", "5"]) == 0
    backfill = json.loads(capsys.readouterr().out)
    assert backfill["updated"] >= 1

    assert main(["vector", "search", "사용자 작업 큐", "--limit", "5"]) == 0
    results = json.loads(capsys.readouterr().out)
    assert results
    assert "sparse_similarity" in results[0]


def test_vector_disabled_keeps_memory_search_working(monkeypatch):
    monkeypatch.setenv("AGENT_SPARSE_VECTOR_ENABLED", "false")
    memory_id = add_memory("디스크 상태 점검", "df -h / 결과를 확인한다.", tags=["core"])

    results = search_memories("디스크 상태", limit=5)

    assert any(item["id"] == memory_id for item in results)

