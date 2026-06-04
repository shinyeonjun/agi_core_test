import json

from agent.core.database import connect, init_db
from agent.core.self_report import build_self_report_context, capture_git_change, list_capability_evidence, refresh_capability_evidence, sync_recent_changes


def test_self_report_syncs_git_change_log(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    init_db()

    result = sync_recent_changes(limit=1)

    assert result["count"] >= 1
    with connect() as conn:
        row = conn.execute("SELECT human_summary, changed_files_json FROM core_change_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["human_summary"]
    assert isinstance(json.loads(row["changed_files_json"]), list)


def test_sync_recent_changes_keeps_recorded_verification(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    init_db()
    captured = capture_git_change("HEAD", verification={"tests": ["fast", "release"], "result": "PASS"})

    sync_recent_changes(limit=1)

    with connect() as conn:
        row = conn.execute("SELECT verification_json FROM core_change_log WHERE commit_hash=?", (captured["commit_hash"],)).fetchone()
    assert row is not None
    verification = json.loads(row["verification_json"])
    assert verification["result"] == "PASS"
    assert verification["tests"] == ["fast", "release"]


def test_capability_evidence_includes_research_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    init_db()

    refresh_capability_evidence()
    items = list_capability_evidence(limit=40)
    research = next(item for item in items if item["capability_key"] == "researcher_loop")

    assert research["status"] == "enabled"
    assert "agent/core/research_loop.py" in research["evidence_files"]
    assert "tests/test_research_loop.py" in research["verification_tests"]
    assert research["confidence"] >= 0.8


def test_self_report_context_grounded_for_improvement_questions(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    init_db()
    capture_git_change("HEAD", verification={"tests": ["fast", "learning"], "result": "PASS"})

    context = build_self_report_context("나 많이 개선했는데 어때?", metrics={"renderer_success_rate": 0.88, "last_eval_result": "PASS", "last_eval_score": 1.0, "memory_vector_coverage": 1.0}, limit=5)

    assert context["focus"] == "change"
    assert context["recent_changes"]
    assert context["top_capabilities"]
    assert any("렌더러" in item or "fallback" in item for item in context["weak_points"])
    payload = json.dumps(context, ensure_ascii=False)
    assert "core_change_log" in payload
    assert "capability_evidence" in payload
