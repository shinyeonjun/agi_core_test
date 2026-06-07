import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_health(tmp_path):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_language_to_human_fails_without_language_config(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUROKERNEL_CODEX_BIN", raising=False)
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/language/to-human",
        json={
            "core_result": {
                "execution_result": {
                    "action_id": "get_memory_usage",
                    "success": True,
                    "result": {"total_bytes": 1024**3, "used_bytes": 256 * 1024**2, "available_bytes": 768 * 1024**2, "used_percent": 25.0},
                }
            }
        },
    )
    assert response.status_code == 500


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_memory_context_includes_preferences_and_recent_messages(tmp_path):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    client.post("/memory/preferences", json={"user_id": "discord:1", "key": "response_length", "value": "short"})
    client.post("/memory/messages", json={"user_id": "discord:1", "channel_id": "chan", "role": "user", "content": "?꾧퉴 紐⑤뜽 遊먯쨾"})

    response = client.get("/memory/context", params={"user_id": "discord:1", "channel_id": "chan"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_preferences"]["response_length"] == "short"
    assert payload["recent_messages"][0]["content"] == "?꾧퉴 紐⑤뜽 遊먯쨾"


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_records_interaction_outcome(tmp_path):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    created = client.post(
        "/tasks",
        json={
            "goal": "메모리 확인",
            "target": "orangepi5",
            "allowed_actions": ["get_memory_usage"],
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        },
    ).json()
    task_id = created["task"]["task_id"]
    result = client.post(f"/tasks/{task_id}/run", json={}).json()

    response = client.post(
        "/memory/interaction-outcomes",
        json={
            "source": "discord",
            "user_id": "discord:1",
            "channel_id": "chan",
            "task_id": task_id,
            "request_text": "메모리 알려줘",
            "response_text": "메모리 확인 완료",
            "task": {"allowed_actions": ["get_memory_usage"]},
            "core_result": result,
        },
    )

    assert response.status_code == 200
    outcome = response.json()["interaction_outcome"]
    assert outcome["required_outputs_json"] == ["memory_usage"]
    assert outcome["answered_outputs_json"] == ["memory_usage"]
    assert outcome["answer_quality"] == "complete"


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_activation_error_returns_conflict_detail(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app
    from neurokernel_seed.harness.activation import ActivationError
    from neurokernel_seed.harness.service import HarnessService

    def fail_activation(self, work_id, *, actor="api"):
        raise ActivationError("live repository has uncommitted changes; activation requires a clean tree")

    monkeypatch.setattr(HarnessService, "activate_work_item", fail_activation)
    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/work-items/work_1/activate", json={"actor": "test"})

    assert response.status_code == 409
    payload = response.json()
    assert payload["detail"]["error_type"] == "activation_failed"
    assert "clean tree" in payload["detail"]["message"]
