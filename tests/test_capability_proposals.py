import importlib.util

import pytest

from neurokernel_seed.harness.service import HarnessService


def test_missing_action_creates_capability_proposal(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)

    result = service.create_capability_proposal_from_intent(
        user_text="CPU 사용률 볼 수 있어?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )

    assert result["created"] is True
    assert result["kind"] == "gap"
    assert result["proposal"]["action_id"] == "get_cpu_usage"
    assert result["proposal"]["status"] == "proposed"

    approved = service.transition_capability_proposal(result["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")
    assert approved["proposal"]["status"] == "approved_for_dev"


def test_existing_action_does_not_create_capability_proposal(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)
    intent = _cpu_usage_intent()
    intent["proposal"] = {**intent["proposal"], "action_id": "get_memory_usage"}

    result = service.create_capability_proposal_from_intent(
        user_text="메모리 볼 수 있어?",
        capability_intent=intent,
    )

    assert result["created"] is False
    assert result["kind"] == "existing_action"
    assert service.capability_proposals()["proposals"] == []


def test_duplicate_capability_proposal_reuses_existing_open_proposal(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)

    first = service.create_capability_proposal_from_intent(user_text="CPU 사용률 봐줘", capability_intent=_cpu_usage_intent())
    second = service.create_capability_proposal_from_intent(user_text="CPU percent 확인 가능?", capability_intent=_cpu_usage_intent())

    assert first["created"] is True
    assert second["created"] is False
    assert second["kind"] == "duplicate"
    assert second["proposal"]["proposal_id"] == first["proposal"]["proposal_id"]
    assert len(service.capability_proposals()["proposals"]) == 1


def test_capability_proposal_cannot_be_activated_in_v1(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)
    result = service.create_capability_proposal_from_intent(user_text="CPU 사용률 봐줘", capability_intent=_cpu_usage_intent())

    with pytest.raises(ValueError):
        service.transition_capability_proposal(result["proposal"]["proposal_id"], "active", actor="test")


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_capability_proposal_from_request(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app
    from neurokernel_seed.language.codex_harness import CodexLanguageHarness

    monkeypatch.setattr(CodexLanguageHarness, "propose_capability", lambda self, user_text, context=None: _cpu_usage_intent())

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    response = client.post(
        "/capability-proposals/from-request",
        json={"user_text": "CPU 사용률 볼 수 있어?", "context": {"user_id": "discord:1", "channel_id": "chan"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"] is True
    assert payload["proposal"]["action_id"] == "get_cpu_usage"


def _cpu_usage_intent():
    return {
        "kind": "gap",
        "reply": "지금은 CPU 사용률을 직접 보는 능력이 없어. 새 능력 후보로 올릴 수 있어.",
        "gap": {
            "gap_type": "missing_action",
            "requested_capability": "현재 CPU 사용률 확인",
            "normalized_request": "orangepi5 현재 CPU 사용률을 조회한다",
            "matched_existing_actions": [{"action_id": "get_cpu_temp", "match_score": 0.52, "reason": "CPU 온도만 확인한다"}],
            "confidence": 0.9,
        },
        "proposal": {
            "action_id": "get_cpu_usage",
            "capability_name": "CPU 사용률 확인",
            "purpose": "Orange Pi 5의 현재 CPU 전체 및 코어별 사용률을 조회한다.",
            "target": "orangepi5",
            "risk_level": "low",
            "side_effect": False,
            "requires_approval": False,
            "inputs": {"type": "object", "properties": {}, "required": []},
            "outputs": {
                "type": "object",
                "properties": {
                    "used_percent": {"type": "number"},
                    "per_core_percent": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["used_percent"],
            },
            "implementation_hint": {
                "executor": "readonly_system",
                "suggested_library": "psutil",
                "notes": "Use psutil.cpu_percent(interval=0.2, percpu=True). No shell or sudo.",
            },
            "test_plan": [
                {"name": "returns_percent", "type": "unit", "assertions": ["0 <= used_percent <= 100"]},
                {"name": "read_only", "type": "safety", "assertions": ["no file writes", "no sudo"]},
            ],
            "safety_notes": ["read-only metric", "no secrets", "no network"],
            "confidence": 0.9,
            "approval_required_for_implementation": True,
            "activation_requires_tests": True,
        },
        "confidence": 0.9,
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }
