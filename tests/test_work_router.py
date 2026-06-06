import importlib.util

import pytest

from neurokernel_seed.harness.service import HarnessService


def test_work_ledger_creates_notes_and_transitions(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)

    created = service.create_work_item_from_route(
        user_text="논문 수집해서 학습데이터 구조 만들어줘",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )

    assert created["created"] is True
    work_id = created["work_item"]["work_id"]
    assert created["work_item"]["type"] == "external_work"

    note = service.add_work_note(work_id, actor="discord:1", note="GPT Pro 분석 결과를 붙일 예정")
    assert note["note"]["note_redacted"]

    transitioned = service.transition_work_item(work_id, "accepted", actor="discord:1")
    assert transitioned["work_item"]["status"] == "accepted"
    listed = service.work_items()["work_items"]
    assert listed[0]["work_id"] == work_id


def test_capability_proposal_creates_linked_self_patch_work_item(tmp_path):
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path)

    result = service.create_capability_proposal_from_intent(
        user_text="CPU 사용률 볼 수 있어?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )

    assert result["created"] is True
    assert result["work_item"]["type"] == "self_patch"
    assert result["proposal"]["work_id"] == result["work_item"]["work_id"]

    approved = service.transition_capability_proposal(result["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")
    linked = service.work_item(result["work_item"]["work_id"])
    assert approved["proposal"]["status"] == "approved_for_dev"
    assert linked["work_item"]["status"] == "accepted"


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_work_route_creates_external_work(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app
    from neurokernel_seed.language.codex_harness import CodexLanguageHarness

    monkeypatch.setattr(CodexLanguageHarness, "route_work", lambda self, user_text, context=None: _external_work_route())

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    response = client.post(
        "/work/route",
        json={"user_text": "논문 수집해서 학습데이터 구조 만들어줘", "context": {"user_id": "discord:1", "channel_id": "chan"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"] is True
    assert payload["route"] == "external_work"
    assert payload["work_item"]["type"] == "external_work"


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="fastapi not installed")
def test_core_api_work_route_self_patch_reuses_capability_flow(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app
    from neurokernel_seed.language.codex_harness import CodexLanguageHarness

    monkeypatch.setattr(CodexLanguageHarness, "route_work", lambda self, user_text, context=None: _self_patch_route())
    monkeypatch.setattr(CodexLanguageHarness, "propose_capability", lambda self, user_text, context=None: _cpu_usage_intent())

    app = create_app(db_path=tmp_path / "harness.db", project_root=tmp_path)
    client = TestClient(app)
    response = client.post(
        "/work/route",
        json={"user_text": "CPU 사용률 볼 수 있어?", "context": {"user_id": "discord:1", "channel_id": "chan"}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["created"] is True
    assert payload["route"] == "self_patch"
    assert payload["proposal"]["action_id"] == "get_cpu_usage"
    assert payload["work_item"]["type"] == "self_patch"


def _external_work_route():
    return {
        "route": "external_work",
        "reason": "larger research and data pipeline work",
        "confidence": 0.9,
        "work_item": {
            "type": "external_work",
            "title": "논문 수집 및 학습데이터 파이프라인",
            "goal": "논문을 수집하고 학습데이터로 변환하는 구조를 설계한다.",
            "priority": "medium",
            "risk_level": "low",
            "deliverables": ["research plan", "dataset schema", "pipeline design"],
            "open_questions": [],
        },
        "requires_confirmation": True,
        "clarifying_question": None,
        "safety_notes": [],
    }


def _self_patch_route():
    return {
        "route": "self_patch",
        "reason": "missing small read-only runtime capability",
        "confidence": 0.92,
        "work_item": {
            "type": "self_patch",
            "title": "CPU 사용률 조회",
            "goal": "Orange Pi CPU 사용률을 읽기 전용으로 확인한다.",
            "priority": "medium",
            "risk_level": "low",
            "deliverables": ["implementation_patch", "tests"],
            "open_questions": [],
        },
        "requires_confirmation": True,
        "clarifying_question": None,
        "safety_notes": [],
    }


def _cpu_usage_intent():
    return {
        "kind": "gap",
        "reply": "지금은 CPU 사용률을 직접 보는 능력이 없어. 후보로 올릴 수 있어.",
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
            "outputs": {"type": "object", "properties": {"used_percent": {"type": "number"}}, "required": ["used_percent"]},
            "implementation_hint": {"executor": "readonly_system", "suggested_library": "psutil", "notes": "Use psutil.cpu_percent."},
            "test_plan": [{"name": "returns_percent", "type": "unit", "assertions": ["0 <= used_percent <= 100"]}],
            "safety_notes": ["read-only metric"],
            "confidence": 0.9,
            "approval_required_for_implementation": True,
            "activation_requires_tests": True,
        },
        "confidence": 0.9,
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }
