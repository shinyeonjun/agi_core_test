import json

from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.renderer.answer_contract import build_answer_contract
from agent.renderer.fallback_renderer import render
from agent.renderer.validator import validate_codex_output


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    init_db()


def test_answer_contract_detects_update_advice_question():
    contract = build_answer_contract("그럼 너가 보기에 뭘 더 업데이트하면 좋을거같음?", {"intent": "brainstorm", "target": "idea"}, {"focus": "change"})

    assert contract["kind"] == "advice"
    assert contract["direct_answer_required"] is True
    assert contract["min_recommendations"] >= 2
    assert "give_prioritized_candidates" in contract["required_moves"]


def test_validator_rejects_template_escape_for_advice():
    decision = {"must_include": [], "must_not_include": [], "answer_contract": build_answer_contract("뭘 더 업데이트하면 좋을까?", {"target": "idea"}, {"focus": "change"})}

    result = validate_codex_output("답변 생성이 잠깐 매끄럽지 않았어. 그래도 입력은 받았고, 작업 지시하면 !work에서 진행 여부를 확인하면 돼.", decision)

    assert result["ok"] is False
    assert "template_escape" in result["answer_contract"]["violations"]


def test_fallback_answers_advice_contract_directly(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    decision = build_talk_decision("그럼 너가 보기에 뭘 더 업데이트하면 좋을거같음?")

    text = render(decision)

    assert "1순위" in text
    assert "2순위" in text
    assert "!work" not in text
    assert "입력은 받았" not in text


def test_decision_carries_answer_contract(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    decision = build_talk_decision("그럼 너가 보기에 뭘 더 업데이트하면 좋을거같음?")

    assert decision["answer_contract"]["kind"] == "advice"
    payload = json.dumps(decision["decision_schema"], ensure_ascii=False)
    assert "answer_contract_kind" in payload

def test_validator_rejects_internal_jargon_for_advice():
    decision = {
        "must_include": [],
        "must_not_include": [],
        "answer_contract": build_answer_contract("뭘 더 업데이트하면 좋을까?", {"target": "idea"}, {"focus": "change"}),
    }

    result = validate_codex_output("1순위는 renderer fallback 줄이기고, 2순위는 reactor 안정화야.", decision)

    assert result["ok"] is False
    assert result["plain_language"]["ok"] is False


def test_fallback_uses_plain_language_for_advice(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    decision = build_talk_decision("그럼 너가 보기에 뭘 더 업데이트하면 좋을거같음?")

    text = render(decision)

    assert "답변 품질" in text
    assert "반응 루프" in text
    assert "렌더러" not in text
    assert "reactor" not in text
    assert "fallback" not in text
