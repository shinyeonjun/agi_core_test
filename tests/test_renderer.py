from agent.renderer.fallback_renderer import render
from agent.renderer.validator import validate_codex_output, validate_output


def test_fallback_renderer_contains_contract_tokens():
    decision = {"version": "v0.3", "selected_goal": {"id": 1, "title": "Answer user input"}, "drive_scores": {"completion": 0.2}, "policy_summary": {"risk_level": "low", "requires_approval": False}, "relevant_memories": [], "relevant_skills": [], "renderer": "fallback"}
    text = render(decision)
    assert "v0.3" in text
    assert "Core" in text


def test_validator_rejects_agi_claim():
    result = validate_output("AGI achieved", must_not_include=["AGI achieved"])
    assert result["ok"] is False


def test_codex_validator_rejects_unsafe_command():
    result = validate_codex_output("v0.3 Core event goal sudo apt install nginx", {"must_include": ["v0.3", "Core", "event", "goal"], "must_not_include": []})
    assert result["ok"] is False
