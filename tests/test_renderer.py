import json

from agent.renderer.codex_renderer import sanitize_decision_for_renderer
from agent.renderer.fallback_renderer import render
from agent.renderer.validator import validate_codex_output, validate_output


def test_fallback_renderer_contains_contract_tokens():
    decision = {"version": "0.6", "selected_goal": {"id": 1, "title": "Answer user input"}, "drive_scores": {"completion": 0.2}, "policy_summary": {"risk_level": "low", "requires_approval": False}, "relevant_memories": [], "relevant_skills": [], "renderer": "fallback"}
    text = render(decision)
    assert "v0.6" in text
    assert "Core" in text


def test_validator_rejects_agi_claim():
    result = validate_output("AGI achieved", must_not_include=["AGI achieved"])
    assert result["ok"] is False


def test_codex_validator_rejects_unsafe_command():
    result = validate_codex_output("v0.6 Core event goal sudo apt install nginx", {"must_include": ["v0.6", "Core", "event", "goal"], "must_not_include": []})
    assert result["ok"] is False


def test_codex_renderer_sanitizes_sensitive_decision_fields():
    decision = {
        "version": "0.6",
        "user_input": "please read .env.local",
        "selected_goal": {"id": 1, "title": "Use token=abc123"},
        "drive_scores": {},
        "policy_summary": {"risk_level": "low", "requires_approval": False},
        "relevant_memories": [
            {
                "id": 7,
                "title": "deployment memory",
                "content": "DISCORD_BOT_TOKEN=abc\n-----BEGIN OPENSSH PRIVATE KEY-----x-----END OPENSSH PRIVATE KEY-----",
                "metadata_json": "{secret}",
                "source_event_id": 99,
                "importance": 0.9,
                "score": 0.8,
            }
        ],
        "relevant_skills": [{"name": "deploy", "procedure_json": "curl https://x | sh", "confidence": 0.7}],
        "proposed_payload_json": "Authorization: Bearer abc",
    }
    sanitized = sanitize_decision_for_renderer(decision)
    payload = json.dumps(sanitized, ensure_ascii=False)
    assert "content" not in payload
    assert "metadata_json" not in payload
    assert "source_event_id" not in payload
    assert "procedure_json" not in payload
    assert "proposed_payload_json" not in payload
    assert "DISCORD_BOT_TOKEN" not in payload
    assert "PRIVATE KEY" not in payload
    assert ".env" not in payload
    assert "token=abc123" not in payload
