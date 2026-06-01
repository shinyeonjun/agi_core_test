import json
from types import SimpleNamespace

from agent.renderer.codex_renderer import render_with_codex, sanitize_decision_for_renderer
from agent.renderer.engine import render_response
from agent.renderer.fallback_renderer import render
from agent.renderer.validator import validate_codex_output, validate_output


def test_fallback_renderer_contains_contract_tokens():
    decision = {"version": "0.8", "selected_goal": {"id": 1, "title": "Answer user input"}, "drive_scores": {"completion": 0.2}, "policy_summary": {"risk_level": "low", "requires_approval": False}, "relevant_memories": [], "relevant_skills": [], "renderer": "fallback"}
    text = render(decision)
    assert "v0.8" in text
    assert "Core" in text


def test_validator_rejects_agi_claim():
    result = validate_output("AGI achieved", must_not_include=["AGI achieved"])
    assert result["ok"] is False


def test_codex_validator_rejects_unsafe_command():
    result = validate_codex_output("v0.6 Core event goal sudo apt install nginx", {"must_include": ["v0.8", "Core", "event", "goal"], "must_not_include": []})
    assert result["ok"] is False


def test_codex_renderer_sanitizes_sensitive_decision_fields():
    decision = {
        "version": "0.8",
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


def test_render_response_uses_codex_strategy(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "codex")
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))

    def fake_render(decision):
        return f"codex reply for {decision['user_input']}"

    monkeypatch.setattr("agent.renderer.engine.render_with_codex", fake_render)

    assert render_response({"user_input": "hello"}) == "codex reply for hello"


def test_codex_renderer_uses_output_last_message(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path))

    def fake_run(args, **kwargs):
        assert "--output-last-message" in args
        assert "--ephemeral" in args
        assert "--sandbox" in args
        assert "model_reasoning_effort=low" in args
        assert kwargs["stdin"] is not None
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("응, 자연어 답변으로 처리했어.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.renderer.codex_renderer.subprocess.run", fake_run)

    text = render_with_codex({
        "version": "0.8",
        "user_input": "ㅎㅇ",
        "selected_goal": {"id": 1, "title": "Answer user input"},
        "policy_summary": {"risk_level": "low", "requires_approval": False},
        "must_include": [],
        "must_not_include": ["AGI achieved"],
    })

    assert "자연어 답변" in text


def test_codex_renderer_respects_component_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_CODEX_RENDERER_MODEL", "renderer-model")
    monkeypatch.setenv("AGENT_CODEX_RENDERER_REASONING", "minimal")
    monkeypatch.setenv("AGENT_CODEX_RENDERER_TIMEOUT", "9")

    def fake_run(args, **kwargs):
        assert args[:4] == ["codex", "exec", "--model", "renderer-model"]
        assert "model_reasoning_effort=minimal" in args
        assert kwargs["timeout"] == 9
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("환경별 빠른 렌더링.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.renderer.codex_renderer.subprocess.run", fake_run)

    text = render_with_codex({
        "version": "0.8",
        "user_input": "ㅎㅇ",
        "selected_goal": {"id": 1, "title": "Answer user input"},
        "policy_summary": {"risk_level": "low", "requires_approval": False},
        "must_include": [],
        "must_not_include": ["AGI achieved"],
    })

    assert "빠른 렌더링" in text
