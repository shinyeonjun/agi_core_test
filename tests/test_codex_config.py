from agent.codex_config import codex_exec_args, codex_exec_config


def test_codex_config_uses_component_reasoning(monkeypatch):
    monkeypatch.setenv("AGENT_CODEX_LANGUAGE_REASONING", "minimal")

    config = codex_exec_config("LANGUAGE", default_reasoning="low", default_timeout=20)

    assert config.reasoning_effort == "minimal"
    assert "-c" in codex_exec_args(config)
    assert "model_reasoning_effort=minimal" in codex_exec_args(config)


def test_codex_config_ignores_invalid_reasoning(monkeypatch):
    monkeypatch.setenv("AGENT_CODEX_RENDERER_REASONING", "turbo")

    config = codex_exec_config("RENDERER", default_reasoning="low", default_timeout=20)

    assert config.reasoning_effort == "low"


def test_codex_config_component_overrides_global(monkeypatch):
    monkeypatch.setenv("AGENT_CODEX_MODEL", "global-model")
    monkeypatch.setenv("AGENT_CODEX_RENDERER_MODEL", "renderer-model")
    monkeypatch.setenv("AGENT_CODEX_TIMEOUT", "99")
    monkeypatch.setenv("AGENT_CODEX_RENDERER_TIMEOUT", "17")

    config = codex_exec_config("RENDERER", default_reasoning="low", default_timeout=20)

    assert config.model == "renderer-model"
    assert config.timeout_seconds == 17
    assert codex_exec_args(config)[:3] == ["codex", "exec", "--model"]
    assert "renderer-model" in codex_exec_args(config)
