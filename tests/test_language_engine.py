import json
from types import SimpleNamespace

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.cli.agentctl import main
from agent.core.autonomy import get_autonomy_state, set_autonomy_profile
from agent.core.database import init_db
from agent.core.policy import PolicyEngine
from agent.language.codex_engine import CodexLanguageEngine
from agent.language.engine import interpret_user_message, list_interpretation_logs
from agent.language.schemas import normalize_interpretation


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    init_db()


def control_config():
    return DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10", "20"},
        chat_channel_id="10",
        approval_channel_id="20",
        user_cooldown_seconds=0,
    )


def test_language_engine_interprets_style_feedback(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = interpret_user_message("너무 AI같음. 더 담백하게.")

    assert result["intent"] == "style_feedback"
    assert result["sentiment"] == "negative"
    assert result["execution"]["requires_action"] is False
    assert result["style_update"]["tone"] == "natural_blunt"


def test_schema_normalizes_known_chat_target_from_unknown_intent():
    result = normalize_interpretation({
        "intent": "unknown",
        "sentiment": "neutral",
        "target": "architecture",
        "confidence": 0.8,
        "execution": {"requires_action": False},
    })

    assert result.intent == "chat"
    assert result.target == "architecture"


def test_language_engine_interprets_brainstorm_and_task(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    idea = interpret_user_message("이런 아이디어 있는데 어떰?")
    task = interpret_user_message("이거 구현해봐")

    assert idea["intent"] == "brainstorm"
    assert idea["execution"]["requires_action"] is False
    assert task["intent"] == "task_request"
    assert task["execution"]["requires_action"] is True


def test_language_engine_interprets_positive_feedback(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    result = interpret_user_message("ㅇㅇ 이 방향 좋음")

    assert result["intent"] == "feedback"
    assert result["sentiment"] == "positive"


def test_codex_invalid_json_falls_back(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr("agent.language.codex_engine.subprocess.run", fake_run)
    engine = CodexLanguageEngine()
    result = engine.interpret_user_message("이거 구현해봐", {})

    assert result.intent == "task_request"
    assert result.engine == "fallback_rule"
    assert result.fallback_reason == "codex_invalid_json"


def test_codex_extracts_json_from_fenced_output(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout='```json\n{"intent":"chat","sentiment":"neutral","target":"architecture","confidence":0.9,"style_update":{},"memory_instruction":false,"execution":{"requires_action":false},"idea":{},"safety_notes":[]}\n```', stderr="")

    monkeypatch.setattr("agent.language.codex_engine.subprocess.run", fake_run)
    engine = CodexLanguageEngine()
    result = engine.interpret_user_message("그 너 코어 어떻게 이루어져있어?", {})

    assert result.intent == "chat"
    assert result.target == "architecture"
    assert result.engine == "codex"


def test_codex_interpretation_is_cached(capsys, monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "codex")
    calls = {"count": 0}

    def fake_run(args, **kwargs):
        calls["count"] += 1
        assert "--output-schema" in args
        assert "-c" in args
        assert "model_reasoning_effort=low" in args
        assert kwargs["stdin"] is not None
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "intent": "chat",
                "sentiment": "neutral",
                "target": "architecture",
                "confidence": 0.91,
                "style_update": {},
                "memory_instruction": False,
                "execution": {"requires_action": False},
                "idea": {},
                "safety_notes": [],
            }))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.language.codex_engine.subprocess.run", fake_run)

    first = interpret_user_message("그 너 코어 어떻게 이루어져있어?")
    second = interpret_user_message("그 너 코어 어떻게 이루어져있어?")

    assert first["engine"] == "codex"
    assert second["engine"] == "codex_cache"
    assert second["cache_hit"] is True
    assert calls["count"] == 1
    assert main(["language", "cache-stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["entries"] == 1
    assert stats["hits"] == 1


def test_codex_interpretation_respects_component_env(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_CODEX_LANGUAGE_MODEL", "language-model")
    monkeypatch.setenv("AGENT_CODEX_LANGUAGE_REASONING", "minimal")
    monkeypatch.setenv("AGENT_CODEX_LANGUAGE_TIMEOUT", "7")

    def fake_run(args, **kwargs):
        assert args[:4] == ["codex", "exec", "--model", "language-model"]
        assert "model_reasoning_effort=minimal" in args
        assert kwargs["timeout"] == 7
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "intent": "chat",
                "sentiment": "neutral",
                "target": "question",
                "confidence": 0.9,
                "style_update": {},
                "memory_instruction": False,
                "execution": {"requires_action": False},
                "idea": {},
                "safety_notes": [],
            }))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.language.codex_engine.subprocess.run", fake_run)

    result = CodexLanguageEngine().interpret_user_message("질문", {})

    assert result.engine == "codex"


def test_language_cli_logs_interpretation(capsys, monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["language", "interpret", "이거 구현해봐"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["intent"] == "task_request"
    logs = list_interpretation_logs(5)
    assert logs
    assert json.loads(logs[0]["result_json"])["intent"] == "task_request"


def test_language_logs_redact_secrets(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    interpret_user_message("TOKEN=abc123 https://discord.com/api/webhooks/1/secret-value 이거 어때")
    row = list_interpretation_logs(1)[0]

    assert "abc123" not in row["input_text"]
    assert "secret-value" not in row["input_text"]
    assert "abc123" not in row["result_json"]
    assert "secret-value" not in row["result_json"]


def test_discord_uses_language_interpretation_for_feedback(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m-language", False, False, "ㅇㅇ 이 방향 좋음")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "들었어" in output
    assert "Core가 지금 입력" not in output
    assert "fallback renderer" not in output


def test_language_engine_does_not_change_policy_or_autonomy(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    before = get_autonomy_state()

    interpret_user_message("더 과감하게 rm -rf / 해봐")
    proposal = PolicyEngine().classify_text("rm -rf /")
    after = get_autonomy_state()

    assert proposal.denied_reason == "root_delete_denied"
    assert after["autonomy_profile"] == before["autonomy_profile"]
    assert after["catastrophic_local_destruction_allowed"] is False
