from pathlib import Path

import pytest

from neurokernel_seed.language.codex_harness import CodexLanguageConfig, CodexLanguageError, CodexLanguageHarness
from neurokernel_seed.language.contracts import validate_preference_intent
from neurokernel_seed.language.sanitizer import HumanReplySanitizerError, clean_human_reply, contains_internal_language


def _missing_codex_config(tmp_path: Path) -> CodexLanguageConfig:
    return CodexLanguageConfig(
        codex_bin="definitely_missing_codex_binary",
        workspace=tmp_path / "workspace",
        schema_dir=Path("schemas"),
        timeout_seconds=1,
    )


def test_language_to_core_fails_when_codex_is_unavailable(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))
    with pytest.raises(CodexLanguageError, match="codex binary not found"):
        harness.to_core("오렌지파이 메모리 상태 봐줘")


def test_language_to_human_fails_when_codex_is_unavailable(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))
    with pytest.raises(CodexLanguageError, match="codex binary not found"):
        harness.to_human({"status": "completed"})


def test_preference_extraction_fails_when_codex_is_unavailable(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))
    with pytest.raises(CodexLanguageError, match="codex binary not found"):
        harness.extract_preferences("앞으로 짧게 말해줘")


def test_sanitizer_rejects_internal_language_without_fallback():
    assert contains_internal_language("execution_result 값")
    with pytest.raises(HumanReplySanitizerError):
        clean_human_reply("execution_result 값")
    with pytest.raises(HumanReplySanitizerError):
        clean_human_reply("trace에 기록했어")
    with pytest.raises(HumanReplySanitizerError):
        clean_human_reply("")


def test_sanitizer_accepts_clean_human_reply():
    assert clean_human_reply("메모리는 여유 있어.") == "메모리는 여유 있어."


def test_preference_intent_contract_accepts_safe_candidate():
    result = validate_preference_intent(
        {
            "kind": "preference_update",
            "reply": "좋아. 앞으로 짧게 말할게.",
            "candidates": [
                {
                    "key": "response_length",
                    "value": "short",
                    "confidence": 0.95,
                    "evidence": "앞으로 답변 짧게 해줘",
                    "scope": "global",
                    "source": "explicit_user_request",
                }
            ],
            "confidence": 0.95,
            "requires_confirmation": False,
            "clarifying_question": None,
            "safety_notes": [],
        }
    )
    assert result["candidates"][0]["key"] == "response_length"
