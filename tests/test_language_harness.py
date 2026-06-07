from pathlib import Path

import pytest

from neurokernel_seed.language.codex_harness import (
    CodexLanguageConfig,
    CodexLanguageError,
    CodexLanguageHarness,
    _capability_prompt,
    _preference_prompt,
    _to_core_prompt,
    _to_human_prompt,
    _work_route_prompt,
)
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
        harness.to_core("오늘 프로젝트 방향에 대해 같이 이야기하자")


def test_language_to_core_routes_catalog_metadata_without_codex(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))

    result = harness.to_core("cpu 코어별 사용률 확인해줘")

    assert result["intent"] == "task"
    assert result["task_spec"]["allowed_actions"] == ["get_cpu_per_core_usage"]
    assert result["requires_confirmation"] is False


def test_language_to_core_routes_memory_lookup_from_catalog_without_codex(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))

    result = harness.to_core("오렌지파이 메모리 상태 봐줘")

    assert result["intent"] == "task"
    assert result["task_spec"]["allowed_actions"] == ["get_memory_usage"]
    assert result["requires_confirmation"] is False


def test_language_to_core_routes_model_file_lookup_to_artifacts_without_codex(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))

    result = harness.to_core("모델파일위치 어디있고 모델파일이름 뭐임?")

    assert result["intent"] == "task"
    assert result["task_spec"]["allowed_actions"] == ["list_artifacts"]
    assert result["task_spec"]["context"]["params"]["path"] == "artifacts"
    assert result["requires_confirmation"] is False


def test_language_to_core_routes_symptom_diagnosis_from_catalog_without_codex(tmp_path):
    harness = CodexLanguageHarness(_missing_codex_config(tmp_path))

    result = harness.to_core("느려짐이 있고 학습 중단 의심돼. 모델 파일 이상도 진단해줘")

    assert result["intent"] == "task"
    assert result["task_spec"]["allowed_actions"] == ["diagnose_system_symptoms"]
    assert result["task_spec"]["context"]["params"]["artifact_path"] == "artifacts"
    assert result["requires_confirmation"] is False


def test_language_to_core_catalog_router_has_no_action_specific_cpu_predicate():
    source = Path("src/neurokernel_seed/language/codex_harness.py").read_text(encoding="utf-8")

    assert "_mentions_per_core_cpu_usage" not in source
    assert "_direct_catalog_intent" not in source


def test_scope_of_agency_is_injected_into_language_prompts():
    prompts = [
        _to_core_prompt("상태 봐줘", {}),
        _to_human_prompt({"kind": "work_status", "progress": []}, style="ko_short", context={}),
        _preference_prompt("앞으로 짧게 말해줘", {}),
        _capability_prompt("CPU 코어별 사용률 볼 수 있게 해줘", {}),
        _work_route_prompt("현재 작업 상태 보여줘", {}),
    ]

    for prompt in prompts:
        assert "# Scope of Agency" in prompt
        assert "Do not describe a planned external work item as active implementation." in prompt
        assert "promote it to a self-patch development work item" in prompt
        assert "Prefer human-facing words" in prompt


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
