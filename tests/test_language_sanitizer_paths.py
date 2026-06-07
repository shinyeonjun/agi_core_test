import pytest

from neurokernel_seed.language.sanitizer import HumanReplySanitizerError, clean_human_reply


def test_sanitizer_rejects_internal_schema_and_action_names():
    for text in (
        "execution_result value",
        "action_id value",
        "list_artifacts result",
    ):
        with pytest.raises(HumanReplySanitizerError):
            clean_human_reply(text)


def test_sanitizer_allows_observed_artifact_paths():
    model_path = "모델 파일은 artifacts/current_runtime_action_model.pt 에 있어."
    benchmark_path = "벤치 결과는 artifacts/model_benchmarks/runtime_action/report_2026.json 에 있어."

    assert clean_human_reply(model_path) == model_path
    assert clean_human_reply(benchmark_path) == benchmark_path
