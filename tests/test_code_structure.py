from neurokernel_seed.harness.code_structure import CodeStructureThresholds, inspect_code_structure


def test_inspect_code_structure_reports_long_file_and_large_function(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    body = "\n".join(f"    value_{index} = {index}" for index in range(20))
    (src / "large_module.py").write_text(
        "def large_function():\n"
        f"{body}\n"
        "    return value_19\n",
        encoding="utf-8",
    )

    report = inspect_code_structure(
        tmp_path,
        thresholds=CodeStructureThresholds(long_file_lines=10, very_long_file_lines=100, large_function_lines=10),
    )

    assert report["schema_version"] == "neurokernel-code-structure-report-v1"
    assert report["candidate_count"] == 1
    candidate = report["candidates"][0]
    assert candidate["path"] == "src/large_module.py"
    assert "long_file" in candidate["reasons"]
    assert "large_function" in candidate["reasons"]
    assert candidate["suggested_next_step"] == "extract_small_helpers_without_changing_behavior"


def test_inspect_code_structure_blocks_path_escape(tmp_path):
    try:
        inspect_code_structure(tmp_path, paths=["../outside"])
    except ValueError as exc:
        assert "escapes project root" in str(exc)
    else:
        raise AssertionError("path escape should be rejected")
