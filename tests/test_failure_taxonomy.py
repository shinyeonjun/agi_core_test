from agent.core.failure import classify_failure, failure_report, recovery_hint


def test_failure_taxonomy_labels_common_outcomes():
    assert classify_failure({"status": "completed", "returncode": 0}) == "success"
    assert classify_failure({"reason": "profile_not_full_device_lab"}) == "profile_block"
    assert classify_failure({"stderr": "ModuleNotFoundError: No module named agent"}) == "environment_issue"
    assert classify_failure({"stderr": "pytest failed"}) == "verification_failed"


def test_failure_report_includes_recovery_hint():
    report = failure_report({"returncode": 124, "stderr": "timeout"})

    assert report["category"] == "timeout"
    assert report["known"] is True
    assert recovery_hint(report["category"])
