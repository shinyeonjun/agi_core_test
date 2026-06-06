import pytest

from neurokernel_seed.harness.task_spec import TaskSpecError, task_spec_from_dict


def test_task_spec_accepts_readonly_task():
    spec = task_spec_from_dict(
        {
            "goal": "check status",
            "target": "orangepi5",
            "allowed_actions": ["get_uptime", "get_memory_usage"],
            "blocked_actions": [],
            "success_criteria": ["status returned"],
            "risk_level": "low",
            "requires_approval": False,
        }
    )
    assert spec.goal == "check status"
    assert spec.allowed_actions == ("get_uptime", "get_memory_usage")


def test_task_spec_rejects_unknown_field():
    with pytest.raises(TaskSpecError):
        task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": [], "surprise": True})


def test_task_spec_rejects_unknown_action():
    with pytest.raises(TaskSpecError):
        task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["raw_shell"]})


def test_task_spec_rejects_blocked_allowed_overlap():
    with pytest.raises(TaskSpecError):
        task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["get_uptime"], "blocked_actions": ["get_uptime"]})


def test_task_spec_requires_approval_for_high_risk():
    with pytest.raises(TaskSpecError):
        task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["reboot"], "risk_level": "high", "requires_approval": False})

