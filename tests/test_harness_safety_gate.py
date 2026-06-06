from neurokernel_seed.harness.action_catalog import ActionDefinition
from neurokernel_seed.harness.safety_gate import check_action_safety
from neurokernel_seed.harness.task_spec import task_spec_from_dict


def test_safety_gate_allows_readonly_allowed_action():
    spec = task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["get_uptime"]})
    decision = check_action_safety(spec, "get_uptime")
    assert decision.decision == "allow"


def test_safety_gate_denies_default_not_allowed_action():
    spec = task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["get_uptime"]})
    decision = check_action_safety(spec, "get_memory_usage")
    assert decision.decision == "deny"


def test_safety_gate_denies_forbidden_action_even_when_allowed():
    spec = task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["delete_file"], "risk_level": "forbidden", "requires_approval": True})
    decision = check_action_safety(spec, "delete_file", approved=True)
    assert decision.decision == "deny"


def test_safety_gate_requires_approval_for_write_action():
    spec = task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["write_file"], "risk_level": "medium", "requires_approval": True})
    decision = check_action_safety(spec, "write_file")
    assert decision.decision == "requires_approval"


def test_safety_gate_denies_inactive_action_even_when_allowed():
    catalog = {
        "get_waiting_probe": ActionDefinition(
            "get_waiting_probe",
            "대기 중 probe",
            "low",
            False,
            False,
            "readonly_command",
            allowed_targets=("orangepi5",),
            status="activation_candidate",
            description="아직 장착 승인 전인 probe",
            test_plan=({"name": "not_runnable", "assertions": ["must not run before activation"]},),
        )
    }
    spec = task_spec_from_dict({"goal": "x", "target": "orangepi5", "allowed_actions": ["get_waiting_probe"]}, catalog=catalog)

    decision = check_action_safety(spec, "get_waiting_probe", catalog=catalog)

    assert decision.decision == "deny"
    assert decision.policy_hits == ("deny_inactive_action",)
