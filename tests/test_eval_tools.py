from agent.eval.harness import list_tasks, run_suite
from agent.tools.system_readonly import redact_output, run_readonly


def test_eval_tasks_exist():
    assert list_tasks("policy")
    assert list_tasks("smoke")


def test_run_policy_suite_passes():
    result = run_suite("policy")
    assert result["result"] == "PASS"
    assert result["release_blocked"] is False


def test_readonly_tool_allowlist():
    result = run_readonly("uptime")
    assert result["requires_approval"] is False
    assert result["risk_level"] == "medium"


def test_redact_output_blocks_private_key():
    assert redact_output("-----BEGIN OPENSSH PRIVATE KEY-----") == "<unsafe output blocked>"
