from agent.core.database import connect, init_db
from agent.eval.harness import list_tasks, run_suite
from agent.tools.system_readonly import redact_output, run_readonly


def test_eval_tasks_exist():
    assert list_tasks("policy")
    assert list_tasks("smoke")


def test_run_policy_suite_passes():
    result = run_suite("policy")
    assert result["result"] == "PASS"
    assert result["isolated"] is True
    assert result["release_blocked"] is False


def test_readonly_tool_allowlist():
    result = run_readonly("uptime")
    assert result["requires_approval"] is False
    assert result["risk_level"] == "medium"


def test_redact_output_blocks_private_key():
    assert redact_output("-----BEGIN OPENSSH PRIVATE KEY-----") == "<unsafe output blocked>"


def test_expanded_eval_suites_exist():
    assert list_tasks("memory")
    assert list_tasks("discord")
    assert list_tasks("reflection")
    assert list_tasks("tool")
    assert list_tasks("workspace")
    assert list_tasks("action")


def test_readonly_tool_missing_binary_is_recorded(monkeypatch):
    from agent.tools import system_readonly
    original = system_readonly.READ_ONLY_COMMANDS["uptime"]
    system_readonly.READ_ONLY_COMMANDS["uptime"] = original.__class__("uptime", ["/definitely/missing/command"])
    try:
        result = system_readonly.run_readonly("uptime")
    finally:
        system_readonly.READ_ONLY_COMMANDS["uptime"] = original
    assert result["returncode"] == 127
    assert result["error"] == "command_not_found"


def test_eval_memory_suite_uses_isolated_database():
    init_db()
    with connect() as conn:
        before = conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"]
    result = run_suite("memory")
    with connect() as conn:
        after = conn.execute("SELECT COUNT(*) AS count FROM memories").fetchone()["count"]
    assert result["result"] == "PASS"
    assert result["isolated"] is True
    assert after == before
