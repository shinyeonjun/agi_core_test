import json
import sys

from agent.cli.agentctl import main
from agent.core.autonomy import arm_catastrophic_destruction, get_autonomy_state, set_autonomy_profile
from agent.core.database import init_db
from agent.tools.action_log import get_action_run, list_action_runs
from agent.tools.full_device import redact_action_output, run_action


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    init_db()


def test_autonomy_profile_defaults_and_set(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    assert get_autonomy_state()["autonomy_profile"] == "safe"
    state = set_autonomy_profile("full_device_lab")
    assert state["autonomy_profile"] == "full_device_lab"
    assert state["full_device_lab_enabled"] is True
    assert state["os_mutation_allowed"] is True
    assert state["external_network_actions_allowed"] is False
    assert state["self_modification_allowed"] == "branch_or_proposal"
    assert state["catastrophic_local_destruction_allowed"] is False


def test_action_run_safe_profile_blocks_execution(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")
    result = run_action("printf hello", cwd=str(tmp_path))
    assert result["executed"] is False
    assert result["reason"] == "profile_not_full_device_lab"
    row = get_action_run(result["id"])
    assert row is not None
    assert row["status"] == "blocked"


def test_action_run_full_device_records_snapshots(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    result = run_action(f'"{sys.executable}" -c "print(\'hello\', end=\'\')"', cwd=str(tmp_path), use_shell=True)
    assert result["executed"] is True
    assert result["returncode"] == 0
    assert result["stdout"] == "hello"
    row = get_action_run(result["id"])
    assert row is not None
    assert row["profile"] == "full_device_lab"
    assert row["status"] == "completed"
    assert json.loads(row["before_snapshot_json"])
    assert json.loads(row["after_snapshot_json"])


def test_action_run_timeout_records_timeout(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    result = run_action(f'"{sys.executable}" -c "import time; time.sleep(2)"', cwd=str(tmp_path), timeout_seconds=1, use_shell=True)
    assert result["status"] == "timeout"
    assert result["returncode"] == 124
    assert get_action_run(result["id"])["status"] == "timeout"


def test_action_run_still_blocks_secret_access_in_full_device(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    result = run_action("cat ~/.ssh/id_rsa", cwd=str(tmp_path))
    assert result["executed"] is False
    assert result["reason"] == "ssh_key_access_denied"


def test_redact_action_output():
    assert "abc123" not in redact_action_output("TOKEN=abc123")
    assert "PRIVATE KEY" not in redact_action_output("-----BEGIN OPENSSH PRIVATE KEY-----x-----END OPENSSH PRIVATE KEY-----")


def test_action_cli_history(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)
    assert main(["autonomy", "set", "full_device_lab"]) == 0
    assert main(["action", "run", f'"{sys.executable}" -c "print(\'cli\', end=\'\')"', "--cwd", str(tmp_path), "--shell"]) == 0
    run_output = capsys.readouterr().out
    assert "cli" in run_output
    assert main(["action", "history", "--limit", "1"]) == 0
    history_output = capsys.readouterr().out
    assert "full_device_lab" in history_output
    assert list_action_runs(1)


def test_catastrophic_destruction_arm_allows_root_delete(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    state = arm_catastrophic_destruction(ttl_seconds=30)
    assert state["catastrophic_local_destruction_allowed"] is True
    from agent.core.policy import PolicyEngine

    proposal = PolicyEngine(profile="full_device_lab").classify_text("rm -rf /")
    assert proposal.denied_reason is None
    assert proposal.requires_approval is False
    assert "full_device_lab_catastrophic_destruction_armed" in proposal.payload["matched_rules"]
