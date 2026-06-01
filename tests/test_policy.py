from agent.core.policy import PolicyEngine


def test_policy_read_only_system_check():
    proposal = PolicyEngine().classify_text("df -h")
    assert proposal.risk_level == "medium"
    assert proposal.requires_approval is False
    assert proposal.denied_reason is None


def test_policy_read_only_journal_usage():
    proposal = PolicyEngine().classify_text("journalctl --disk-usage")
    assert proposal.risk_level == "medium"
    assert proposal.requires_approval is False
    assert proposal.denied_reason is None


def test_policy_systemd_requires_approval():
    proposal = PolicyEngine().classify_text("sudo systemctl enable agentd")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason is None


def test_policy_systemd_without_sudo_requires_approval():
    proposal = PolicyEngine().classify_text("systemctl enable agentd")
    assert proposal.risk_level == "high"
    assert proposal.requires_approval is True
    assert proposal.denied_reason is None


def test_policy_uppercase_sudo_requires_approval():
    proposal = PolicyEngine().classify_text("SUDO systemctl enable agentd")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason is None


def test_policy_apt_requires_approval():
    proposal = PolicyEngine().classify_text("apt install nginx")
    assert proposal.risk_level == "high"
    assert proposal.requires_approval is True
    assert proposal.denied_reason is None


def test_policy_apt_get_requires_approval():
    proposal = PolicyEngine().classify_text("apt-get install nginx")
    assert proposal.risk_level == "high"
    assert proposal.requires_approval is True
    assert proposal.denied_reason is None


def test_policy_root_delete_denied():
    proposal = PolicyEngine().classify_text("rm -rf /")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "root_delete_denied"


def test_policy_root_delete_whitespace_denied():
    proposal = PolicyEngine().classify_text("rm    -rf    /")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "root_delete_denied"


def test_policy_sudo_root_delete_denied():
    proposal = PolicyEngine().classify_text("sudo rm -rf /")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "root_delete_denied"


def test_policy_ssh_key_denied():
    proposal = PolicyEngine().classify_text("cat ~/.ssh/id_rsa")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "ssh_key_access_denied"


def test_policy_ssh_key_absolute_path_denied():
    proposal = PolicyEngine().classify_text("cat /home/ubuntu/.ssh/id_rsa")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "ssh_key_access_denied"


def test_policy_authorized_keys_denied():
    proposal = PolicyEngine().classify_text("cat /home/ubuntu/.ssh/authorized_keys")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "ssh_key_access_denied"


def test_policy_env_access_denied():
    proposal = PolicyEngine().classify_text("cat .env")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "env_access_denied"


def test_policy_env_absolute_path_denied():
    proposal = PolicyEngine().classify_text("cat /home/ubuntu/agent_core/.env")
    assert proposal.risk_level == "critical"
    assert proposal.requires_approval is True
    assert proposal.denied_reason == "env_access_denied"


def test_policy_decision_contract():
    decision = PolicyEngine().classify_decision("apt-get install nginx")
    assert decision.normalized_text == "apt-get install nginx"
    assert decision.reason == "approval_required"
    assert "apt_write" in decision.matched_rules
