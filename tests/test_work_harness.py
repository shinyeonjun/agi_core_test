from agent.lab.work_harness import build_work_harness_contract, format_work_harness_prompt, summarize_work_harness, verification_gate_summary


def test_work_harness_contract_carries_scope_and_gates():
    contract = build_work_harness_contract(
        "Core 렌더러 개선해줘",
        goal_id=10,
        task_id=20,
        backend="native_loop",
        sandbox="workspace-write",
        max_iterations=2,
        verification_commands=["python -m agent.cli.agentctl test run fast --json"],
        self_improvement=True,
    )

    summary = summarize_work_harness(contract)

    assert contract.mode == "native_work_harness"
    assert "approve" in contract.phases
    assert "read_or_store_secrets" in contract.forbidden_actions
    assert summary["verification_count"] == 1
    assert summary["self_improvement"] is True


def test_work_harness_prompt_redacts_secret_like_text():
    contract = build_work_harness_contract(
        "token=abc123 Core 고쳐줘",
        goal_id=None,
        task_id=None,
        backend="native_loop",
        sandbox="workspace-write",
        max_iterations=1,
    )

    prompt = format_work_harness_prompt(contract)

    assert "[REDACTED]" in prompt
    assert "abc123" not in prompt
    assert "Work Harness Contract" in prompt


def test_verification_gate_summary_uses_latest_evidence():
    evidence = [
        {"verification": [{"command": "pytest", "returncode": 1}]},
        {"verification": [{"command": "pytest", "returncode": 0}, {"command": "audit", "returncode": 0}]},
    ]

    result = verification_gate_summary(evidence)

    assert result == {"passed": True, "total": 2, "failed": []}
