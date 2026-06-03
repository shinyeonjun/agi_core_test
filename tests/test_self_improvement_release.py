import json

from agent.cli.agentctl import main
from agent.core.capabilities import collect_capability_map
from agent.core.database import init_db
from agent.core.self_improvement_release import (
    build_self_improvement_release_plan,
    classify_release_paths,
    evaluate_release_candidate,
    normalize_changed_files,
)


def test_normalize_changed_files_handles_git_status_lines():
    assert normalize_changed_files([" M agent/core/foo.py", "R  old.py -> agent/core/new.py", "?? tests/test_x.py"]) == [
        "agent/core/foo.py",
        "agent/core/new.py",
        "tests/test_x.py",
    ]


def test_classify_release_paths_escalates_sensitive_changes():
    result = classify_release_paths([" M .env", " M systemd/agent-core.service", " M pyproject.toml"])

    assert result["inferred_risk"] == "critical"
    assert ".env" in result["secret_paths"]
    assert "systemd/agent-core.service" in result["service_paths"]
    assert "pyproject.toml" in result["dependency_paths"]
    assert set(result["change_tags"]) >= {"secret_boundary", "service_change", "dependency_change"}


def test_release_gate_ready_only_after_full_quality_chain():
    result = evaluate_release_candidate(
        changed_files=[" M agent/core/example.py"],
        worktree_isolated=True,
        tests_passed=True,
        audit_passed=True,
        eval_passed=True,
        review_passed=True,
        approval_status="approved",
        risk_level="medium",
        rollback_plan="git branch backup before merge",
    )

    assert result["status"] == "ready_to_apply"
    assert result["can_apply_to_main"] is True
    assert result["can_auto_apply"] is False
    assert result["approval_required"] is True
    assert result["failed_gates"] == []


def test_release_gate_blocks_secrets_and_non_isolated_work():
    result = evaluate_release_candidate(
        changed_files=[" M .env"],
        worktree_isolated=False,
        tests_passed=True,
        audit_passed=True,
        eval_passed=True,
        review_passed=True,
        approval_status="approved",
        risk_level="low",
        rollback_plan="git checkout -- .env",
    )

    assert result["status"] == "blocked"
    assert set(result["failed_gates"]) >= {"isolation", "secret_boundary"}
    assert result["effective_risk"] == "critical"
    assert result["can_apply_to_main"] is False


def test_release_gate_waits_for_review_and_approval():
    review_result = evaluate_release_candidate(
        changed_files=[" M agent/core/example.py"],
        worktree_isolated=True,
        tests_passed=True,
        audit_passed=True,
        eval_passed=True,
        review_passed=False,
        approval_status="pending",
        rollback_plan="drop worktree",
    )
    approval_result = evaluate_release_candidate(
        changed_files=[" M agent/core/example.py"],
        worktree_isolated=True,
        tests_passed=True,
        audit_passed=True,
        eval_passed=True,
        review_passed=True,
        approval_status="pending",
        rollback_plan="drop worktree",
    )

    assert review_result["status"] == "needs_review"
    assert "review" in review_result["review_gates"]
    assert approval_result["status"] == "needs_approval"
    assert "approval" in approval_result["approval_gates"]


def test_release_plan_exposes_research_backed_phases():
    plan = build_self_improvement_release_plan("Core 개선", request="자가개선 신뢰성 강화")

    assert [phase["name"] for phase in plan["phases"]] == [
        "scope",
        "isolate",
        "implement",
        "verify",
        "review",
        "approve",
        "apply",
        "observe",
        "rollback",
    ]
    assert {item["key"] for item in plan["research_basis"]} >= {"react", "reflexion", "swe_agent"}


def test_release_cli_plan_and_gate(capsys):
    assert main(["release", "plan", "Core 개선", "--request", "자가개선 신뢰성 강화"]) == 0
    plan_payload = json.loads(capsys.readouterr().out)
    assert plan_payload["title"] == "Core 개선"
    assert len(plan_payload["phases"]) == 9

    assert main(
        [
            "release",
            "gate",
            "--changed-file",
            " M agent/core/example.py",
            "--tests-passed",
            "--audit-passed",
            "--eval-passed",
            "--review-passed",
            "--approval",
            "approved",
            "--rollback-plan",
            "drop worktree",
        ]
    ) == 0
    gate_payload = json.loads(capsys.readouterr().out)
    assert gate_payload["status"] == "ready_to_apply"


def test_capability_map_mentions_release_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "repo"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    (tmp_path / "repo").mkdir()
    init_db()

    capabilities = collect_capability_map()

    assert any(item["name"] == "self_improvement_release_gate" for item in capabilities["direct"])
    assert any(item["name"] == "self_improvement_code_planner" for item in capabilities["direct"])
