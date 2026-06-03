from agent.core.self_code_review import review_codex_work_result


def _verification(commands):
    return {
        "evidence_ledger": [
            {
                "verification": [
                    {"command": command, "returncode": 0, "stdout": "ok", "stderr": ""}
                    for command in commands
                ]
            }
        ]
    }


def test_self_code_review_blocks_self_improvement_without_worktree():
    result = {
        "status": "codex_work_completed",
        "changed_files": [" M agent/core/example.py"],
        "unsafe_changed_files": [],
        "report": "수정 완료",
    }

    review = review_codex_work_result(result, self_improvement=True)

    assert review["verdict"] == "blocked"
    assert any(item["code"] == "self_improvement_not_isolated" for item in review["findings"])
    assert review["release_gate"]["can_apply_to_main"] is False


def test_self_code_review_waits_for_approval_after_full_verification():
    result = {
        "status": "codex_work_completed",
        "changed_files": [" M agent/core/example.py"],
        "unsafe_changed_files": [],
        "report": "수정 완료",
        "worktree_status": "created",
        "worktree_branch": "codex/native-loop-task-1-abcd1234",
        **_verification(["python -m pytest -q", "python -m agent.cli.agentctl audit", "python -m agent.cli.agentctl eval run"]),
    }

    review = review_codex_work_result(result, self_improvement=True)

    assert review["verdict"] == "needs_approval"
    assert review["release_gate"]["status"] == "needs_approval"
    assert review["release_gate"]["can_auto_apply"] is False


def test_self_code_review_reports_missing_audit_and_eval():
    result = {
        "status": "codex_work_completed",
        "changed_files": [" M agent/core/example.py"],
        "unsafe_changed_files": [],
        "report": "수정 완료",
        "worktree_status": "created",
        **_verification(["python -m pytest -q"]),
    }

    review = review_codex_work_result(result, self_improvement=True)
    codes = {item["code"] for item in review["findings"]}

    assert review["verdict"] == "needs_validation"
    assert {"audit_missing", "eval_missing"} <= codes


def test_self_code_review_requires_successful_audit_and_eval_commands():
    result = {
        "status": "codex_work_completed",
        "changed_files": [" M agent/core/example.py"],
        "unsafe_changed_files": [],
        "report": "?섏젙 ?꾨즺",
        "worktree_status": "created",
        "evidence_ledger": [
            {
                "verification": [
                    {"command": "python -m pytest -q", "returncode": 0, "stdout": "ok", "stderr": ""},
                    {"command": "python -m agent.cli.agentctl audit", "returncode": 127, "stdout": "", "stderr": "FileNotFoundError"},
                    {"command": "python -m agent.cli.agentctl eval run", "returncode": 127, "stdout": "", "stderr": "FileNotFoundError"},
                ]
            }
        ],
    }

    review = review_codex_work_result(result, self_improvement=True)
    codes = {item["code"] for item in review["findings"]}

    assert review["verdict"] == "needs_validation"
    assert {"audit_missing", "eval_missing"} <= codes
