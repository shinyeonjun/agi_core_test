from agent.bridge.formatter import format_chat_reply


def test_format_chat_reply_treats_codex_work_completed_as_done():
    output = format_chat_reply(
        "코드 고쳐줘",
        {
            "decision": {
                "user_directed_goal": {"id": 7},
                "policy_summary": {"risk_level": "low", "requires_approval": False, "denied": False},
                "language_interpretation": {"intent": "task_request", "target": "code_change"},
            },
            "task_result": {
                "status": "codex_work_completed",
                "task_id": 3,
                "artifact_type": "codex_work_report",
                "report": "테스트 통과.",
            },
        },
    )

    assert "완료" in output
    assert "codex_work_report" in output
    assert "테스트 통과" in output
