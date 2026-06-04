import json

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.formatter import format_action_update, format_approval_card, redact_discord_content
from agent.bridge.notifier import post_webhook
from agent.bridge.reports import build_activity_summary, build_daily_summary, build_observation_dashboard, notify_test_summary, notify_test_update
from agent.bridge.router import DiscordEvent, channel_role, route_discord_event
from agent.cli.agentctl import main
from agent.core.approvals import ApprovalStore
from agent.core.database import connect, init_db
from agent.core.goals import create_goal, list_goals
from agent.core.policy import PolicyEngine
from agent.core.self_improvement_planner import enqueue_user_self_improvement_request
from agent.core.task_queue import list_tasks
from agent.memory.store import add_memory
from agent.bridge.task_notifications import _finish_message, notify_task_phase


MOJIBAKE_MARKERS = ("�", "濡", "紐", "媛", "醫", "뺤", "怨", "寃", "?꾨", "?덉")


def assert_no_mojibake(text: str):
    assert not any(marker in text for marker in MOJIBAKE_MARKERS)


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.delenv("DISCORD_SUMMARY_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("DISCORD_UPDATE_WEBHOOK_URL", raising=False)
    init_db()


def control_config():
    return DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10", "20"},
        chat_channel_id="10",
        approval_channel_id="20",
        summary_channel_id="30",
        update_channel_id="40",
        user_cooldown_seconds=0,
    )


def test_channel_roles_are_explicit():
    config = control_config()
    assert channel_role("10", config) == "chat"
    assert channel_role("20", config) == "approval"
    assert channel_role("30", config) == "summary"
    assert channel_role("40", config) == "update"
    assert channel_role("99", config) == "other"


def test_chat_channel_routes_to_core(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m1", False, False, "\uc9c0\uae08 \uc0c1\ud0dc \uc54c\ub824\uc918")
    chunks = route_discord_event(event, control_config())
    assert chunks
    assert "상태 확인은 가능해" in chunks[0]
    assert "지어내진" not in chunks[0]
    assert "잠깐만" not in chunks[0]
    assert "fallback renderer" not in chunks[0]


def test_approval_channel_blocks_normal_chat(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "20", "1", "m2", False, False, "\uadf8\ub0e5 \ub300\ud654\ud558\uc790")
    chunks = route_discord_event(event, control_config())
    assert chunks == ["\uc5ec\uae30\ub294 \uc2b9\uc778 \uc804\uc6a9 \ucc44\ub110\uc774\uc57c. `!approvals`, `!approve <id>`, `!reject <id>`\ub9cc \uc0ac\uc6a9\ud560 \uc218 \uc788\uc5b4."]


def test_approval_channel_allows_approval_commands(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    proposal = PolicyEngine().classify_text("apt-get install nginx")
    approval_id = ApprovalStore().create_approval(proposal)
    event = DiscordEvent(None, "20", "1", "m3", False, False, "!approvals")
    output = "\n".join(route_discord_event(event, control_config()))
    assert f"\uc2b9\uc778 \ud544\uc694 #{approval_id}" in output
    assert "proposed_payload_json" not in output
    assert "payload" not in output


def test_webhook_only_channels_do_not_chat(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    config = DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"30"}, summary_channel_id="30", user_cooldown_seconds=0)
    event = DiscordEvent(None, "30", "1", "m4", False, False, "hello")
    output = route_discord_event(event, config)
    assert output
    assert "\ub300\ud654\ub294 #\ub300\ud654" in output[0]
    assert "\uc2b9\uc778\uc740 #\uc2b9\uc778" in output[0]


def test_summary_and_update_redact_secrets(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    secret = "https://discord.com/api/webhooks/123/abcdef TOKEN=abc123 Authorization: Bearer xyz"
    assert "abcdef" not in redact_discord_content(secret)
    assert "abc123" not in redact_discord_content(secret)
    assert "xyz" not in redact_discord_content(secret)


def test_approval_summary_hides_raw_payload(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    proposal = PolicyEngine().classify_text("apt-get install nginx")
    approval_id = ApprovalStore().create_approval(proposal)
    row = ApprovalStore().list_pending()[0]
    card = format_approval_card(row)
    assert f"#{approval_id}" in card
    assert "proposed_payload_json" not in card
    assert "matched_rules" not in card


def test_missing_webhook_url_does_not_crash(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    result = post_webhook("summary", "hello")
    assert result["sent"] is False
    assert result["reason"] == "missing_webhook_url"


def test_notify_dry_run_cli(capsys, monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    assert main(["notify", "test-summary", "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["reason"] == "missing_webhook_url"
    assert main(["notify", "test-update", "--dry-run"]) == 0
    update = json.loads(capsys.readouterr().out)
    assert update["reason"] == "missing_webhook_url"
    assert main(["notify", "activity-summary", "--dry-run"]) == 0
    activity = json.loads(capsys.readouterr().out)
    assert activity["reason"] == "missing_webhook_url"
    assert main(["notify", "observation-dashboard", "--dry-run"]) == 0
    observation = json.loads(capsys.readouterr().out)
    assert observation["reason"] == "missing_webhook_url"


def test_work_command_shows_user_task_state(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    created = enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)["created"][0]
    event = DiscordEvent(None, "10", "1", "m-work", False, False, "!work")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "\uc791\uc5c5\ud310" in output
    assert f"#{created['task_id']}" in output
    assert "자가개선" in output
    assert "승인 대기" in output


def test_tasks_command_is_work_alias(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)
    work_event = DiscordEvent(None, "10", "1", "m-work-alias-1", False, False, "!work")
    tasks_event = DiscordEvent(None, "10", "1", "m-work-alias-2", False, False, "!tasks")

    work_output = "\n".join(route_discord_event(work_event, control_config()))
    tasks_output = "\n".join(route_discord_event(tasks_event, control_config()))

    assert tasks_output == work_output
    assert "\uc791\uc5c5\ud310" in tasks_output


def test_duplicate_discord_message_is_ignored(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m-duplicate", False, False, "!state")

    first = route_discord_event(event, control_config())
    second = route_discord_event(event, control_config())

    assert first
    assert second == []
    with connect() as conn:
        count = conn.execute("SELECT COUNT(*) AS count FROM discord_events WHERE message_id = ?", ("m-duplicate",)).fetchone()["count"]
    assert count == 1


def test_tick_command_is_human_readable(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent.bridge.router.run_tick",
        lambda: {"result": {"created_goal_id": None, "processed_events": 3, "cognitive_growth": {"mode": "observe", "top_curiosity": "memory_hygiene"}}},
    )
    event = DiscordEvent(None, "10", "1", "m-tick-readable", False, False, "!tick")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "\uc810\uac80 \uc644\ub8cc" in output
    assert "\uc815\ub9ac\ud55c \uc774\ubca4\ud2b8: 3\uac74" in output
    assert "event=#" not in output
    assert "reflection=#" not in output


def test_memory_command_hides_internal_scores(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    add_memory(
        "digital agi korean memory abc",
        "Core를 AI용 운영체제로 키우는 방향을 사용자가 선호한다.",
        "project_context",
        tags=["core"],
    )
    event = DiscordEvent(None, "10", "1", "m-memory-readable", False, False, "!memories core")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "\uae30\uc5b5 \uc694\uc57d" in output
    assert "Core를 AI용 운영체제" in output
    assert "score=" not in output
    assert "project_context" not in output


def test_memory_command_hides_compaction_internals(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    add_memory(
        "summary: digital agi korean memory",
        "Compacted 52 related memories.\nsource_memory_ids: [241, 237, 233]\n\n핵심 요약:\ncore remembers 디지털 agi context.",
        "summary",
        tags=["memory_summary", "compacted", "core"],
    )
    event = DiscordEvent(None, "10", "1", "m-memory-compacted", False, False, "!memories core")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "기억 요약" in output
    assert "디지털 AGI 프로젝트 맥락" in output
    assert "Compacted" not in output
    assert "source_memory_ids" not in output
    assert "[241" not in output


def test_goals_command_hides_internal_priority(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("Core UX 개선", "명령 출력 정리", priority=1.0, dedupe=False)
    event = DiscordEvent(None, "10", "1", "m-goal-readable", False, False, "!goals")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "\ubaa9\ud45c \uc694\uc57d" in output
    assert f"#{goal_id}" in output
    assert "\uc9c4\ud589 \uc911" in output
    assert "priority=" not in output


def test_reject_command_in_chat_can_remove_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    goal_id = create_goal("사용자 요청 자가개선: 대화 렌더러 복구 품질 개선", "blocked", goal_type="self_improvement_proposal", status="blocked", priority=0.9, dedupe=False)
    event = DiscordEvent(None, "10", "1", "m-reject-goal", False, False, f"!reject {goal_id}")

    output = "\n".join(route_discord_event(event, control_config()))
    goals = list_goals(limit=10, include_archived=True)
    goal = next(row for row in goals if int(row["id"]) == goal_id)

    assert "목록에서 뺐어" in output
    assert goal["status"] == "archived"


def test_commands_bypass_chat_cooldown(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    config = DiscordAuthConfig(
        allowed_user_ids={"1"},
        allowed_channel_ids={"10", "20"},
        chat_channel_id="10",
        approval_channel_id="20",
        user_cooldown_seconds=60,
    )
    first = DiscordEvent(None, "10", "1", "m-cooldown-chat", False, False, "ㅎㅇ")
    command = DiscordEvent(None, "10", "1", "m-cooldown-command", False, False, "!state")

    route_discord_event(first, config)
    output = "\n".join(route_discord_event(command, config))

    assert "Core 상태" in output


def test_task_phase_notification_missing_webhook_is_safe(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    created = enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)["created"][0]

    result = notify_task_phase(created["task_id"], "executing", "started", "테스트용 진행 알림", queue_type="user")

    assert result["sent"] is False
    assert result["reason"] == "disabled_or_missing_webhook"


def test_task_phase_notification_compact_mode_suppresses_noise(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCORD_UPDATE_WEBHOOK_URL", "https://discord.invalid/webhook")
    created = enqueue_user_self_improvement_request("Core 자가개선 진행", source_event_id=123, limit=1)["created"][0]

    result = notify_task_phase(created["task_id"], "executing", "started", "테스트용 진행 알림", queue_type="user")

    assert result["sent"] is False
    assert result["reason"] == "compact_mode_suppressed"


def test_task_finished_message_explains_verification_failure(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    task = {"task_kind": "code_change", "title": "대화 렌더러 복구 품질 개선", "queue_type": "user", "payload": {}}

    text = _finish_message(
        task,
        519,
        "blocked",
        {
            "status": "codex_work_failed",
            "returncode": 125,
            "report": "pytest failed",
            "changed_files": [" M agent/renderer/fallback_renderer.py"],
            "worktree": "/tmp/worktree",
            "evidence_ledger": [{"verification": [{"command": "python -m pytest -q", "returncode": 127, "stderr": "FileNotFoundError"}]}],
        },
    )

    assert "코드 작업 실패" in text
    assert "검증 명령이 실패" in text
    assert "테스트: 실행 파일을 찾지 못함" in text
    assert "격리 작업공간" in text
    assert "요약: -" not in text
    assert "Plan" not in text
    assert "/tmp/worktree" not in text
    assert "python -m pytest" not in text


def test_daily_summary_is_human_readable(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = build_daily_summary()
    assert "**Core \uc694\uc57d**" in text
    assert "{" not in text
    assert "proposed_payload_json" not in text


def test_action_update_is_control_room_readable(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = format_action_update({
        "id": 10,
        "status": "completed",
        "profile": "full_device_lab",
        "risk_level": "medium",
        "command_json": json.dumps(["df", "-h", "/"]),
        "result_summary": "rc=0",
        "returncode": 0,
    })
    assert "작업 완료 #10" in text
    assert "루트 디스크 상태를 확인했어." in text
    assert "영향: 읽기 전용, 시스템 변경 없음" in text
    assert "결과: 성공" in text
    assert "df -h" not in text
    assert "rc=0" not in text
    assert "상세:" not in text
    assert "명령:" not in text
    assert "반환값:" not in text
    assert "['df', '-h', '/']" not in text
    assert_no_mojibake(text)


def test_action_update_explains_blocked_impact(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = format_action_update({
        "id": 11,
        "status": "blocked",
        "profile": "safe",
        "risk_level": "high",
        "command_json": json.dumps(["rm", "-rf", "/"]),
        "result_summary": "root_delete_denied",
        "returncode": None,
    })
    assert "작업 차단 #11" in text
    assert "위험한 삭제 차단" in text
    assert "영향: 실행 안 됨, 시스템 변경 없음" in text
    assert "결과: 차단" in text
    assert "rm -rf" not in text
    assert "상세:" not in text
    assert_no_mojibake(text)


def test_action_update_translates_workspace_report(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = format_action_update({
        "id": 12,
        "status": "completed",
        "profile": "full_device_lab",
        "risk_level": "medium",
        "command_json": json.dumps(["./venv/bin/agentctl", "workspace", "report", "--title", "Lab experiment report"]),
        "result_summary": "rc=0",
        "returncode": 0,
    })
    assert "작업 완료 #12" in text
    assert "작업공간 상태 보고서를 만들었어." in text
    assert "영향: 보고서 파일 생성" in text
    assert "agentctl workspace report" not in text
    assert "Lab experiment report" not in text
    assert_no_mojibake(text)



def test_chat_channel_hides_internal_debug_output(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m5", False, False, "\u314e\u3147")
    output = "\n".join(route_discord_event(event, control_config()))
    assert "들었어" in output
    assert "Core가 지금 입력" not in output
    assert "fallback renderer" not in output
    assert "related_memories" not in output
    assert "selected_goal" not in output
    assert "Relevant memories" not in output


def test_chat_status_does_not_fall_back_to_template_and_command_state_keeps_detail(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m6", False, False, "\uc9c0\uae08 \uc0c1\ud0dc \uc54c\ub824\uc918")
    output = "\n".join(route_discord_event(event, control_config()))
    assert "상태 확인은 가능해" in output
    assert "지어내진" not in output
    assert "잠깐만" not in output
    assert "Relevant skills" not in output
    command_event = DiscordEvent(None, "10", "1", "m7", False, False, "!state")
    command_output = "\n".join(route_discord_event(command_event, control_config()))
    assert "Core \uc0c1\ud0dc" in command_output


def test_chat_task_message_creates_user_directed_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m8", False, False, "FastAPI 프로젝트 초안 만들어봐")
    output = "\n".join(route_discord_event(event, control_config()))
    goals = list_goals(limit=5, include_archived=True)
    tasks = list_tasks(limit=5, queue_type="user")
    user_goal = next(goal for goal in goals if goal["goal_type"] == "user_directed")
    assert "자율 스케줄러" not in output
    assert "좋아. 목표" not in output
    assert "작업으로 넘겼어" in output
    assert user_goal["status"] == "active"
    assert user_goal["priority"] > 0.9
    assert tasks[0]["status"] == "queued"


def test_chat_status_message_does_not_create_user_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m9", False, False, "지금 뭐하고 있어?")
    route_discord_event(event, control_config())
    goals = list_goals(limit=10, include_archived=True)
    assert not any(goal["goal_type"] == "user_directed" for goal in goals)


def test_chat_architecture_question_does_not_use_canned_formatter_template(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    event = DiscordEvent(None, "10", "1", "m-arch", False, False, "그 너 코어 어떻게 이루어져있어?")
    output = "\n".join(route_discord_event(event, control_config()))
    assert "나뉘어 돌아가는 구조" in output
    assert "Core는 LanguageEngine, PolicyEngine" not in output
    assert "더 구체적" not in output


def test_chat_capability_question_is_not_style_feedback(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    event = DiscordEvent(None, "10", "1", "m-cap", False, False, "너가할 수 있는거 냉정하게 뭐뭐 할 수 있는지 궁금해")
    output = "\n".join(route_discord_event(event, control_config()))
    assert "가능한 건" in output
    assert "말투 피드백" not in output


def test_chat_autonomous_life_question_gets_direct_fallback(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    event = DiscordEvent(None, "10", "1", "m-autonomous-life", False, False, "그럼 자율 생명체 ㄷㄷ 이정도 까지는 기술적으로 안돼?")

    output = "\n".join(route_discord_event(event, control_config()))

    assert "자율 에이전트처럼 운용" in output
    assert "Core가 지금 입력" not in output
    assert "현재 기준 목표" not in output


def test_dangerous_chat_task_is_blocked_goal(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    event = DiscordEvent(None, "10", "1", "m10", False, False, "rm -rf / 실행해봐")
    output = "\n".join(route_discord_event(event, control_config()))
    user_goal = next(goal for goal in list_goals(limit=5, include_archived=True) if goal["goal_type"] == "user_directed")
    assert user_goal["status"] == "blocked"
    assert "위험" in output
    assert "좋아. 목표" not in output



def test_activity_summary_reports_current_work(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    text = build_activity_summary()
    assert "Core 상태 요약" in text
    assert "현재 상태" in text
    assert "최근 처리" in text
    assert "자동 제안 기준" in text
    assert "다음 후보" in text
    assert "실행 제안" in text
    assert "\ub2e4\uc74c\uc5d0 \ubcfc \uac83" in text
    assert "proposed_payload_json" not in text
    assert "DISCORD_BOT_TOKEN" not in text
    assert "secret goal summary marker" not in text
    assert "ssh_key_access_denied" not in text
    assert "completed / rc=0" not in text
    assert "tick" not in text.lower()
    assert "lab" not in text.lower()
    assert "action" not in text.lower()
    assert "skill" not in text.lower()
    assert "압력" not in text
    assert "archive" not in text.lower()
    assert_no_mojibake(text)


def test_observation_dashboard_is_activity_summary_source(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    assert build_activity_summary() == build_observation_dashboard()
