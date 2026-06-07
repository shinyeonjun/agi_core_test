import asyncio

import io
import urllib.error

import pytest

from neurokernel_seed.discord_bot.core_client import CoreClient, CoreClientError
from neurokernel_seed.discord_bot.bot import (
    DiscordBotConfig,
    _command_line_from_content,
    _discord_chunks,
    _handle_command,
    _is_auto_executable_task,
    _message_allowed,
    _parse_user_ids,
)
from neurokernel_seed.discord_bot.commands import build_benchmark_task, build_preset_task, parse_task_json
from neurokernel_seed.discord_bot.work_status import build_work_status_payload as _build_work_status_payload
from neurokernel_seed.discord_bot.work_status import format_work_notification as _format_work_notification
from neurokernel_seed.discord_bot.work_status import latest_self_patch_result as _latest_self_patch_result


TEST_CHANNEL_ID = 123456789012345678


def _bot_config(**overrides):
    payload = {"token": "token", "channel_id": TEST_CHANNEL_ID, "core_url": "http://127.0.0.1:8765", "prefix": ""}
    payload.update(overrides)
    return DiscordBotConfig(**payload)


def test_build_preset_task_uses_readonly_action():
    task = build_preset_task("disk")
    assert task["allowed_actions"] == ["get_disk_usage"]
    assert task["requires_approval"] is False
    assert task["mode"] == "readonly"


def test_build_preset_task_can_inspect_code_structure():
    task = build_preset_task("code-structure")
    assert task["allowed_actions"] == ["inspect_code_structure"]
    assert task["context"]["params"]["paths"] == ["src", "tests"]
    assert task["requires_approval"] is False
    assert task["mode"] == "readonly"


def test_build_preset_task_rejects_unknown_preset():
    with pytest.raises(ValueError):
        build_preset_task("delete_everything")


def test_build_benchmark_task_is_low_risk():
    task = build_benchmark_task(episodes=2)
    assert task["allowed_actions"] == ["run_safe_benchmark"]
    assert task["context"]["params"]["episodes"] == 2
    assert task["context"]["params"]["model"] == "artifacts/current_world_model.onnx"
    assert task["risk_level"] == "low"


def test_parse_task_json_requires_object():
    assert parse_task_json('{"goal":"x"}') == {"goal": "x"}
    with pytest.raises(ValueError):
        parse_task_json("[1, 2, 3]")


def test_message_allowed_only_configured_channel():
    class Channel:
        id = TEST_CHANNEL_ID

    class Author:
        id = 123

    class Message:
        channel = Channel()
        guild = object()
        author = Author()

    config = _bot_config()
    assert _message_allowed(Message(), config)


def test_message_denies_other_channels_and_dms_by_default():
    class OtherChannel:
        id = 1

    class GuildMessage:
        channel = OtherChannel()
        guild = object()
        author = type("Author", (), {"id": 123})()

    class DmMessage:
        channel = OtherChannel()
        guild = None
        author = type("Author", (), {"id": 123})()

    config = _bot_config()
    assert not _message_allowed(GuildMessage(), config)
    assert not _message_allowed(DmMessage(), config)


def test_discord_chunks_keeps_short_messages_intact():
    assert _discord_chunks("hello") == ["hello"]
    assert len(_discord_chunks("x" * 4000, limit=1000)) == 4


def test_allowed_user_filter_blocks_other_users():
    class Channel:
        id = TEST_CHANNEL_ID

    class Message:
        channel = Channel()
        guild = object()
        author = type("Author", (), {"id": 222})()

    config = _bot_config(allowed_user_ids=(111,))
    assert not _message_allowed(Message(), config)


def test_allowed_user_filter_allows_configured_user():
    class Channel:
        id = TEST_CHANNEL_ID

    class Message:
        channel = Channel()
        guild = object()
        author = type("Author", (), {"id": 111})()

    config = _bot_config(allowed_user_ids=(111,))
    assert _message_allowed(Message(), config)


def test_parse_user_ids_accepts_single_or_many_values():
    assert _parse_user_ids("1") == (1,)
    assert _parse_user_ids("1, 2;3") == (1, 2, 3)


def test_core_client_extracts_structured_http_error_message(monkeypatch):
    body = b'{"detail":{"error_type":"activation_failed","message":"activation requires a clean tree"}}'

    def fail_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1/work-items/work_1/activate",
            code=409,
            msg="Conflict",
            hdrs={},
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    with pytest.raises(CoreClientError, match="clean tree"):
        CoreClient("http://127.0.0.1").post("/work-items/work_1/activate", {"actor": "test"})


def test_empty_prefix_routes_unknown_text_to_conversation():
    config = _bot_config(prefix="", reply_without_prefix=True)
    assert _command_line_from_content("ㅎㅎㅎㅎ", config) == "auto ㅎㅎㅎㅎ"
    assert _command_line_from_content("암마", config) == "auto 암마"


def test_empty_prefix_keeps_known_commands_available():
    config = _bot_config(prefix="", reply_without_prefix=True)
    assert _command_line_from_content("help", config) == "help"
    assert _command_line_from_content("memory recent", config) == "memory recent"


def test_nonempty_prefix_keeps_prefixed_command_mode():
    config = _bot_config(prefix="!nk", reply_without_prefix=False)
    assert _command_line_from_content("ㅎㅎㅎㅎ", config) is None
    assert _command_line_from_content("!nk help", config) == "help"


def test_auto_work_status_uses_work_router_and_language_humanizer():
    class FakeCore:
        def __init__(self):
            self.calls = []

        def get(self, path):
            self.calls.append(("GET", path))
            if path == "/work-items?limit=10":
                return {
                    "work_items": [
                        {
                            "work_id": "work_item_1",
                            "type": "external_work",
                            "title": "자동 진단 리포트 기능",
                            "status": "planned",
                        }
                    ]
                }
            if path == "/work-jobs?limit=10":
                return {
                    "jobs": [
                        {
                            "job_id": "job_1",
                            "work_id": "work_item_1",
                            "work_title": "자동 진단 리포트 기능",
                            "status": "completed",
                        }
                    ]
                }
            raise AssertionError(f"unexpected GET {path}")

        def post(self, path, payload=None):
            self.calls.append(("POST", path))
            if path == "/language/preferences":
                return {"kind": "none"}
            if path == "/language/to-core":
                return {
                    "reply": "최근 기록을 확인해볼게.",
                    "task_spec": {
                        "risk_level": "low",
                        "requires_approval": False,
                        "mode": "readonly",
                        "allowed_actions": ["get_recent_trace"],
                    },
                }
            if path == "/work/route":
                return {
                    "route": "work_status",
                    "reason": "The user asks for current work progress.",
                    "confidence": 0.95,
                    "work_item": None,
                    "requires_confirmation": False,
                    "clarifying_question": None,
                    "safety_notes": [],
                }
            if path == "/language/to-human":
                core_result = payload.get("core_result") if isinstance(payload, dict) else {}
                assert core_result.get("kind") == "work_status"
                assert core_result.get("work_items")[0]["title"] == "자동 진단 리포트 기능"
                assert core_result.get("progress")[0]["automation_stage"] == "plan_recorded_no_implementation_worker_running"
                return {"reply": "진행 중인 건 자동 진단 리포트 기능이고, 최근 실행은 끝났어."}
            raise AssertionError(f"unexpected POST {path}")

    core = FakeCore()
    config = _bot_config(prefix="", reply_without_prefix=True)
    response = asyncio.run(_handle_command("auto 현재 작업 상태 보여줘", core, config, user_id="u1", channel_id="c1"))

    assert response.text == "진행 중인 건 자동 진단 리포트 기능이고, 최근 실행은 끝났어."
    assert core.calls == [
        ("POST", "/language/preferences"),
        ("POST", "/language/to-core"),
        ("POST", "/work/route"),
        ("GET", "/work-items?limit=10"),
        ("GET", "/work-jobs?limit=10"),
        ("POST", "/language/to-human"),
    ]


def test_work_status_payload_explains_external_work_is_planned_not_attached():
    payload = _build_work_status_payload(
        [
            {
                "work_id": "work1",
                "type": "external_work",
                "title": "자동 진단 리포트 기능",
                "status": "planned",
                "priority": "high",
                "risk_level": "low",
            }
        ],
        [{"job_id": "job1", "work_id": "work1", "status": "completed"}],
    )

    assert payload["kind"] == "work_status"
    assert payload["progress"][0]["automation_stage"] == "plan_recorded_no_implementation_worker_running"
    assert payload["progress"][0]["worker_action_required"] is True
    assert payload["progress"][0]["activation_possible"] is False
    assert payload["progress"][0]["promotion_possible"] is True


def test_work_status_payload_marks_self_patch_waiting_for_activation():
    payload = _build_work_status_payload(
        [
            {
                "work_id": "work2",
                "type": "self_patch",
                "title": "CPU 코어별 사용률 확인",
                "status": "waiting_approval",
                "priority": "high",
                "risk_level": "low",
            }
        ],
        [{"job_id": "job2", "work_id": "work2", "status": "completed"}],
    )

    assert payload["progress"][0]["automation_stage"] == "patch_ready_waiting_for_activation_approval"
    assert payload["progress"][0]["user_action_required"] is True
    assert payload["progress"][0]["activation_possible"] is True
    assert payload["progress"][0]["promotion_possible"] is False


def test_work_status_payload_hides_promotion_when_external_work_has_self_patch_child():
    payload = _build_work_status_payload(
        [
            {
                "work_id": "parent",
                "type": "external_work",
                "title": "diagnostic report",
                "status": "planned",
                "priority": "high",
                "risk_level": "low",
            },
            {
                "work_id": "child",
                "parent_work_id": "parent",
                "type": "self_patch",
                "title": "diagnostic report",
                "status": "waiting_approval",
                "priority": "high",
                "risk_level": "low",
            },
        ],
        [{"job_id": "job_child", "work_id": "child", "status": "completed"}],
    )

    parent = payload["progress"][0]
    assert parent["automation_stage"] == "implementation_patch_ready_waiting_for_activation"
    assert parent["child_work_id"] == "child"
    assert parent["activation_possible"] is True
    assert parent["promotion_possible"] is False


def test_format_work_notification_offers_promote_for_planned_external_work():
    text, view_kind = _format_work_notification(
        {
            "work_item": {
                "work_id": "work1",
                "type": "external_work",
                "title": "?먮룞 吏꾨떒 由ы룷??湲곕뒫",
                "status": "planned",
            },
            "events": [],
        }
    )

    assert view_kind == "promote"
    assert "계획" in text
    assert "개발 작업" in text


def test_format_work_notification_hides_promote_for_planned_external_work_with_child():
    text, view_kind = _format_work_notification(
        {
            "work_item": {
                "work_id": "parent",
                "type": "external_work",
                "title": "diagnostic report",
                "status": "planned",
            },
            "child_work_items": [
                {
                    "work_id": "child",
                    "type": "self_patch",
                    "title": "diagnostic report",
                    "status": "waiting_approval",
                }
            ],
            "events": [],
        }
    )

    assert view_kind is None
    assert "장착 승인" in text


def test_auto_executable_allows_low_risk_readonly_lookup():
    assert _is_auto_executable_task(
        {
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
            "allowed_actions": ["get_memory_usage"],
        }
    )


def test_auto_executable_allows_per_core_cpu_lookup():
    assert _is_auto_executable_task(
        {
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
            "allowed_actions": ["get_cpu_per_core_usage"],
        }
    )


def test_auto_executable_blocks_reboot_even_if_task_exists():
    assert not _is_auto_executable_task(
        {
            "risk_level": "high",
            "requires_approval": True,
            "mode": "readonly",
            "allowed_actions": ["reboot"],
        }
    )


def test_auto_executable_blocks_benchmark_for_conversational_mode():
    assert not _is_auto_executable_task(
        {
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
            "allowed_actions": ["run_safe_benchmark"],
        }
    )


def test_latest_self_patch_result_reads_completed_event_payload():
    result = _latest_self_patch_result(
        [
            {"event_type": "running", "payload_json": "{}"},
            {"event_type": "job_completed", "payload_json": '{"result":{"status":"test_failed","changed_files":["a.py"]}}'},
        ]
    )

    assert result == {"status": "test_failed", "changed_files": ["a.py"]}


def test_format_work_notification_reports_failed_patch_with_retry_button_kind():
    text, view_kind = _format_work_notification(
        {
            "work_item": {"work_id": "work1", "title": "CPU 코어별 사용률 확인", "status": "reviewing"},
            "events": [
                {
                    "event_type": "job_completed",
                    "payload_json": '{"result":{"status":"test_failed","changed_files":["src/x.py","tests/test_x.py"]}}',
                }
            ],
        }
    )

    assert view_kind == "retry"
    assert "테스트 실패" in text
    assert "CPU 코어별 사용률 확인" in text
    assert "src/x.py" in text


def test_format_work_notification_offers_activation_for_patch_ready():
    text, view_kind = _format_work_notification(
        {
            "work_item": {"work_id": "work1", "title": "CPU 코어별 사용률 확인", "status": "waiting_approval"},
            "events": [
                {
                    "event_type": "job_completed",
                    "payload_json": '{"result":{"status":"patch_ready","changed_files":["src/x.py"]}}',
                }
            ],
        }
    )

    assert view_kind == "activation"
    assert "테스트를 통과" in text


def test_format_work_notification_reports_codex_failed_patch_with_retry_button_kind():
    text, view_kind = _format_work_notification(
        {
            "work_item": {"work_id": "work1", "title": "CPU 코어별 사용률 확인", "status": "reviewing"},
            "events": [
                {
                    "event_type": "self_patch_failed",
                    "payload_json": '{"result":{"status":"codex_failed","error":"timeout"}}',
                }
            ],
        }
    )

    assert view_kind == "retry"
    assert "개발 워커 실행 실패" in text
