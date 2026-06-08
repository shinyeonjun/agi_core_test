import pytest

from neurokernel_seed.discord_bot.models import DiscordBotConfig
from neurokernel_seed.discord_bot.notifier import is_autonomous_proposal, notify_work_changes
from neurokernel_seed.discord_bot.work_status import format_work_notification


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_proposed_self_patch_notification_uses_capability_proposal_view():
    text, view_kind = format_work_notification(
        {
            "work_item": {
                "work_id": "work_active_model",
                "type": "self_patch",
                "title": "Read active model status",
                "status": "proposed",
            },
            "capability_proposal": {"proposal_id": "prop_active_model"},
            "events": [],
        }
    )

    assert "Read active model status" in text
    assert view_kind == "proposal:prop_active_model"


def test_autonomous_model_training_proposal_is_first_poll_notifiable():
    assert is_autonomous_proposal(
        {
            "work_id": "work_runtime_training",
            "type": "training_pipeline",
            "status": "proposed",
            "linked_entity_type": "model_improvement",
            "metadata_json": {"execution_kind": "training_pipeline"},
        }
    )


@pytest.mark.anyio
async def test_first_poll_sends_autonomous_model_proposal():
    channel = FakeChannel()
    core = FakeCore(
        {
            "work_item": {
                "work_id": "work_runtime_training",
                "type": "training_pipeline",
                "title": "runtime_action 모델 재학습 후보",
                "status": "proposed",
                "linked_entity_type": "model_improvement",
                "metadata_json": {"model_slot": "runtime_action", "execution_kind": "training_pipeline"},
            },
            "events": [],
        }
    )
    item = {
        "work_id": "work_runtime_training",
        "type": "training_pipeline",
        "title": "runtime_action 모델 재학습 후보",
        "status": "proposed",
        "updated_at": "2026-06-08 00:21:41",
        "linked_entity_type": "model_improvement",
        "metadata_json": {"execution_kind": "training_pipeline"},
    }

    await notify_work_changes(
        FakeClient(channel),
        core,
        DiscordBotConfig(token="x", channel_id=123, core_url="http://127.0.0.1:8765", prefix="!"),
        proposal_view_factory=None,
        work_view_factory=lambda work_id: f"work-view:{work_id}",
        activation_view_factory=None,
        retry_view_factory=None,
        promote_view_factory=None,
        items=[item],
        seen={},
        first_poll=True,
    )

    assert channel.sent
    assert "runtime_action 모델 재학습 후보" in channel.sent[0]["content"]
    assert channel.sent[0]["view"] == "work-view:work_runtime_training"


@pytest.mark.anyio
async def test_first_poll_skips_non_autonomous_old_work():
    channel = FakeChannel()
    item = {
        "work_id": "work_old",
        "type": "external_work",
        "title": "old work",
        "status": "planned",
        "updated_at": "2026-06-07 00:00:00",
    }

    await notify_work_changes(
        FakeClient(channel),
        FakeCore({}),
        DiscordBotConfig(token="x", channel_id=123, core_url="http://127.0.0.1:8765", prefix="!"),
        proposal_view_factory=None,
        work_view_factory=None,
        activation_view_factory=None,
        retry_view_factory=None,
        promote_view_factory=None,
        items=[item],
        seen={},
        first_poll=True,
    )

    assert channel.sent == []


class FakeCore:
    def __init__(self, detail):
        self.detail = detail

    def get(self, path):
        return self.detail


class FakeClient:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, channel_id):
        return self.channel


class FakeChannel:
    def __init__(self):
        self.sent = []

    async def send(self, content, view=None):
        self.sent.append({"content": content, "view": view})
