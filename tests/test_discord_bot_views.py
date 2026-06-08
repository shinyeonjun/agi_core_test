import asyncio

from neurokernel_seed.discord_bot.views import build_view_factories


class FakeButtonStyle:
    success = "success"
    secondary = "secondary"
    danger = "danger"
    primary = "primary"


class FakeView:
    def __init__(self, timeout=None):
        self.timeout = timeout


class FakeUi:
    View = FakeView

    @staticmethod
    def button(**kwargs):
        def decorator(func):
            func.__discord_button__ = kwargs
            return func

        return decorator


class FakeDiscord:
    ui = FakeUi
    ButtonStyle = FakeButtonStyle


class FakeResponse:
    def __init__(self, events):
        self.events = events
        self._done = False

    def is_done(self):
        return self._done

    async def defer(self):
        self.events.append("defer")
        self._done = True

    async def send_message(self, content, *, ephemeral=False):
        self.events.append(("response_send", content, ephemeral))
        self._done = True


class FakeFollowup:
    def __init__(self, events):
        self.events = events

    async def send(self, content, *, ephemeral=False):
        self.events.append(("followup_send", content, ephemeral))


class FakeInteraction:
    def __init__(self, events):
        self.events = events
        self.user = type("User", (), {"id": 1})()
        self.response = FakeResponse(events)
        self.followup = FakeFollowup(events)

    async def edit_original_response(self, content, *, view=None):
        self.events.append(("edit_original", content, view))


def _factories(core):
    return build_view_factories(
        discord=FakeDiscord,
        core=core,
        allowed_user_ids=(),
        activation_verify_note=lambda payload: "",
    )


def test_work_button_acknowledges_before_core_transition():
    events = []

    class Core:
        def post(self, path, payload):
            events.append(("post", path, payload))
            return {"work_item": {"title": "테스트 작업"}}

    view = _factories(Core()).work("work1")
    interaction = FakeInteraction(events)

    asyncio.run(view.accept_work(interaction, None))

    assert events[0] == "defer"
    assert events[1][0] == "post"
    assert events[2][0] == "edit_original"


def test_work_button_reports_core_error_after_ack_with_followup():
    events = []

    class Core:
        def post(self, path, payload):
            events.append(("post", path, payload))
            raise RuntimeError("core offline")

    view = _factories(Core()).work("work1")
    interaction = FakeInteraction(events)

    asyncio.run(view.accept_work(interaction, None))

    assert events[0] == "defer"
    assert events[1][0] == "post"
    assert events[2][0] == "followup_send"
    assert "core offline" in events[2][1]
