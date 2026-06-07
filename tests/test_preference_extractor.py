import asyncio

from neurokernel_seed.discord_bot.bot import _maybe_save_conversational_preferences
from neurokernel_seed.memory.preference_extractor import extract_preference_candidates, preference_candidates_from_intent


def test_raw_text_does_not_create_preference_candidates():
    assert extract_preference_candidates("앞으로 답변 짧게 해줘") == []


def test_structured_preference_intent_creates_candidates():
    intent = {
        "kind": "preference_update",
        "reply": "알겠어. 앞으로 짧게 말할게.",
        "candidates": [
            {
                "key": "response_length",
                "value": "short",
                "scope": "global",
                "source": "explicit_user_request",
                "confidence": 0.95,
                "evidence": "앞으로 답변 짧게 해줘",
            }
        ],
        "confidence": 0.95,
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }

    candidates = preference_candidates_from_intent(intent)

    assert [(item.key, item.value, item.scope) for item in candidates] == [("response_length", "short", "global")]


def test_structured_preference_intent_rejects_unknown_key():
    intent = {
        "kind": "preference_update",
        "reply": "저장할게.",
        "candidates": [
            {
                "key": "private_token",
                "value": "abc",
                "scope": "global",
                "source": "explicit_user_request",
                "confidence": 0.95,
                "evidence": "내 토큰 기억해",
            }
        ],
        "confidence": 0.95,
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }

    try:
        preference_candidates_from_intent(intent)
    except Exception as exc:
        assert "unknown preference key" in str(exc)
    else:
        raise AssertionError("unknown preference key should be rejected")


def test_conversational_preference_save_uses_core_api():
    class FakeCore:
        def __init__(self):
            self.posts = []

        def post(self, path, payload):
            self.posts.append((path, payload))
            if path == "/language/preferences":
                return {
                    "kind": "preference_update",
                    "reply": "알겠어. 앞으로 답변은 짧게 할게.",
                    "candidates": [
                        {
                            "key": "response_length",
                            "value": "short",
                            "scope": "global",
                            "source": "explicit_user_request",
                            "confidence": 0.95,
                            "evidence": "앞으로 답변 짧게 해줘",
                        }
                    ],
                    "confidence": 0.95,
                    "requires_confirmation": False,
                    "clarifying_question": None,
                    "safety_notes": [],
                }
            return {"saved": True}

    core = FakeCore()
    reply = asyncio.run(_maybe_save_conversational_preferences("앞으로 답변 짧게 해줘", core, user_id="discord:1"))

    assert reply == "알겠어. 앞으로 답변은 짧게 할게."
    assert core.posts == [
        (
            "/language/preferences",
            {"user_text": "앞으로 답변 짧게 해줘", "context": {"source": "discord", "user_id": "discord:1"}},
        ),
        (
            "/memory/preferences",
            {
                "user_id": "discord:1",
                "key": "response_length",
                "value": "short",
                "scope": "global",
                "source": "explicit_user_request",
                "confidence": 0.95,
            },
        ),
    ]
