from agent.core.learner import detect_feedback, list_skills, update_after_turn


def test_detect_feedback(monkeypatch):
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    assert detect_feedback("\uc88b\uc544 \uacc4\uc18d") == "positive"
    assert detect_feedback("\uc544\ub2c8 \ub2e4\uc2dc") == "negative"
    assert detect_feedback("plain neutral") == "neutral"


def test_negative_feedback_updates_skill_and_memory(monkeypatch):
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    result = update_after_turn("\uc544\ub2c8 \ub2e4\uc2dc \ud574\uc918", None, None, {"selected_goal_id": None})
    assert result["feedback"] == "negative"
    skills = list_skills()
    assert any(skill["name"] == "core_talk_pipeline" for skill in skills)
