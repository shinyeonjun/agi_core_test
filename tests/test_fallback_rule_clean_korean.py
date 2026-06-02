from agent.language.fallback_rule import FallbackRuleLanguageEngine, classify_user_goal_kind_rule, detect_style_feedback_rule


def test_clean_korean_code_change_tokens():
    assert classify_user_goal_kind_rule("코드 버그 고쳐줘") == "code_change"
    assert classify_user_goal_kind_rule("리팩토링하고 pytest 돌려줘") == "code_change"


def test_clean_korean_style_feedback_tokens():
    assert detect_style_feedback_rule("이 말투 좋음. 계속 기억해둬")["feedback_type"] == "positive_style"  # type: ignore[index]
    assert detect_style_feedback_rule("너무 AI같음. 더 담백하게")["feedback_type"] == "too_ai_like"  # type: ignore[index]


def test_clean_korean_capability_target():
    result = FallbackRuleLanguageEngine().interpret_user_message("너 뭐 할 수 있어?")
    assert result.target == "capabilities"
    assert result.execution["requires_action"] is False
