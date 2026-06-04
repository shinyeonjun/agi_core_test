from __future__ import annotations

import re

PLAIN_LANGUAGE_DIRECTIVE = """Plain-language glossary for user-facing Korean replies:
- Say "답변 품질" or "답변 엔진" instead of "렌더러/renderer".
- Say "복구 답변" instead of "fallback".
- Say "필요할 때 반응하는 루프" instead of "reactor/리액터/wake signal".
- Say "반영 전 안전 검사" instead of "release gate/릴리즈 게이트".
- Say "격리 작업공간" instead of "worktree".
- Say "안전 점검", "자체 평가", and "테스트" instead of audit/eval/pytest when the user did not ask for command-level detail.
- Keep code filenames, commit hashes, and exact commands only when the user asks for technical evidence.
"""

_TERM_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"대화 답변 엔진 성공률"), "답변 성공률"),
    (re.compile(r"답변 엔진 성공률"), "답변 성공률"),
    (re.compile(r"fallback\s+renderer", re.IGNORECASE), "복구 답변"),
    (re.compile(r"renderer\s+success\s+rate", re.IGNORECASE), "답변 성공률"),
    (re.compile(r"(?<![A-Za-z])fallback(?![A-Za-z])", re.IGNORECASE), "복구 답변"),
    (re.compile(r"(?<![A-Za-z])renderer(?![A-Za-z])", re.IGNORECASE), "답변 엔진"),
    (re.compile(r"렌더러"), "답변 엔진"),
    (re.compile(r"(?<![A-Za-z])reactor(?![A-Za-z])", re.IGNORECASE), "반응 루프"),
    (re.compile(r"리액터"), "반응 루프"),
    (re.compile(r"wake\s+signal", re.IGNORECASE), "깨우는 신호"),
    (re.compile(r"release\s+gate", re.IGNORECASE), "반영 전 안전 검사"),
    (re.compile(r"릴리즈\s*게이트"), "반영 전 안전 검사"),
    (re.compile(r"\bworktree\b", re.IGNORECASE), "격리 작업공간"),
    (re.compile(r"\bCodex\s+worker\b", re.IGNORECASE), "코드 작업자"),
    (re.compile(r"\bworker\b", re.IGNORECASE), "작업자"),
    (re.compile(r"\bpytest\b", re.IGNORECASE), "테스트"),
    (re.compile(r"\baudit\b", re.IGNORECASE), "안전 점검"),
    (re.compile(r"\beval\b", re.IGNORECASE), "자체 평가"),
    (re.compile(r"language, memory, goal, policy, 답변 엔진, scheduler", re.IGNORECASE), "대화 해석, 기억, 목표, 정책, 답변 품질, 작업 흐름"),
    (re.compile(r"language, memory", re.IGNORECASE), "대화 해석, 기억"),
)

_PUBLIC_JARGON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![A-Za-z])fallback(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])renderer(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z])reactor(?![A-Za-z])", re.IGNORECASE),
    re.compile(r"wake\s+signal", re.IGNORECASE),
    re.compile(r"release\s+gate", re.IGNORECASE),
    re.compile(r"\bworktree\b", re.IGNORECASE),
    re.compile(r"렌더러"),
    re.compile(r"리액터"),
    re.compile(r"릴리즈\s*게이트"),
)


def humanize_public_text(text: str) -> str:
    result = str(text)
    # Some replacements expose a second plain-language phrase after the first pass
    # such as "renderer" -> "답변 엔진" -> "답변" in metric labels.
    for _ in range(2):
        previous = result
        for pattern, replacement in _TERM_REPLACEMENTS:
            result = pattern.sub(replacement, result)
        if result == previous:
            break
    return result


def public_jargon_violations(text: str) -> list[str]:
    found: list[str] = []
    for pattern in _PUBLIC_JARGON_PATTERNS:
        if pattern.search(text):
            found.append(pattern.pattern)
    return sorted(set(found))
