from __future__ import annotations

from pathlib import Path
from typing import Any

from .trace import redact_text


RETRY_PLAN_SCHEMA_VERSION = "neurokernel-self-patch-retry-plan-v1"
MAX_PATCH_EXCERPT_CHARS = 12_000


def build_retry_plan(payload: dict[str, Any]) -> dict[str, Any] | None:
    retry = payload.get("retry") if isinstance(payload.get("retry"), dict) else None
    if not retry:
        return None
    previous_result = retry.get("previous_result") if isinstance(retry.get("previous_result"), dict) else {}
    failure_analysis = previous_result.get("failure_analysis") if isinstance(previous_result.get("failure_analysis"), dict) else {}
    test_tail = previous_result.get("test") if isinstance(previous_result.get("test"), dict) else {}
    patch_path = str(previous_result.get("patch_path") or "").strip()
    plan = {
        "schema_version": RETRY_PLAN_SCHEMA_VERSION,
        "requested_by": retry.get("requested_by"),
        "previous_status": retry.get("previous_status"),
        "previous_job_id": previous_result.get("job_id"),
        "previous_patch_path": patch_path or None,
        "previous_result_status": previous_result.get("status"),
        "primary_failure": failure_analysis.get("primary_failure"),
        "failure_summary": failure_analysis.get("summary"),
        "next_step": failure_analysis.get("next_step"),
        "changed_files": _string_list(previous_result.get("changed_files")),
        "failing_tests": _extract_failing_tests(str(test_tail.get("stdout_tail") or "")),
        "test_stdout_tail": str(test_tail.get("stdout_tail") or "")[-4_000:],
        "test_stderr_tail": str(test_tail.get("stderr_tail") or "")[-4_000:],
        "previous_patch_excerpt": _patch_excerpt(patch_path),
        "retry_directives": _retry_directives(failure_analysis),
    }
    return plan


def retry_plan_markdown(plan: dict[str, Any] | None) -> str:
    if not plan:
        return "No retry context. This is a first implementation attempt."
    lines = [
        "# Retry Plan",
        "",
        f"- schema: `{plan.get('schema_version')}`",
        f"- previous_job_id: `{plan.get('previous_job_id')}`",
        f"- previous_status: `{plan.get('previous_result_status')}`",
        f"- primary_failure: `{plan.get('primary_failure')}`",
        f"- next_step: `{plan.get('next_step')}`",
        "",
        "## Required Retry Behavior",
    ]
    lines.extend(f"- {item}" for item in _string_list(plan.get("retry_directives")))
    lines.extend(["", "## Changed Files From Failed Attempt"])
    changed = _string_list(plan.get("changed_files"))
    lines.extend(f"- `{item}`" for item in changed) if changed else lines.append("- none")
    lines.extend(["", "## Failing Tests"])
    failing = _string_list(plan.get("failing_tests"))
    lines.extend(f"- `{item}`" for item in failing) if failing else lines.append("- no parsed failing test names; inspect test tails below")
    lines.extend(
        [
            "",
            "## Previous Test Tail",
            "```text",
            str(plan.get("test_stdout_tail") or "")[-4_000:],
            str(plan.get("test_stderr_tail") or "")[-2_000:],
            "```",
            "",
            "## Previous Patch Excerpt",
            "```diff",
            str(plan.get("previous_patch_excerpt") or ""),
            "```",
        ]
    )
    return "\n".join(lines)


def _retry_directives(failure_analysis: dict[str, Any]) -> list[str]:
    primary = str(failure_analysis.get("primary_failure") or "unknown")
    directives = [
        "Start by reading this retry plan and the failed changed files before editing.",
        "Fix the failing contract instead of repeating the same patch shape.",
        "Do not weaken, delete, or skip existing tests unless the approved work item explicitly changes that contract.",
        "Run the failing tests first, then run the full configured test command before stopping.",
        "Keep the retry patch smaller than the failed attempt unless the failure proves a missing dependency boundary.",
    ]
    if primary == "test_failed":
        directives.append("The previous patch compiled enough to run tests; prioritize the failing assertions over broad redesign.")
    elif primary == "diff_check_failed":
        directives.append("Fix whitespace/format issues first, then rerun tests.")
    elif primary in {"codex_timeout_or_stall", "test_timeout_or_stall"}:
        directives.append("Reduce scope and add a narrow regression test before broader changes.")
    elif primary == "codex_failed_no_patch":
        directives.append("Inspect the previous Codex tail and produce a minimal patch; do not stop at analysis.")
    return directives


def _extract_failing_tests(text: str) -> list[str]:
    tests: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("FAILED "):
            parts = line.split()
            if len(parts) >= 2:
                tests.append(parts[1])
        elif line.startswith("________") and line.endswith("________"):
            label = line.strip("_ ").strip()
            if label:
                tests.append(label)
    return list(dict.fromkeys(tests))


def _patch_excerpt(patch_path: str) -> str:
    if not patch_path:
        return ""
    path = Path(patch_path)
    if not path.exists() or not path.is_file():
        return ""
    text = redact_text(path.read_text(encoding="utf-8", errors="replace"), max_chars=MAX_PATCH_EXCERPT_CHARS)
    if len(text) <= MAX_PATCH_EXCERPT_CHARS:
        return text
    head = text[: MAX_PATCH_EXCERPT_CHARS // 2]
    tail = text[-MAX_PATCH_EXCERPT_CHARS // 2 :]
    return f"{head}\n\n[... previous patch excerpt truncated ...]\n\n{tail}"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
