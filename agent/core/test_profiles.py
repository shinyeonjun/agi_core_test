from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


FAST_TESTS = [
    "tests/test_smoke.py",
    "tests/test_policy.py",
    "tests/test_failure_taxonomy.py",
    "tests/test_fallback_rule_clean_korean.py",
    "tests/test_renderer.py",
    "tests/test_codex_config.py",
    "tests/test_dependency_doctor.py",
]

CHAT_TESTS = [
    "tests/test_bridge.py",
    "tests/test_bridge_codex_work_formatter.py",
    "tests/test_language_engine.py",
    "tests/test_discord_control_plane.py",
]

INTEGRATION_TESTS = [
    "tests/test_capability_codex_worker.py",
    "tests/test_codex_worker_policy.py",
    "tests/test_cognitive_engine.py",
    "tests/test_cognitive_pipeline.py",
    "tests/test_core_pipeline.py",
    "tests/test_dashboard.py",
    "tests/test_db_migrations.py",
    "tests/test_eval_tools.py",
    "tests/test_full_device.py",
    "tests/test_goal_generator.py",
    "tests/test_lab_loop.py",
    "tests/test_learner.py",
    "tests/test_memory_intelligence.py",
    "tests/test_metrics.py",
    "tests/test_observability.py",
    "tests/test_observation_mode.py",
    "tests/test_operating_intelligence.py",
    "tests/test_project_execution_loop.py",
    "tests/test_reactor.py",
    "tests/test_research_ingestion.py",
    "tests/test_runtime_isolation.py",
    "tests/test_self_code_review.py",
    "tests/test_self_improvement_planner.py",
    "tests/test_self_improvement_release.py",
    "tests/test_self_map.py",
    "tests/test_sparse_vectors.py",
    "tests/test_style_learning.py",
    "tests/test_task_queue_split.py",
    "tests/test_workspace.py",
]


@dataclass(frozen=True)
class TestProfile:
    name: str
    purpose: str
    pytest_args: tuple[str, ...]
    checks: tuple[str, ...] = ()
    expected: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "pytest_args": list(self.pytest_args),
            "checks": list(self.checks),
            "expected": self.expected,
        }


PROFILES: dict[str, TestProfile] = {
    "smoke": TestProfile(
        name="smoke",
        purpose="Small liveness checks for Core boot, DB, memory, talk, and tick.",
        pytest_args=("tests/test_smoke.py", "-q"),
        expected="seconds on Orange Pi",
    ),
    "fast": TestProfile(
        name="fast",
        purpose="Default quick checks for Core liveness, policy, renderer, dependency, and failure taxonomy.",
        pytest_args=(*FAST_TESTS, "-q"),
        expected="about 20-40 seconds on Orange Pi",
    ),
    "chat": TestProfile(
        name="chat",
        purpose="Chat and Discord checks for bridge, language interpretation, command UX, approvals, and notifications.",
        pytest_args=(*CHAT_TESTS, "-q"),
        expected="focused chat/Discord checks",
    ),
    "integration": TestProfile(
        name="integration",
        purpose="Broader cross-component checks for queues, workers, reactor, DB, and self-improvement.",
        pytest_args=(*INTEGRATION_TESTS, "-q"),
        expected="several minutes on Orange Pi",
    ),
    "full": TestProfile(
        name="full",
        purpose="Complete pytest suite for major merges or release validation.",
        pytest_args=("-q",),
        expected="slow on Orange Pi",
    ),
    "release": TestProfile(
        name="release",
        purpose="Practical pre-apply gate: fast pytest, then audit and eval.",
        pytest_args=(*FAST_TESTS, "-q"),
        checks=("audit", "eval"),
        expected="a little slower than fast",
    ),
}
PROFILE_NAMES = tuple(PROFILES.keys())


def list_test_profiles() -> list[dict[str, Any]]:
    return [profile.to_dict() for profile in PROFILES.values()]


def get_test_profile(name: str) -> TestProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        available = ", ".join(PROFILES)
        raise ValueError(f"unknown test profile: {name}. available={available}") from exc


def _command_preview(args: list[str]) -> str:
    return " ".join(args)


def plan_test_profile(name: str) -> dict[str, Any]:
    profile = get_test_profile(name)
    steps = [{"type": "pytest", "command": _command_preview([sys.executable, "-m", "pytest", *profile.pytest_args])}]
    for check in profile.checks:
        if check == "audit":
            steps.append({"type": "audit", "command": _command_preview([sys.executable, "-m", "agent.cli.agentctl", "audit"])})
        elif check == "eval":
            steps.append({"type": "eval", "command": _command_preview([sys.executable, "-m", "agent.cli.agentctl", "eval", "run"])})
    return {**profile.to_dict(), "steps": steps}


def _run_command(args: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(args, text=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=timeout, check=False)
        return {
            "command": _command_preview(args),
            "returncode": completed.returncode,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "stdout": (completed.stdout or "")[-4000:],
            "stderr": (completed.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": _command_preview(args),
            "returncode": 124,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": exc.stderr if isinstance(exc.stderr, str) else "timeout",
        }


def run_test_profile(name: str, *, extra_pytest_args: list[str] | None = None, timeout: int | None = None) -> dict[str, Any]:
    profile = get_test_profile(name)
    extra = tuple(extra_pytest_args or [])
    started = time.monotonic()
    steps: list[dict[str, Any]] = []
    pytest_result = _run_command([sys.executable, "-m", "pytest", *profile.pytest_args, *extra], timeout=timeout)
    pytest_result["type"] = "pytest"
    steps.append(pytest_result)
    if pytest_result["returncode"] != 0:
        return {"profile": profile.name, "ok": False, "failed_step": "pytest", "duration_ms": int((time.monotonic() - started) * 1000), "steps": steps}

    for check in profile.checks:
        if check == "audit":
            command = [sys.executable, "-m", "agent.cli.agentctl", "audit"]
        elif check == "eval":
            command = [sys.executable, "-m", "agent.cli.agentctl", "eval", "run"]
        else:
            continue
        result = _run_command(command, timeout=timeout)
        result["type"] = check
        steps.append(result)
        if result["returncode"] != 0:
            return {"profile": profile.name, "ok": False, "failed_step": check, "duration_ms": int((time.monotonic() - started) * 1000), "steps": steps}

    return {"profile": profile.name, "ok": True, "failed_step": None, "duration_ms": int((time.monotonic() - started) * 1000), "steps": steps}


def compact_test_result_lines(result: dict[str, Any]) -> list[str]:
    status = "PASS" if result.get("ok") else "FAIL"
    lines = [f"test profile: {result.get('profile')} - {status}", f"total: {int(result.get('duration_ms') or 0) / 1000:.1f}s"]
    for step in result.get("steps", []):
        step_status = "PASS" if step.get("returncode") == 0 else f"FAIL rc={step.get('returncode')}"
        lines.append(f"- {step.get('type')}: {step_status} / {int(step.get('duration_ms') or 0) / 1000:.1f}s")
    return lines
