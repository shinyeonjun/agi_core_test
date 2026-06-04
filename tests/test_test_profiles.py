import json
from types import SimpleNamespace

from agent.cli.agentctl import main
from agent.core.test_profiles import list_test_profiles, plan_test_profile, run_test_profile


def test_test_profiles_expose_os_like_layers():
    names = [item["name"] for item in list_test_profiles()]

    assert names == ["smoke", "fast", "learning", "chat", "integration", "full", "release"]


def test_fast_profile_keeps_pytest_scope_smaller_than_full():
    fast = plan_test_profile("fast")
    full = plan_test_profile("full")

    assert "tests/test_smoke.py" in fast["steps"][0]["command"]
    assert "tests/test_answer_contract.py" in fast["steps"][0]["command"]
    assert "tests/test_discord_control_plane.py" not in fast["steps"][0]["command"]
    assert "tests/test_task_queue_split.py" not in fast["steps"][0]["command"]
    assert full["steps"][0]["command"].endswith(" -m pytest -q")


def test_learning_profile_keeps_research_checks_focused():
    learning = plan_test_profile("learning")

    assert "tests/test_advanced_learning.py" in learning["steps"][0]["command"]
    assert "tests/test_research_loop.py" in learning["steps"][0]["command"]
    assert "tests/test_discord_control_plane.py" not in learning["steps"][0]["command"]


def test_chat_profile_keeps_control_plane_checks_focused():
    chat = plan_test_profile("chat")

    assert "tests/test_bridge.py" in chat["steps"][0]["command"]
    assert "tests/test_discord_control_plane.py" in chat["steps"][0]["command"]
    assert "tests/test_task_queue_split.py" not in chat["steps"][0]["command"]


def test_release_profile_adds_audit_and_eval_after_fast_tests():
    release = plan_test_profile("release")
    step_types = [step["type"] for step in release["steps"]]

    assert step_types == ["pytest", "audit", "eval"]


def test_run_profile_stops_after_pytest_failure(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout="failed", stderr="")

    monkeypatch.setattr("agent.core.test_profiles.subprocess.run", fake_run)

    result = run_test_profile("release")

    assert result["ok"] is False
    assert result["failed_step"] == "pytest"
    assert len(calls) == 1


def test_run_profile_executes_release_checks_when_pytest_passes(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("agent.core.test_profiles.subprocess.run", fake_run)

    result = run_test_profile("release")

    assert result["ok"] is True
    assert [step["type"] for step in result["steps"]] == ["pytest", "audit", "eval"]
    assert calls[1][-1] == "audit"
    assert calls[2][-2:] == ["eval", "run"]


def test_agentctl_test_plan_cli(capsys):
    code = main(["test", "plan", "fast"])
    data = json.loads(capsys.readouterr().out)

    assert code == 0
    assert data["name"] == "fast"
    assert data["steps"][0]["type"] == "pytest"


def test_learning_profile_includes_self_report_grounding():
    from agent.core.test_profiles import plan_test_profile

    commands = "\n".join(step["command"] for step in plan_test_profile("learning")["steps"])
    assert "tests/test_self_report.py" in commands
