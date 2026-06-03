from pathlib import Path
from types import SimpleNamespace

from agent.core.autonomy import set_autonomy_profile
from agent.core.database import init_db


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "repo"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    (tmp_path / "repo").mkdir()
    init_db()


def test_self_improvement_prompt_boundaries_do_not_trigger_secret_policy(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_BACKEND", "native_loop")
    monkeypatch.setenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", "python -m pytest -q")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)
    (tmp_path / "repo" / ".git").mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args[:3] == ["git", "worktree", "add"]:
            Path(args[-2]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M agent/core/example.py\n", stderr="")
        if args[:4] == ["python", "-m", "pytest", "-q"]:
            return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")
        assert args[:2] == ["codex", "exec"]
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("self improvement completed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)

    from agent.lab.codex_worker import run_codex_work

    prompt = "\n".join(
        [
            "Core Native Work Loop",
            "Hard boundaries:",
            "- Do not read or modify .env, tokens, credentials, SSH keys, systemd, apt, or OS settings.",
        ]
    )
    result = run_codex_work(prompt, goal_id=1, task_id=2, self_improvement=True)

    assert result["status"] == "codex_work_completed"
    assert result.get("reason") != "secret_access_denied"
