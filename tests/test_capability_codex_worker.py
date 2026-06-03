import json
from pathlib import Path
from types import SimpleNamespace

from agent.cli.agentctl import main
from agent.core.autonomy import set_autonomy_profile
from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.goals import create_goal, list_goals
from agent.core.task_queue import enqueue_task, list_tasks
from agent.lab.planner import run_user_task


def setup_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_CORE_HOME", str(tmp_path / "repo"))
    monkeypatch.setenv("AGENT_CORE_DB_PATH", str(tmp_path / "agent.db"))
    monkeypatch.setenv("AGENT_CORE_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("AGENT_CORE_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("AGENT_LANGUAGE_ENGINE", "rule")
    monkeypatch.setenv("AGENT_CHAT_RENDERER", "fallback")
    (tmp_path / "repo").mkdir()
    init_db()


def test_capability_map_reaches_decision(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)

    decision = build_talk_decision("너 뭐 할 수 있어?")

    assert decision["capability_map"]["direct"]
    assert any(item["name"] == "codex_work_worker" for item in decision["capability_map"]["worker_mediated"])


def test_capability_cli(monkeypatch, tmp_path, capsys):
    setup_isolated(monkeypatch, tmp_path)

    assert main(["capability", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["profile"]
    assert data["worker_mediated"]


def test_code_change_user_task_runs_codex_worker(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_TIMEOUT", "9")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)
    (tmp_path / "repo" / ".git").mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M agent/core/example.py\n", stderr="")
        assert args[:2] == ["codex", "exec"]
        assert "--output-last-message" in args
        assert "--ephemeral" in args
        assert args[args.index("--sandbox") + 1] == "workspace-write"
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("수정 완료. 테스트는 생략했어.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)
    goal_id = create_goal(
        "Fix code",
        "코드 버그 수정해줘",
        goal_type="user_directed",
        status="active",
        priority=0.98,
        metadata={"priority_owner": "user", "task_kind": "code_change", "raw_user_text": "코드 버그 수정해줘"},
        dedupe=False,
    )
    task_id = enqueue_task("user", goal_id=goal_id, task_kind="code_change", title="Fix code", source="test", priority=0.98)

    result = run_user_task(task_id)
    task = next(row for row in list_tasks(limit=5, queue_type="user") if row["id"] == task_id)
    goal = next(row for row in list_goals(limit=5, include_archived=True) if row["id"] == goal_id)

    assert result["status"] == "codex_work_completed"
    assert result["artifact_type"] == "codex_work_report"
    assert result["task_id"] == task_id
    assert task["status"] == "done"
    assert goal["status"] == "done"


def test_codex_worker_blocks_safe_profile(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("safe")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("코드 고쳐줘", goal_id=1, task_id=2)

    assert result["status"] == "codex_work_blocked"
    assert "profile_not_full_device_lab" in result["blockers"]
    assert result["executed"] is False


def test_codex_worker_blocks_invalid_sandbox(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_SANDBOX", "danger-full-access")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("코드 고쳐줘", goal_id=1, task_id=2)

    assert result["status"] == "codex_work_blocked"
    assert result["reason"] == "codex_worker_not_available"
    assert "invalid_codex_work_sandbox" in result["blockers"]


def test_self_improvement_requires_native_loop_backend(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_BACKEND", "codex")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("Core 자가개선 코드를 만들어줘", goal_id=1, task_id=2, self_improvement=True)

    assert result["status"] == "codex_work_blocked"
    assert result["reason"] == "self_improvement_requires_native_loop"
    assert result["backend"] == "codex"
    assert result["executed"] is False


def test_codex_worker_marks_unsafe_changed_files_blocked(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)
    (tmp_path / "repo" / ".git").mkdir()
    calls = {"status": 0}

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "status"]:
            calls["status"] += 1
            stdout = "" if calls["status"] == 1 else " M .env\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("수정 완료.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("코드 고쳐줘", goal_id=1, task_id=2)

    assert result["status"] == "codex_work_blocked"
    assert result["executed"] is False
    assert result["unsafe_changed_files"] == [".env"]


def test_native_loop_backend_uses_worktree_and_verification(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_BACKEND", "native_loop")
    monkeypatch.setenv("AGENT_WORK_LOOP_TIMEOUT", "300")
    monkeypatch.setenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", "python -m pytest -q")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)
    (tmp_path / "repo" / ".git").mkdir()
    calls = {"codex": None, "worktree": None, "verify": None}

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args[:3] == ["git", "worktree", "add"]:
            path = args[-2]
            calls["worktree"] = path
            Path(path).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M agent/core/example.py\n", stderr="")
        if args[:4] == ["python", "-m", "pytest", "-q"]:
            calls["verify"] = {"args": args, "cwd": kwargs["cwd"], "timeout": kwargs["timeout"]}
            return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")
        assert args[:2] == ["codex", "exec"]
        calls["codex"] = {"args": args, "cwd": kwargs["cwd"], "timeout": kwargs["timeout"]}
        assert kwargs["timeout"] == 300
        assert "Core Native Work Loop" in args[-1]
        assert "Autonomous study/self-improvement loops are separate" in args[-1]
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("네이티브 작업 루프 완료. 테스트 통과.")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("Core worker backend 개선해줘", goal_id=1, task_id=2)

    assert result["status"] == "codex_work_completed"
    assert result["backend"] == "native_loop"
    assert result["mode"] == "native_loop"
    assert result["worktree_branch"].startswith("codex/native-loop-task-2-")
    assert calls["codex"]["cwd"] == Path(result["worktree"])
    assert calls["verify"]["cwd"] == Path(result["worktree"])
    assert calls["worktree"] == result["worktree"]
    assert result["changed_files"] == [" M agent/core/example.py"]
    assert result["iterations_used"] == 1
    assert result["verification_commands"] == ["python -m pytest -q"]
    assert result["integration_status"] == "worktree_pending_review"
    assert result["evidence_ledger"][0]["verification"][0]["returncode"] == 0


def test_native_loop_retries_after_failed_verification(monkeypatch, tmp_path):
    setup_isolated(monkeypatch, tmp_path)
    set_autonomy_profile("full_device_lab")
    monkeypatch.setenv("AGENT_CODEX_WORKER_ENABLED", "1")
    monkeypatch.setenv("AGENT_CODEX_WORK_BACKEND", "native_loop")
    monkeypatch.setenv("AGENT_WORK_LOOP_ITERATIONS", "2")
    monkeypatch.setenv("AGENT_WORK_LOOP_VERIFY_COMMANDS", "python -m pytest -q")
    monkeypatch.setattr("agent.core.capabilities.shutil.which", lambda name: "codex" if name == "codex" else None)
    (tmp_path / "repo" / ".git").mkdir()
    calls = {"codex": 0, "verify": 0}

    def fake_run(args, **kwargs):
        if args[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args[:3] == ["git", "worktree", "add"]:
            Path(args[-2]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args[:2] == ["git", "status"]:
            return SimpleNamespace(returncode=0, stdout=" M agent/core/example.py\n", stderr="")
        if args[:4] == ["python", "-m", "pytest", "-q"]:
            calls["verify"] += 1
            if calls["verify"] == 1:
                return SimpleNamespace(returncode=1, stdout="", stderr="failed")
            return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")
        assert args[:2] == ["codex", "exec"]
        calls["codex"] += 1
        if calls["codex"] == 2:
            assert '"returncode": 1' in args[-1]
        output_path = args[args.index("--output-last-message") + 1]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(f"iteration {calls['codex']} report")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("agent.lab.codex_worker.subprocess.run", fake_run)

    from agent.lab.codex_worker import run_codex_work

    result = run_codex_work("테스트 실패까지 고쳐줘", goal_id=1, task_id=2)

    assert result["status"] == "codex_work_completed"
    assert calls["codex"] == 2
    assert calls["verify"] == 2
    assert result["iterations_used"] == 2
    assert result["evidence_ledger"][0]["verification"][0]["returncode"] == 1
    assert result["evidence_ledger"][1]["verification"][0]["returncode"] == 0
