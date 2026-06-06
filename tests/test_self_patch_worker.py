import subprocess
from pathlib import Path

from neurokernel_seed.harness.self_patch import CodexSelfPatchWorker, SelfPatchConfig


def test_self_patch_worker_creates_patch_artifact(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath = ['src']\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    runner = FakeSelfPatchRunner()
    worker = CodexSelfPatchWorker(
        SelfPatchConfig(
            project_root=project,
            run_root=tmp_path / "runs",
            codex_bin="codex",
            test_command=("python", "-m", "pytest", "-q"),
        ),
        runner=runner,
    )

    result = worker.run(
        job_id="job_self_patch_1",
        work={
            "work_id": "work_cpu_usage",
            "type": "self_patch",
            "title": "CPU usage",
            "goal": "Add a read-only CPU usage action",
            "metadata_json": {"action_id": "get_cpu_usage"},
        },
        payload={"job_id": "job_self_patch_1", "work_id": "work_cpu_usage"},
    )

    assert result["status"] == "patch_ready"
    assert result["patch_bytes"] > 0
    assert "src/demo.py" in result["changed_files"]
    assert Path(result["patch_path"]).exists()
    assert Path(result["run_dir"], "summary.json").exists()
    assert Path(result["run_dir"], "contract.json").exists()
    assert Path(result["run_dir"], "evidence.json").exists()
    assert Path(result["run_dir"], "summary.md").exists()


def test_self_patch_worker_uses_git_worktree_for_git_repo(tmp_path):
    project = _make_git_project(tmp_path)
    runner = FakeSelfPatchRunner()
    worker = CodexSelfPatchWorker(
        SelfPatchConfig(
            project_root=project,
            run_root=tmp_path / "runs",
            codex_bin="codex",
            test_command=("python", "-m", "pytest", "-q"),
            isolation_mode="auto",
        ),
        runner=runner,
    )

    result = worker.run(
        job_id="job_worktree",
        work={"work_id": "work_cpu_usage", "type": "self_patch", "title": "CPU usage", "goal": "Add CPU usage"},
        payload={"job_id": "job_worktree", "work_id": "work_cpu_usage"},
    )

    assert result["status"] == "patch_ready"
    assert result["isolation"]["mode"] == "worktree"
    assert "src/demo.py" in result["changed_files"]
    assert (Path(result["workspace"]) / ".git").exists()


def test_self_patch_worker_blocks_diff_check_failure(tmp_path):
    project = _make_git_project(tmp_path)
    runner = FakeSelfPatchRunner(trailing_whitespace=True)
    worker = CodexSelfPatchWorker(
        SelfPatchConfig(
            project_root=project,
            run_root=tmp_path / "runs",
            codex_bin="codex",
            test_command=("python", "-m", "pytest", "-q"),
            isolation_mode="worktree",
        ),
        runner=runner,
    )

    result = worker.run(
        job_id="job_bad_ws",
        work={"work_id": "work_cpu_usage", "type": "self_patch", "title": "CPU usage", "goal": "Add CPU usage"},
        payload={"job_id": "job_bad_ws", "work_id": "work_cpu_usage"},
    )

    assert result["status"] == "diff_check_failed"
    assert result["diff_check"]["returncode"] != 0
    assert result["next_required_action"] == "worker_review_patch_format_errors"


class FakeSelfPatchRunner:
    def __init__(self, *, trailing_whitespace: bool = False):
        self.trailing_whitespace = trailing_whitespace

    def __call__(self, cmd, *, cwd, input_text=None, timeout_seconds=30):
        if cmd[:2] == ["git", "diff"]:
            return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
        if cmd[0] == "git":
            return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
        if "codex" in cmd:
            target = Path(cwd) / "src" / "demo.py"
            target.write_text("VALUE = 2  \n" if self.trailing_whitespace else "VALUE = 2\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="patched", stderr="")
        if cmd[:3] == ["python", "-m", "pytest"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")
        return subprocess.CompletedProcess(cmd, 99, stdout="", stderr="unexpected command")


def _make_git_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\npythonpath = ['src']\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test User")
    _git(project, "add", "-A")
    _git(project, "commit", "-m", "baseline")
    return project


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=True)
