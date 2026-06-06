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


class FakeSelfPatchRunner:
    def __call__(self, cmd, *, cwd, input_text=None, timeout_seconds=30):
        if cmd[:2] == ["git", "diff"]:
            return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
        if cmd[0] == "git":
            return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=False)
        if "codex" in cmd:
            target = Path(cwd) / "src" / "demo.py"
            target.write_text("VALUE = 2\n", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="patched", stderr="")
        if cmd[:3] == ["python", "-m", "pytest"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="1 passed", stderr="")
        return subprocess.CompletedProcess(cmd, 99, stdout="", stderr="unexpected command")
