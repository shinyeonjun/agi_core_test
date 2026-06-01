from pathlib import Path

import pytest

from agent.cli.agentctl import main
from agent.workspace.executor import create_project_spec, ensure_workspace, safe_workspace_path, write_text_artifact
from agent.workspace.store import list_project_specs, list_workspace_artifacts


def test_workspace_init_creates_expected_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    result = ensure_workspace()
    assert Path(result["root"]).exists()
    for name in ["scratch", "outputs", "tasks", "reports", "projects", "experiments"]:
        assert (tmp_path / "workspace" / name).is_dir()


def test_workspace_write_stays_inside_root(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    artifact = write_text_artifact("scratch", "note.txt", "hello workspace\n", "test", "note")
    assert artifact["relative_path"] == "scratch/note.txt"
    assert Path(artifact["path"]).read_text(encoding="utf-8") == "hello workspace\n"
    assert list_workspace_artifacts(1)[0]["relative_path"] == "scratch/note.txt"


def test_workspace_blocks_path_escape(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    with pytest.raises(ValueError):
        safe_workspace_path("scratch", "../outside.txt")


def test_workspace_blocks_secret_like_content(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    with pytest.raises(ValueError):
        write_text_artifact("scratch", "secret.txt", "API_KEY=abc123\n")


def test_project_spec_records_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    artifact = create_project_spec("Tiny Report", "Create a safe report", ["workspace only"])
    assert artifact["relative_path"].startswith("projects/")
    assert artifact["project_id"] > 0
    assert list_project_specs(1)[0]["title"] == "Tiny Report"


def test_workspace_cli_self_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    assert main(["self-check", "workspace"]) == 0
    output = capsys.readouterr().out
    assert "scratch/" in output
