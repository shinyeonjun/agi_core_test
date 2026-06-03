import json
from types import SimpleNamespace

from agent.cli.agentctl import main
from agent.core.dependency_doctor import dependency_doctor, install_missing_python_dependencies, project_python_requirements


def test_project_python_requirements_reads_pyproject_and_dev_tools():
    requirements = project_python_requirements()

    names = {item["distribution"] for item in requirements}
    assert "discord.py" in names
    assert "pytest" in names


def test_dependency_doctor_reports_current_python_venv_status():
    result = dependency_doctor()

    assert result["scope"] == "python_venv"
    assert result["requires_approval"] is False
    assert "apt/system package install" in result["blocked_actions"]
    assert all("requirement" in item for item in result["checks"])


def test_dependency_doctor_install_uses_python_pip_only(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(
        "agent.core.dependency_doctor.check_python_dependencies",
        lambda root=None, include_dev=True: [
            {"requirement": "missing-demo>=1", "distribution": "missing-demo", "installed": False, "version": None, "source": "dev"}
        ]
        if not calls.get("after")
        else [
            {"requirement": "missing-demo>=1", "distribution": "missing-demo", "installed": True, "version": "1.0", "source": "dev"}
        ],
    )

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["cwd"] = kwargs["cwd"]
        calls["after"] = True
        return SimpleNamespace(returncode=0, stdout="installed", stderr="")

    monkeypatch.setattr("agent.core.dependency_doctor.subprocess.run", fake_run)

    result = install_missing_python_dependencies(root=tmp_path)

    assert result["installed"] is True
    assert calls["args"][1:4] == ["-m", "pip", "install"]
    assert "missing-demo>=1" in calls["args"]


def test_deps_doctor_cli(capsys):
    code = main(["deps", "doctor"])
    output = json.loads(capsys.readouterr().out)

    assert code in {0, 1}
    assert output["scope"] == "python_venv"
    assert "checks" in output
