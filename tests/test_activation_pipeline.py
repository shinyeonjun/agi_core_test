import subprocess
from pathlib import Path

import pytest

from neurokernel_seed.harness.activation import ActivationConfig, ActivationError, ActivationService
from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.service import HarnessService


def test_activation_applies_patch_runs_tests_and_marks_proposal_active(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(
        user_text="CPU 사용률 볼 수 있어?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="discord:1")
    patch_path = _make_patch(project, tmp_path, work_id=work_id)
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={"patch_path": str(patch_path)})
        memory.add_work_event(
            work_id,
            "job_completed",
            actor="worker",
            payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": ["src/demo.py"]}},
        )
        memory.conn.commit()

    result = ActivationService(
        ActivationConfig(
            db_path=db_path,
            project_root=project,
            self_patch_run_root=project / "artifacts" / "self_patch",
            test_command=("python", "-c", "from pathlib import Path; assert Path('src/demo.py').read_text().strip() == 'VALUE = 2'"),
            verify_activation=False,
        )
    ).activate_work_item(work_id, actor="test")

    assert result["activated"] is True
    assert result["service_reload_required"] is True
    assert result["pre_activation_sha"] != result["post_activation_sha"]
    assert result["commit"]["returncode"] == 0
    assert Path(result["archive_path"]).is_file()
    assert (project / "src" / "demo.py").read_text(encoding="utf-8").strip() == "VALUE = 2"
    assert _git_output(project, "status", "--porcelain").strip() == ""
    detail = service.work_item(work_id)
    assert detail["work_item"]["status"] == "completed"
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "active"


def test_activation_completes_promoted_external_work_parent(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    with HarnessMemory(db_path) as memory:
        parent = memory.create_work_item(
            work_id="work_parent",
            work_type="external_work",
            title="diagnostic report",
            goal="Add diagnostic report capability",
            status="planned",
            priority="high",
            risk_level="low",
            metadata={},
            actor="test",
        )
        child = memory.create_work_item(
            work_id="work_child",
            work_type="self_patch",
            title="diagnostic report",
            goal="Add diagnostic report capability",
            status="accepted",
            priority="high",
            risk_level="low",
            parent_work_id=parent["work_id"],
            linked_entity_type="promoted_external_work",
            linked_entity_id=parent["work_id"],
            metadata={},
            actor="test",
        )
        patch_path = _make_patch(project, tmp_path, work_id=child["work_id"])
        memory.transition_work_item(child["work_id"], "waiting_approval", actor="test", payload={"patch_path": str(patch_path)})
        memory.add_work_event(
            child["work_id"],
            "job_completed",
            actor="worker",
            payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": ["src/demo.py"]}},
        )
        memory.conn.commit()

    result = ActivationService(
        ActivationConfig(
            db_path=db_path,
            project_root=project,
            self_patch_run_root=project / "artifacts" / "self_patch",
            test_command=("python", "-c", "from pathlib import Path; assert Path('src/demo.py').read_text().strip() == 'VALUE = 2'"),
            verify_activation=False,
        )
    ).activate_work_item("work_child", actor="test")

    assert result["activated"] is True
    with HarnessMemory(db_path) as memory:
        assert memory.get_work_item("work_child")["status"] == "completed"
        assert memory.get_work_item("work_parent")["status"] == "completed"


def test_activation_blocks_dirty_live_repo(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="CPU 사용률", capability_intent=_cpu_usage_intent())
    work_id = proposal["work_item"]["work_id"]
    proposal_id = proposal["proposal"]["proposal_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_patch(project, tmp_path, work_id=work_id)
    (project / "README.md").write_text("dirty\n", encoding="utf-8")
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path)}})
        memory.conn.commit()

    with pytest.raises(ActivationError, match="clean tree"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                verify_activation=False,
            )
        ).activate_work_item(work_id, actor="test")


def test_activation_rolls_back_when_tests_fail(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="CPU 사용률", capability_intent=_cpu_usage_intent())
    work_id = proposal["work_item"]["work_id"]
    proposal_id = proposal["proposal"]["proposal_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_patch(project, tmp_path, work_id=work_id)
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path)}})
        memory.conn.commit()

    with pytest.raises(ActivationError, match="tests failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "raise SystemExit(1)"),
                verify_activation=False,
            )
        ).activate_work_item(work_id, actor="test")

    assert (project / "src" / "demo.py").read_text(encoding="utf-8").strip() == "VALUE = 1"
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"
    assert _git_output(project, "status", "--porcelain").strip() == ""


def test_activation_rolls_back_when_commit_fails(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="CPU 사용률", capability_intent=_cpu_usage_intent())
    work_id = proposal["work_item"]["work_id"]
    proposal_id = proposal["proposal"]["proposal_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_patch(project, tmp_path, work_id=work_id)
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path)}})
        memory.conn.commit()

    with pytest.raises(ActivationError, match="commit failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                verify_activation=False,
            ),
            runner=_runner_that_fails_commit,
        ).activate_work_item(work_id, actor="test")

    assert (project / "src" / "demo.py").read_text(encoding="utf-8").strip() == "VALUE = 1"
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"
    assert _git_output(project, "status", "--porcelain").strip() == ""


def test_activation_verifies_registry_action_before_marking_active(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="테스트 probe 추가", capability_intent=_registry_probe_intent())
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_registry_patch(project)
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path)}})
        memory.conn.commit()

    result = ActivationService(
        ActivationConfig(
            db_path=db_path,
            project_root=project,
            self_patch_run_root=project / "artifacts" / "self_patch",
            test_command=("python", "-c", "pass"),
            action_registry_path=Path("registry/actions.json"),
        )
    ).activate_work_item(work_id, actor="test")

    assert result["verification"]["passed"] is True
    assert result["verification"]["action_id"] == "get_test_probe"
    assert result["verification"]["smoke"]["execution_result"]["result"] == {"ok": True}
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "active"


def test_activation_rolls_back_when_registry_smoke_fails(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="테스트 probe 추가", capability_intent=_registry_probe_intent())
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_registry_patch(project, command=["python", "-c", "raise SystemExit(3)"])
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path)}})
        memory.conn.commit()

    with pytest.raises(ActivationError, match="verification failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                action_registry_path=Path("registry/actions.json"),
            )
        ).activate_work_item(work_id, actor="test")

    assert not (project / "registry" / "actions.json").exists()
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "approved_for_dev"


def test_activation_verifies_registry_v2_output_contract(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="테스트 probe v2 추가", capability_intent=_registry_probe_intent("get_test_probe_v2"))
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_registry_v2_patch(project, action_id="get_test_probe_v2")
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(
            work_id,
            "job_completed",
            actor="worker",
            payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": ["registry/actions.json"]}},
        )
        memory.conn.commit()

    result = ActivationService(
        ActivationConfig(
            db_path=db_path,
            project_root=project,
            self_patch_run_root=project / "artifacts" / "self_patch",
            test_command=("python", "-c", "pass"),
            action_registry_path=Path("registry/actions.json"),
        )
    ).activate_work_item(work_id, actor="test")

    assert result["verification"]["passed"] is True
    assert result["verification"]["schema_check"]["passed"] is True
    assert result["verification"]["output_check"]["passed"] is True
    assert result["verification"]["action_status"] == "active"
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "active"


def test_activation_rolls_back_when_registry_v2_output_contract_fails(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="테스트 probe v2 추가", capability_intent=_registry_probe_intent("get_test_probe_v2"))
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_registry_v2_patch(
        project,
        action_id="get_test_probe_v2",
        command=["python", "-c", "import json; print(json.dumps({'wrong': True}))"],
    )
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(
            work_id,
            "job_completed",
            actor="worker",
            payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": ["registry/actions.json"]}},
        )
        memory.conn.commit()

    with pytest.raises(ActivationError, match="verification failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                action_registry_path=Path("registry/actions.json"),
            )
        ).activate_work_item(work_id, actor="test")

    assert not (project / "registry" / "actions.json").exists()
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "approved_for_dev"


def test_activation_rolls_back_when_registry_v2_output_type_mismatches(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="테스트 probe v2 추가", capability_intent=_registry_probe_intent("get_test_probe_v2"))
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_registry_v2_patch(
        project,
        action_id="get_test_probe_v2",
        command=["python", "-c", "import json; print(json.dumps({'ok': 'true'}))"],
    )
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(
            work_id,
            "job_completed",
            actor="worker",
            payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": ["registry/actions.json"]}},
        )
        memory.conn.commit()

    with pytest.raises(ActivationError, match="verification failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                action_registry_path=Path("registry/actions.json"),
            )
        ).activate_work_item(work_id, actor="test")

    assert not (project / "registry" / "actions.json").exists()
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"
    assert service.capability_proposal(proposal_id)["proposal"]["status"] == "approved_for_dev"


def test_activation_preflight_blocks_sensitive_changed_files(tmp_path):
    project = _make_git_project(tmp_path)
    db_path = tmp_path / "harness.db"
    service = HarnessService(db_path=db_path, project_root=project)
    proposal = service.create_capability_proposal_from_intent(user_text="위험한 파일 수정", capability_intent=_registry_probe_intent())
    proposal_id = proposal["proposal"]["proposal_id"]
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal_id, "approved_for_dev", actor="test")
    patch_path = _make_blocked_env_patch(project)
    with HarnessMemory(db_path) as memory:
        memory.transition_work_item(work_id, "waiting_approval", actor="test", payload={})
        memory.add_work_event(work_id, "job_completed", actor="worker", payload={"job_id": "job1", "result": {"status": "patch_ready", "patch_path": str(patch_path), "changed_files": [".env"]}})
        memory.conn.commit()

    with pytest.raises(ActivationError, match="preflight failed"):
        ActivationService(
            ActivationConfig(
                db_path=db_path,
                project_root=project,
                self_patch_run_root=project / "artifacts" / "self_patch",
                test_command=("python", "-c", "pass"),
                verify_activation=False,
            )
        ).activate_work_item(work_id, actor="test")

    assert not (project / ".env").exists()
    assert service.work_item(work_id)["work_item"]["status"] == "reviewing"


def _make_git_project(tmp_path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (project / "src" / "demo.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(project, "init")
    _git(project, "config", "user.name", "Test")
    _git(project, "config", "user.email", "test@example.invalid")
    _git(project, "add", "-A")
    _git(project, "commit", "-m", "baseline")
    return project


def _make_patch(project: Path, tmp_path, *, work_id: str) -> Path:
    (project / "src" / "demo.py").write_text("VALUE = 2\n", encoding="utf-8")
    patch = project / "artifacts" / "self_patch" / "job1" / "proposal.patch"
    patch.parent.mkdir(parents=True)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=project, text=True, encoding="utf-8", capture_output=True, check=True).stdout
    patch.write_text(diff, encoding="utf-8")
    subprocess.run(["git", "checkout", "--", "src/demo.py"], cwd=project, check=True)
    return patch


def _make_registry_patch(project: Path, command: list[str] | None = None) -> Path:
    command = command or ["python", "-c", "import json; print(json.dumps({'ok': True}))"]
    (project / "registry").mkdir()
    (project / "registry" / "actions.json").write_text(
        """{
  "schema_version": "neurokernel-action-registry-v1",
  "actions": [
    {
      "action_id": "get_test_probe",
      "title": "테스트 probe",
      "risk_level": "low",
      "side_effect": false,
      "requires_approval": false,
      "executor": "readonly_command",
      "allowed_targets": ["local"],
      "executor_config": {
        "command": %s,
        "output": "json",
        "timeout_seconds": 5
      },
      "test_plan": [{"name": "returns_ok", "assertions": ["ok is true"]}]
    }
  ]
}
"""
        % __import__("json").dumps(command, ensure_ascii=False),
        encoding="utf-8",
    )
    patch = project / "artifacts" / "self_patch" / "job1" / "proposal.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "add", "-N", "registry/actions.json"], cwd=project, check=True)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=project, text=True, encoding="utf-8", capture_output=True, check=True).stdout
    patch.write_text(diff, encoding="utf-8")
    subprocess.run(["git", "reset", "--", "registry/actions.json"], cwd=project, check=True)
    subprocess.run(["git", "clean", "-fd", "registry"], cwd=project, check=True)
    return patch


def _make_registry_v2_patch(project: Path, *, action_id: str, command: list[str] | None = None) -> Path:
    command = command or ["python", "-c", "import json; print(json.dumps({'ok': True}))"]
    (project / "registry").mkdir()
    (project / "registry" / "actions.json").write_text(
        """{
  "schema_version": "neurokernel-action-registry-v2",
  "actions": [
    {
      "action_id": "%s",
      "version": "1.0.0",
      "title": "테스트 probe v2",
      "description": "activation v2 출력 계약 확인용 probe",
      "status": "active",
      "risk_level": "low",
      "side_effect": false,
      "requires_approval": false,
      "role": "readonly_action",
      "executor": "readonly_command",
      "target": ["local"],
      "inputs_schema": {"type": "object", "properties": {}, "required": []},
      "outputs_schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
      "executor_config": {
        "command": %s,
        "output": "json",
        "timeout_seconds": 5
      },
      "test_plan": ["ok 필드를 반환한다"],
      "examples": ["테스트 probe 확인"]
    }
  ]
}
"""
        % (action_id, __import__("json").dumps(command, ensure_ascii=False)),
        encoding="utf-8",
    )
    patch = project / "artifacts" / "self_patch" / "job1" / "proposal.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "add", "-N", "registry/actions.json"], cwd=project, check=True)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=project, text=True, encoding="utf-8", capture_output=True, check=True).stdout
    patch.write_text(diff, encoding="utf-8")
    subprocess.run(["git", "reset", "--", "registry/actions.json"], cwd=project, check=True)
    subprocess.run(["git", "clean", "-fd", "registry"], cwd=project, check=True)
    return patch


def _make_blocked_env_patch(project: Path) -> Path:
    (project / ".env").write_text("TOKEN=should_not_activate\n", encoding="utf-8")
    patch = project / "artifacts" / "self_patch" / "job1" / "proposal.patch"
    patch.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "add", "-N", ".env"], cwd=project, check=True)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=project, text=True, encoding="utf-8", capture_output=True, check=True).stdout
    patch.write_text(diff, encoding="utf-8")
    subprocess.run(["git", "reset", "--", ".env"], cwd=project, check=True)
    subprocess.run(["git", "clean", "-fd", ".env"], cwd=project, check=True)
    return patch


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=True)


def _git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, text=True, encoding="utf-8", capture_output=True, check=True).stdout


def _runner_that_fails_commit(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    if cmd[:2] == ["git", "commit"]:
        return subprocess.CompletedProcess(cmd, 1, "", "simulated commit failure")
    return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", capture_output=True, timeout=timeout_seconds, check=False)


def _cpu_usage_intent():
    return {
        "kind": "gap",
        "reply": "CPU 사용률 확인 능력을 후보로 올릴게.",
        "gap": {
            "gap_type": "missing_action",
            "requested_capability": "현재 CPU 사용률 확인",
            "normalized_request": "orangepi5 현재 CPU 사용률을 조회한다",
            "matched_existing_actions": [],
            "confidence": 0.9,
        },
        "proposal": {
            "action_id": "get_cpu_usage",
            "capability_name": "CPU 사용률 확인",
            "purpose": "Orange Pi 5의 현재 CPU 사용률을 조회한다.",
            "target": "orangepi5",
            "risk_level": "low",
            "side_effect": False,
            "requires_approval": False,
            "inputs": {"type": "object", "properties": {}, "required": []},
            "outputs": {"type": "object", "properties": {"used_percent": {"type": "number"}}, "required": ["used_percent"]},
            "implementation_hint": {"executor": "readonly_system", "suggested_library": "psutil", "notes": "read-only"},
            "test_plan": [{"name": "returns_percent", "type": "unit", "assertions": ["0 <= used_percent <= 100"]}],
            "safety_notes": ["read-only"],
            "confidence": 0.9,
            "approval_required_for_implementation": True,
            "activation_requires_tests": True,
        },
        "confidence": 0.9,
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }


def _registry_probe_intent(action_id: str = "get_test_probe"):
    payload = _cpu_usage_intent()
    payload["proposal"] = {
        **payload["proposal"],
        "action_id": action_id,
        "capability_name": "테스트 probe",
        "purpose": "registry 기반 테스트 probe를 조회한다.",
        "target": "local",
        "outputs": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    }
    return payload
