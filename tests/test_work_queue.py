from pathlib import Path

from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.harness.work_queue import InMemoryWorkQueue
from neurokernel_seed.harness.worker import WorkDispatcher, WorkDispatcherConfig


def test_accepting_work_item_enqueues_job_once(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    created = service.create_work_item_from_route(
        user_text="논문 수집 파이프라인 만들어줘",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )
    work_id = created["work_item"]["work_id"]

    accepted = service.transition_work_item(work_id, "accepted", actor="discord:1")
    duplicate = service.enqueue_work_item(work_id, actor="discord:1")

    assert accepted["queue"]["queued"] is True
    assert len(queue.messages) == 1
    assert duplicate["queued"] is False
    assert duplicate["reason"] == "active_job_exists"
    jobs = service.work_jobs(work_id=work_id)["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"


def test_capability_approval_enqueues_self_patch_job(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    proposal = service.create_capability_proposal_from_intent(
        user_text="CPU 사용률 볼 수 있어?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )

    approved = service.transition_capability_proposal(proposal["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")

    assert approved["queue"]["queued"] is True
    assert queue.messages[0][0] == "self_patch"
    assert queue.messages[0][2]["work_id"] == proposal["work_item"]["work_id"]


def test_dispatcher_moves_accepted_work_to_planned_and_acks_job(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    created = service.create_work_item_from_route(
        user_text="논문 수집 파이프라인 만들어줘",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )
    work_id = created["work_item"]["work_id"]
    service.transition_work_item(work_id, "accepted", actor="discord:1")

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=Path(tmp_path / "harness.db"), worker_id="test-worker", queues=("external_work",), once=True),
        work_queue=queue,
    )
    processed = dispatcher.run_once()

    assert processed == 1
    assert queue.acked
    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "planned"
    assert item["jobs"][0]["status"] == "completed"
    assert item["jobs"][0]["heartbeat_at"] is not None
    assert item["notes"]


def test_pipeline_status_reports_stale_running_jobs(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    created = service.create_work_item_from_route(
        user_text="long work",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )
    work_id = created["work_item"]["work_id"]
    service.transition_work_item(work_id, "accepted", actor="discord:1")
    job = service.work_jobs(work_id=work_id)["jobs"][0]
    from neurokernel_seed.harness.memory import HarnessMemory

    with HarnessMemory(tmp_path / "harness.db") as memory:
        memory.mark_work_job_running(job["job_id"], worker_id="test-worker")
        memory.conn.execute("UPDATE work_jobs SET heartbeat_at=datetime('now', '-10 minutes'), updated_at=datetime('now', '-10 minutes') WHERE job_id=?", (job["job_id"],))
        memory.conn.commit()

    status = service.work_pipeline_status(stale_after_seconds=60)

    assert status["summary"]["stale_running_job_count"] == 1
    assert status["stale_running_jobs"][0]["job_id"] == job["job_id"]
    assert "stale_running_jobs" in status["summary"]["attention"]


def test_promoting_external_work_creates_child_self_patch_and_enqueues(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    created = service.create_work_item_from_route(
        user_text="자동 진단 리포트 기능 만들어줘",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )
    parent_id = created["work_item"]["work_id"]
    service.transition_work_item(parent_id, "accepted", actor="discord:1")

    promoted = service.promote_work_item_to_self_patch(parent_id, actor="discord:1")

    assert promoted["promoted"] is True
    child = promoted["child_work_item"]
    assert child["type"] == "self_patch"
    assert child["status"] == "accepted"
    assert child["parent_work_id"] == parent_id
    assert promoted["queue"]["queued"] is True
    assert queue.messages[-1][0] == "self_patch"
    assert queue.messages[-1][2]["work_id"] == child["work_id"]
    detail = service.work_item(parent_id)
    assert any(event["event_type"] == "promoted_to_self_patch" for event in detail["events"])


def test_promoting_external_work_reuses_existing_child_self_patch(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    created = service.create_work_item_from_route(
        user_text="자동 진단 리포트 기능 만들어줘",
        user_id="discord:1",
        channel_id="chan",
        route_decision=_external_work_route(),
    )
    parent_id = created["work_item"]["work_id"]
    first = service.promote_work_item_to_self_patch(parent_id, actor="discord:1")
    second = service.promote_work_item_to_self_patch(parent_id, actor="discord:1")

    assert first["promoted"] is True
    assert second["promoted"] is False
    assert second["reason"] == "child_self_patch_exists"
    assert second["child_work_item"]["work_id"] == first["child_work_item"]["work_id"]
    assert len(queue.messages) == 1


def test_dispatcher_runs_self_patch_and_waits_for_activation_approval(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    proposal = service.create_capability_proposal_from_intent(
        user_text="CPU ?ъ슜瑜?蹂????덉뼱?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=Path(tmp_path / "harness.db"), project_root=tmp_path, worker_id="test-worker", queues=("self_patch",), once=True),
        work_queue=queue,
        self_patch_runner=FakeSelfPatchRunner(),
    )
    processed = dispatcher.run_once()

    assert processed == 1
    assert queue.acked
    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "waiting_approval"
    assert item["jobs"][0]["status"] == "completed"
    assert item["jobs"][0]["payload_json"]["work_type"] == "self_patch"
    assert "patch=" in item["notes"][-1]["note_redacted"]


def test_retry_reviewing_self_patch_requeues_with_previous_failure_context(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    proposal = service.create_capability_proposal_from_intent(
        user_text="CPU ???쒐몴?癰?????됰선?",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=Path(tmp_path / "harness.db"), project_root=tmp_path, worker_id="test-worker", queues=("self_patch",), once=True),
        work_queue=queue,
        self_patch_runner=FakeFailedSelfPatchRunner(),
    )
    dispatcher.run_once()

    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "reviewing"
    retry = service.retry_work_item(work_id, actor="discord:1")

    assert retry["queued"] is True
    assert retry["work_item"]["status"] == "running"
    retry_payload = queue.messages[-1][2]
    assert retry_payload["retry"]["previous_status"] == "reviewing"
    assert retry_payload["retry"]["previous_result"]["status"] == "test_failed"
    assert retry_payload["retry"]["previous_result"]["test"]["returncode"] == 1
    assert retry_payload["retry"]["previous_result"]["failure_analysis"]["primary_failure"] == "test_failed"
    assert retry_payload["retry"]["previous_result"]["failure_analysis"]["next_step"] == "inspect_test_tail_and_fix_patch"


def test_dispatcher_returns_self_patch_exception_to_reviewing(tmp_path):
    queue = InMemoryWorkQueue()
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=tmp_path, work_queue=queue)
    proposal = service.create_capability_proposal_from_intent(
        user_text="CPU failure path",
        user_id="discord:1",
        channel_id="chan",
        capability_intent=_cpu_usage_intent(),
    )
    work_id = proposal["work_item"]["work_id"]
    service.transition_capability_proposal(proposal["proposal"]["proposal_id"], "approved_for_dev", actor="discord:1")

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=Path(tmp_path / "harness.db"), project_root=tmp_path, worker_id="test-worker", queues=("self_patch",), once=True),
        work_queue=queue,
        self_patch_runner=FakeRaisingSelfPatchRunner(),
    )
    dispatcher.run_once()

    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "reviewing"
    assert item["jobs"][0]["status"] == "dead_letter"
    assert queue.acked
    assert queue.dead_letters

    retry = service.retry_work_item(work_id, actor="discord:1")
    retry_payload = queue.messages[-1][2]
    assert retry["queued"] is True
    assert retry_payload["retry"]["previous_result"]["status"] == "codex_failed"
    assert "boom" in retry_payload["retry"]["previous_result"]["error"]


def test_dispatcher_defaults_to_trigger_style_blocking_read():
    assert WorkDispatcherConfig().block_ms == 0
    assert WorkDispatcherConfig().idle_sleep_seconds == 0.0


class FakeSelfPatchRunner:
    def run(self, *, job_id, work, payload):
        return {
            "status": "patch_ready",
            "job_id": job_id,
            "work_id": work["work_id"],
            "patch_path": "artifacts/self_patch/job/proposal.patch",
            "changed_files": ["src/neurokernel_seed/harness/executors/readonly_system.py"],
            "test": {"returncode": 0},
        }


class FakeFailedSelfPatchRunner:
    def run(self, *, job_id, work, payload):
        return {
            "status": "test_failed",
            "job_id": job_id,
            "work_id": work["work_id"],
            "patch_path": "artifacts/self_patch/job/proposal.patch",
            "changed_files": ["src/neurokernel_seed/harness/executors/readonly_system.py"],
            "test": {"returncode": 1, "stdout_tail": "assert False", "stderr_tail": ""},
            "diff_check": {"returncode": 0, "stdout_tail": "", "stderr_tail": ""},
            "failure_analysis": {
                "schema_version": "neurokernel-self-patch-failure-analysis-v1",
                "primary_failure": "test_failed",
                "summary": "Patch exists, but the verification command failed.",
                "retryable": True,
                "next_step": "inspect_test_tail_and_fix_patch",
                "signals": {"test_returncode": 1},
            },
        }


class FakeRaisingSelfPatchRunner:
    def run(self, *, job_id, work, payload):
        raise RuntimeError("boom from self-patch runner")


def _external_work_route():
    return {
        "route": "external_work",
        "reason": "larger queued work",
        "confidence": 0.9,
        "work_item": {
            "type": "external_work",
            "title": "논문 수집 파이프라인",
            "goal": "논문을 수집하고 학습데이터 후보로 변환하는 파이프라인을 설계한다.",
            "priority": "medium",
            "risk_level": "low",
            "deliverables": ["plan", "pipeline"],
            "open_questions": [],
        },
        "requires_confirmation": True,
        "clarifying_question": None,
        "safety_notes": [],
    }


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
