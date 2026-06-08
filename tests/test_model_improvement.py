import subprocess
from pathlib import Path

from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.model_improvement import ModelImprovementService
from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.harness.training_worker import TrainingPipelineWorker, TrainingWorkerConfig
from neurokernel_seed.harness.work_queue import InMemoryWorkQueue, queue_name_for_work_type
from neurokernel_seed.harness.worker import WorkDispatcher, WorkDispatcherConfig


def test_model_improvement_proposes_runtime_training_from_accumulated_candidates(tmp_path):
    queue = InMemoryWorkQueue()
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path, work_queue=queue)
    task_id = _create_task_with_known_candidates(db, service, groups=6, candidates_per_group=2)

    result = service.propose_model_improvements(
        min_known_runtime_candidates=10,
        min_new_known_runtime_candidates=5,
        min_runtime_ranking_groups=5,
        target_runtime_top1=0.65,
        actor="test",
    )

    assert task_id
    assert result["created_count"] == 1
    work = result["created"][0]["work_item"]
    assert work["type"] == "training_pipeline"
    assert work["status"] == "proposed"
    metadata = work["metadata_json"]
    assert metadata["model_slot"] == "runtime_action"
    assert metadata["source"] == "orangepi_accumulated_data"
    assert metadata["source_snapshot"]["known_candidate_outcomes"] == 12
    assert metadata["nk_command"]["action"] == "runtime-pipeline"


def test_model_improvement_does_not_duplicate_open_runtime_training_work(tmp_path):
    queue = InMemoryWorkQueue()
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path, work_queue=queue)
    _create_task_with_known_candidates(db, service, groups=6, candidates_per_group=2)

    first = service.propose_model_improvements(
        min_known_runtime_candidates=10,
        min_new_known_runtime_candidates=5,
        min_runtime_ranking_groups=5,
        actor="test",
    )
    second = service.propose_model_improvements(
        min_known_runtime_candidates=10,
        min_new_known_runtime_candidates=5,
        min_runtime_ranking_groups=5,
        actor="test",
    )

    assert first["created_count"] == 1
    assert second["created_count"] == 0
    assert second["skipped"][0]["reason"] == "open_runtime_training_work_exists"


def test_training_pipeline_queue_and_worker_complete_job(tmp_path):
    queue = InMemoryWorkQueue()
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path, work_queue=queue)
    _create_task_with_known_candidates(db, service, groups=6, candidates_per_group=2)
    proposed = service.propose_model_improvements(
        min_known_runtime_candidates=10,
        min_new_known_runtime_candidates=5,
        min_runtime_ranking_groups=5,
        actor="test",
    )
    work_id = proposed["created"][0]["work_item"]["work_id"]

    accepted = service.transition_work_item(work_id, "accepted", actor="test")
    assert accepted["queue"]["queued"] is True
    assert queue.messages[0][0] == "training_pipeline"
    assert queue_name_for_work_type("training_pipeline") == "training_pipeline"

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=db, project_root=tmp_path, worker_id="trainer", queues=("training_pipeline",), once=True),
        work_queue=queue,
        training_runner=TrainingPipelineWorker(
            TrainingWorkerConfig(project_root=tmp_path, enabled=True, timeout_seconds=30),
            runner=lambda cmd, cwd, timeout_seconds: subprocess.CompletedProcess(cmd, 0, '{"status":"completed"}', ""),
        ),
    )
    assert dispatcher.run_once() == 1

    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "completed"
    assert item["jobs"][0]["status"] == "completed"
    assert item["notes"][-1]["note_redacted"].startswith("Training job")


def test_training_worker_disabled_blocks_instead_of_pretending_success(tmp_path):
    queue = InMemoryWorkQueue()
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path, work_queue=queue)
    _create_task_with_known_candidates(db, service, groups=6, candidates_per_group=2)
    proposed = service.propose_model_improvements(
        min_known_runtime_candidates=10,
        min_new_known_runtime_candidates=5,
        min_runtime_ranking_groups=5,
        actor="test",
    )
    work_id = proposed["created"][0]["work_item"]["work_id"]
    service.transition_work_item(work_id, "accepted", actor="test")

    dispatcher = WorkDispatcher(
        config=WorkDispatcherConfig(db_path=db, project_root=tmp_path, worker_id="edge-worker", queues=("training_pipeline",), once=True),
        work_queue=queue,
        training_runner=TrainingPipelineWorker(TrainingWorkerConfig(project_root=tmp_path, enabled=False)),
    )
    dispatcher.run_once()

    item = service.work_item(work_id)
    assert item["work_item"]["status"] == "blocked"
    assert "waiting_for_training_node" in item["notes"][-1]["note_redacted"]


def _create_task_with_known_candidates(db: Path, service: HarnessService, *, groups: int, candidates_per_group: int) -> str:
    created = service.create_task(
        {
            "goal": "runtime candidate data",
            "target": "orangepi5",
            "allowed_actions": ["get_cpu_temp"],
            "context": {},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    task_id = created["task"]["task_id"]
    with HarnessMemory(db) as memory:
        for group_index in range(groups):
            candidates = []
            for candidate_index in range(candidates_per_group):
                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "action_id": f"action_{candidate_index}",
                        "selected": candidate_index == 0,
                        "executed": True,
                        "execution_result_known": True,
                        "outcome": {"success": candidate_index == 0, "reward": 1.0 if candidate_index == 0 else 0.1},
                        "target_mask": {"success": True, "reward": True},
                    }
                )
            memory.record_experience(
                task_id=task_id,
                phase=f"probe_{group_index}",
                status="completed",
                decision_policy="test",
                before_state={"group": group_index},
                after_state={"done": True},
                outcome={"success": True},
                learning_masks={"candidate_outcomes_known": True},
                candidates=candidates,
            )
    return task_id
