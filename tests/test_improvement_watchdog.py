from neurokernel_seed.harness.action_catalog import build_action_catalog
from neurokernel_seed.harness.improvement import ImprovementService
from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.service import HarnessService


def test_improvement_analyzer_detects_uncovered_missing_output(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        for index in range(2):
            memory.add_interaction_outcome(
                request_text=f"need active model status {index}",
                response_text="only artifact files were listed",
                required_outputs=["active_model_status"],
                answered_outputs=["artifacts"],
                missing_outputs=["active_model_status"],
                answer_quality="partial",
            )

    service = ImprovementService(
        db_path=db,
        catalog=build_action_catalog(),
        work_items=HarnessService(db_path=db, project_root=tmp_path).work_items_service,
    )

    result = service.analyze(min_gap_count=2, lookback=20)

    assert result["candidate_gaps"][0]["output_key"] == "active_model_status"
    assert result["candidate_gaps"][0]["count"] == 2


def test_improvement_service_proposes_self_patch_from_repeated_gap(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        for index in range(2):
            memory.add_interaction_outcome(
                user_id="discord:1",
                channel_id="chan",
                request_text=f"need active model status {index}",
                response_text="only artifact files were listed",
                required_outputs=["active_model_status"],
                answered_outputs=["artifacts"],
                missing_outputs=["active_model_status"],
                answer_quality="partial",
            )

    service = HarnessService(db_path=db, project_root=tmp_path)

    result = service.propose_improvements(min_gap_count=2, lookback=20, actor="test")

    assert result["created_count"] == 1
    created = result["created"][0]
    assert created["proposal"]["action_id"] == "get_active_model_status"
    assert created["work_item"]["type"] == "self_patch"
    assert created["work_item"]["status"] == "proposed"


def test_improvement_service_reuses_existing_open_proposal(tmp_path):
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        for index in range(2):
            memory.add_interaction_outcome(
                request_text=f"need active model status {index}",
                response_text="only artifact files were listed",
                required_outputs=["active_model_status"],
                answered_outputs=["artifacts"],
                missing_outputs=["active_model_status"],
                answer_quality="partial",
            )

    service = HarnessService(db_path=db, project_root=tmp_path)

    first = service.propose_improvements(min_gap_count=2, lookback=20, actor="test")
    second = service.propose_improvements(min_gap_count=2, lookback=20, actor="test")

    assert first["created_count"] == 1
    assert second["created_count"] == 0
    assert second["skipped"][0]["kind"] == "duplicate"
