from neurokernel_seed.harness.memory import HarnessMemory
from neurokernel_seed.harness.service import HarnessService


def test_self_improvement_analyzes_missing_model_and_code_deficits(tmp_path):
    project = _project_with_large_module(tmp_path)
    db = tmp_path / "harness.db"
    with HarnessMemory(db) as memory:
        for index in range(2):
            memory.add_interaction_outcome(
                request_text=f"active model status missing {index}",
                response_text="artifact list only",
                required_outputs=["active_model_status"],
                answered_outputs=["artifacts"],
                missing_outputs=["active_model_status"],
                answer_quality="partial",
            )

    service = HarnessService(db_path=db, project_root=project)

    result = service.analyze_self_improvement(min_gap_count=2, lookback=20, max_code_candidates=3, min_code_score=60)
    kinds = {item["kind"] for item in result["deficits"]}

    assert result["status"] == "completed"
    assert {"missing_output", "world_training_connection", "code_structure"} <= kinds
    assert result["autonomy_boundary"]["develop"] == "requires user approval on proposed work"
    readiness = result["signals"]["readiness"]
    assert readiness["schema_version"] == "neurokernel-self-improvement-readiness-v1"
    assert {item["id"] for item in readiness["criteria"]} >= {
        "deficit_detection",
        "approval_boundary",
        "work_queue_available",
        "runtime_learning_data",
        "conversation_task_linking",
    }


def test_self_improvement_proposes_approval_gated_self_patch_and_deduplicates(tmp_path):
    project = _project_with_large_module(tmp_path)
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=project)

    first = service.propose_self_improvement(max_code_candidates=2, min_code_score=60, max_proposals_per_cycle=2, actor="test")
    second = service.propose_self_improvement(max_code_candidates=2, min_code_score=60, max_proposals_per_cycle=2, actor="test")

    assert first["created_count"] >= 1
    created_work = [item["work_item"] for item in first["created"] if item.get("work_item")]
    assert any(item["type"] == "self_patch" for item in created_work)
    assert all(item["status"] == "proposed" for item in created_work)
    assert any(item.get("kind") == "duplicate" for item in second["skipped"])


def test_self_improvement_readiness_flags_unavailable_queue(tmp_path):
    project = _project_with_large_module(tmp_path)
    service = HarnessService(db_path=tmp_path / "harness.db", project_root=project)

    result = service.analyze_self_improvement(max_code_candidates=1, min_code_score=60)

    assert any(item["deficit_id"] == "work_pipeline:queue_unavailable" for item in result["deficits"])
    queue_check = next(item for item in result["signals"]["readiness"]["criteria"] if item["id"] == "work_queue_available")
    assert queue_check["passed"] is False


def test_self_improvement_api_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from neurokernel_seed.api.server import create_app

    project = _project_with_large_module(tmp_path)
    client = TestClient(create_app(db_path=tmp_path / "harness.db", project_root=project))

    analyzed = client.get("/self-improvement/analyze", params={"max_code_candidates": 2, "min_code_score": 60}).json()
    proposed = client.post("/self-improvement/propose", json={"max_code_candidates": 2, "min_code_score": 60, "max_proposals_per_cycle": 1}).json()

    assert analyzed["schema_version"] == "neurokernel-self-improvement-analysis-v1"
    assert proposed["schema_version"] == "neurokernel-self-improvement-proposal-v1"
    assert proposed["created_count"] == 1


def _project_with_large_module(tmp_path):
    project = tmp_path / "project"
    package = project / "src" / "sample"
    package.mkdir(parents=True)
    lines = ["def giant(value):", "    total = 0"]
    for index in range(250):
        lines.append(f"    total += value + {index}")
    lines.append("    return total")
    for index in range(25):
        lines.append("")
        lines.append(f"def helper_{index}():")
        lines.append(f"    return {index}")
    (package / "large_module.py").write_text("\n".join(lines), encoding="utf-8")
    return project
