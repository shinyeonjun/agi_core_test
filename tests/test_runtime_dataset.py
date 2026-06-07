import json

from neurokernel_seed.core.gate import ActionGate
from neurokernel_seed.core.schema import Action, ActionSpec, Prediction, WorldState
from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.replay.runtime_dataset import (
    RUNTIME_REPLAY_SCHEMA_VERSION,
    RuntimeReplayEtlConfig,
    check_runtime_replay_gates,
    export_runtime_replay,
    run_runtime_replay_etl,
    validate_runtime_replay,
)


def test_runtime_replay_exports_harness_decisions_and_results(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "목록 확인",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])

    out = tmp_path / "runtime_replay.jsonl"
    manifest = export_runtime_replay(db, out)
    validation = validate_runtime_replay(out)
    gates = check_runtime_replay_gates(out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert manifest["schema_version"] == RUNTIME_REPLAY_SCHEMA_VERSION
    assert validation["accepted"] is True
    assert gates["passed"] is True
    assert rows[0]["decision"]["candidate_actions"] == [{"action_id": "list_artifacts"}]
    assert rows[0]["execution"]["action_id"] == "list_artifacts"
    assert rows[0]["outcome"]["success"] is True
    assert rows[0]["task"]["status"] == "executing"
    assert rows[0]["task"]["final_status"] == "completed"
    assert rows[0]["row_id"].startswith("runtime_")
    assert rows[0]["lineage"]["decision_id"] == 1
    assert rows[0]["lineage"]["experience_id"].startswith("exp_run_")
    assert rows[0]["experience"]["experience_aligned"] is True
    assert rows[0]["experience"]["decision_policy"] == "deterministic_safety_first"
    assert rows[0]["experience"]["model_used"] is False
    candidate = rows[0]["candidate_outcomes"][0]
    assert len(rows[0]["candidate_outcomes"]) == 1
    assert candidate["action_id"] == "list_artifacts"
    assert candidate["params"] == {"path": "."}
    assert candidate["safety_decision"]["decision"] == "allow"
    assert candidate["model_score"]["model_used"] is False
    assert candidate["model_score"]["reason"] == "model_unavailable"
    assert candidate["selected"] is True
    assert candidate["executed"] is True
    assert candidate["execution_result_known"] is True
    assert candidate["outcome"]["success"] is True
    assert candidate["outcome"]["reward"] == 1.0
    assert candidate["target_mask"] == {
        "success": True,
        "reward": True,
        "duration_seconds": True,
        "failure_present": True,
    }


def test_runtime_replay_etl_commits_only_after_gates_pass(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "목록 확인",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])
    out = tmp_path / "runtime_replay.jsonl"

    result = run_runtime_replay_etl(RuntimeReplayEtlConfig(db_path=db, out_path=out, min_rows=1))

    assert result["ready_for_runtime_training"] is True
    assert out.exists()
    assert result["sha256"]
    assert result["extraction"]["source_snapshot"]["table_counts"]["action_decisions"] == 1


def test_runtime_replay_etl_quarantines_failed_gate_without_overwrite(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    created = service.create_task(
        {
            "goal": "목록 확인",
            "target": "orangepi5",
            "allowed_actions": ["list_artifacts"],
            "context": {"params": {"path": "."}},
            "risk_level": "low",
            "requires_approval": False,
            "mode": "readonly",
        }
    )
    service.run(created["task"]["task_id"])
    out = tmp_path / "runtime_replay.jsonl"
    out.write_text("existing\n", encoding="utf-8")

    result = run_runtime_replay_etl(RuntimeReplayEtlConfig(db_path=db, out_path=out, min_rows=99))

    assert result["ready_for_runtime_training"] is False
    assert out.read_text(encoding="utf-8") == "existing\n"
    assert "quarantine" in result
    assert result["gates"]["gates"]["min_rows"]["passed"] is False


def test_runtime_replay_rejects_secret_like_keys(tmp_path):
    replay = tmp_path / "runtime_replay.jsonl"
    replay.write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_REPLAY_SCHEMA_VERSION,
                "task": {"task_id": "task_1"},
                "decision": {"candidate_actions": [], "chosen_action": None},
                "execution": {"token": "abc"},
                "outcome": {"success": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_runtime_replay(replay)

    assert result["accepted"] is False
    assert any("forbidden key path" in item["error"] for item in result["errors"])


def test_model_only_score_excludes_old_visibility_adjustment():
    state = WorldState(
        env_name="lock.test",
        state_id="s0",
        vector=(0.0,),
        facts={"visibility_mode": "hidden", "current_slot_known": False, "is_last_step_index": False},
    )
    action = Action("inspect")
    spec = ActionSpec("inspect", reveals_information=True, consumes_progress_step=False)
    prediction = Prediction(next_state_vector=(0.0,), reward=0.0, progress_delta=0.0, information_gain=0.0, success_probability=0.0)
    gate = ActionGate((spec,), predictor=object(), mode="model_only")  # type: ignore[arg-type]

    breakdown = gate.score_breakdown(state, action, prediction)

    assert breakdown["model_score"] == 0.0
    assert breakdown["final_score"] == 0.0
    retired_key = "legacy_" + "visibility_guard"
    assert retired_key not in breakdown
