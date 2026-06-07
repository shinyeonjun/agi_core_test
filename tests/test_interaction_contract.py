from neurokernel_seed.harness.interaction_contract import interaction_contract_from_runtime


def test_interaction_contract_marks_partial_answer_from_actual_runtime_result():
    contract = interaction_contract_from_runtime(
        request_text="메모리, cpu, 저장공간 알려줘",
        response_text="CPU만 확인됨",
        task={"allowed_actions": ["get_cpu_per_core_usage"]},
        core_result={
            "status": "completed",
            "action": "get_cpu_per_core_usage",
            "execution_result": {
                "action_id": "get_cpu_per_core_usage",
                "success": True,
                "result": {"core_count": 8, "per_core_percent": [{"core": 0, "used_percent": 0.0}]},
            },
        },
    )

    assert contract["required_outputs"] == ["cpu_usage", "disk_usage", "memory_usage"]
    assert contract["answered_outputs"] == ["cpu_usage"]
    assert contract["missing_outputs"] == ["disk_usage", "memory_usage"]
    assert contract["answer_quality"] == "partial"


def test_interaction_contract_tracks_active_model_status_gap():
    contract = interaction_contract_from_runtime(
        request_text="current runtime/world model status",
        response_text="artifact files exist",
        task={"allowed_actions": ["list_artifacts"]},
        core_result={
            "status": "completed",
            "action": "list_artifacts",
            "execution_result": {
                "success": True,
                "action_id": "list_artifacts",
                "result": {"path": "artifacts", "items": []},
            },
        },
    )

    assert "active_model_status" in contract["required_outputs"]
    assert "artifacts" in contract["answered_outputs"]
    assert "active_model_status" in contract["missing_outputs"]
