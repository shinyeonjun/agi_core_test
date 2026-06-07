import json

from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.replay.real_world_transitions import REAL_WORLD_TRANSITION_SCHEMA_VERSION, export_real_world_transitions


def test_real_world_transition_export_uses_accumulated_runtime_data(tmp_path):
    db = tmp_path / "harness.db"
    service = HarnessService(db_path=db, project_root=tmp_path)
    task = {
        "goal": "메모리 확인",
        "target": "orangepi5",
        "allowed_actions": ["get_memory_usage"],
        "context": {},
        "risk_level": "low",
        "requires_approval": False,
        "mode": "readonly",
    }
    created = service.create_task(task)
    payload = service.run(created["task"]["task_id"])
    service.add_interaction_outcome(
        {
            "source": "discord",
            "user_id": "discord:1",
            "channel_id": "chan",
            "task_id": created["task"]["task_id"],
            "request_text": "메모리 알려줘",
            "response_text": "메모리 확인 완료",
            "task": task,
            "core_result": payload,
        }
    )

    out = tmp_path / "real_world_transitions.jsonl"
    manifest = export_real_world_transitions(db, out)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    assert manifest["schema_version"] == REAL_WORLD_TRANSITION_SCHEMA_VERSION
    assert manifest["rows"] == 1
    assert rows[0]["schema_version"] == REAL_WORLD_TRANSITION_SCHEMA_VERSION
    assert rows[0]["action"]["action_id"] == "get_memory_usage"
    assert rows[0]["interaction"]["interaction_aligned"] is True
    assert rows[0]["state"]["required_outputs"] == ["memory_usage"]
    assert rows[0]["next_state"]["answered_outputs"] == ["memory_usage"]
