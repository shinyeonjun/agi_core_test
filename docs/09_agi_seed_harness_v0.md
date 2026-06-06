# AGI Seed Harness v0

Harness v0는 NeuroKernel Core를 Discord/LLM/Codex에 연결하기 전, 실제 작업 루프를 안전하게 감싸는 실행 뼈대다.

## Scope

v0에서 하는 것:

- Action Catalog
- Task Spec validation
- Safety Gate
- SQLite Memory
- Task State Machine
- Read-only Executor
- Core API surface

v0에서 하지 않는 것:

- Discord integration
- LLM integration
- arbitrary shell execution
- write executor
- git push
- deploy
- reboot
- secret access
- autonomous high-risk execution

## Core Principle

기본값은 deny다.

LLM이나 Discord가 붙더라도 자연어가 바로 실행기로 가지 않는다.

```text
natural language
-> Task Spec
-> validator
-> Safety Gate
-> state machine
-> allowlisted executor
-> SQLite trace
```

## CLI Smoke

```bash
python -m neurokernel_seed.cli harness-actions
python -m neurokernel_seed.cli harness-status
```

Task Spec 예시:

```json
{
  "goal": "Orange Pi 상태 확인",
  "target": "orangepi5",
  "allowed_actions": ["get_uptime", "get_memory_usage", "get_disk_usage"],
  "blocked_actions": [],
  "success_criteria": ["CPU/RAM/DISK 상태 반환"],
  "risk_level": "low",
  "requires_approval": false,
  "timeout_seconds": 30,
  "mode": "readonly"
}
```

실행:

```bash
python -m neurokernel_seed.cli harness-create-task --task-file task.json
python -m neurokernel_seed.cli harness-dry-run <task_id>
python -m neurokernel_seed.cli harness-run <task_id>
```

## API

FastAPI는 선택 의존성이다.

```bash
pip install fastapi uvicorn
python -m neurokernel_seed.cli serve-core-api --host 0.0.0.0 --port 8765
```

Endpoints:

- `GET /health`
- `GET /status`
- `GET /actions`
- `POST /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/dry-run`
- `POST /tasks/{task_id}/run`
- `POST /tasks/{task_id}/approve`
- `POST /tasks/{task_id}/reject`
- `GET /tasks/{task_id}/trace`
- `GET /trace/recent`
- `POST /benchmark`

## Memory

기록은 append-first 방식이다.

SQLite tables:

- `tasks`
- `task_events`
- `action_decisions`
- `execution_results`
- `approvals`
- `traces`

## Current Runtime Note

Orange Pi에서는 ONNX CPU가 현재 작은 world model에 가장 빠르다.
RKNN NPU backend는 검증됐지만 optional로 둔다.
