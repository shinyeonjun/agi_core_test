# AGI Seed Harness v0

이 문서는 NeuroKernel Core를 Discord, LLM 언어기관, Codex 개발 워커와 연결하기 위한 실행 하네스 기준이다.

현재 목표는 "모든 것을 자동 실행하는 챗봇"이 아니다. 목표는 실패를 기록하고, 필요한 능력을 제안하고, 승인된 개발 작업을 격리된 공간에서 구현/테스트한 뒤 사람이 검토할 수 있는 패치로 남기는 것이다.

## v0에서 되는 것

- Action Catalog
- TaskSpec 검증
- Safety Gate
- SQLite Memory
- Task State Machine
- Read-only Executor
- Core API
- Discord bot
- Codex 언어기관
- Capability Proposal
- Redis Work Queue
- SelfPatchWorker 1차 루프

## v0에서 아직 안 되는 것

- 패치 자동 장착
- 동적 Action Registry 활성화
- 대형 프로젝트 워커
- 논문 수집/데이터셋 변환 워커
- 학습 자동화 워커
- 승인 없는 파일 쓰기/배포/git push
- 비밀값 조회
- 고위험 자율 실행

## 핵심 원칙

기본값은 실행이 아니라 기록과 검증이다.

```text
natural language
-> language organ
-> TaskSpec / WorkRoute / CapabilityIntent
-> validator
-> Safety Gate
-> state machine
-> executor or work queue
-> trace / patch / test result
-> human approval
```

## Runtime Task 흐름

이미 존재하는 low-risk action은 TaskSpec으로 변환되어 바로 실행될 수 있다.

```bash
python -m neurokernel_seed.cli harness-actions
python -m neurokernel_seed.cli harness-status
```

TaskSpec 예시:

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

## SelfPatchWorker 흐름

없는 능력은 바로 성공한 척하지 않는다.

```text
사용자: cpu 사용률 알려줘
-> 언어기관: 현재 action catalog에 없음
-> Core: capability proposal 생성
-> Discord: 개발 후보 승인 버튼 표시
-> 승인
-> Redis queue: self_patch job enqueue
-> WorkDispatcher
-> SelfPatchWorker
-> artifacts/self_patch/<job_id>/workspace 에 격리 복사
-> Codex 개발 세션 실행
-> pytest 실행
-> proposal.patch / summary.json 저장
-> work item 상태를 waiting_approval/reviewing/blocked로 전환
```

SelfPatchWorker는 라이브 repo를 직접 수정하지 않는다. 패치 산출물을 만들고 멈춘다.

상태 의미:

- `waiting_approval`: 패치가 있고 테스트가 통과했다. 사람이 검토 후 장착해야 한다.
- `reviewing`: 패치는 있지만 테스트가 실패했다. 분석/수정이 필요하다.
- `blocked`: 패치가 없거나 작업 조건이 부족하다.

## Core API

```bash
python -m neurokernel_seed.cli serve-core-api --host 127.0.0.1 --port 8765
```

주요 endpoint:

- `GET /health`
- `GET /status`
- `GET /actions`
- `POST /tasks`
- `POST /tasks/{task_id}/run`
- `GET /capability-proposals`
- `POST /capability-proposals/from-request`
- `POST /capability-proposals/{proposal_id}/approve-dev`
- `GET /work-items`
- `POST /work-items/{work_id}/enqueue`
- `POST /work-items/{work_id}/status`

## Work Worker

```bash
python -m neurokernel_seed.cli serve-work-worker \
  --db data/harness.db \
  --project-root . \
  --queues self_patch external_work \
  --block-ms 0
```

Orange Pi 통합 실행은 `tools/run_orangepi_stack.sh`가 Core API, Discord bot, Work Worker를 함께 띄운다.

## Memory

SQLite tables:

- `tasks`
- `task_events`
- `execution_results`
- `traces`
- `conversation_messages`
- `user_preferences`
- `capability_gaps`
- `capability_proposals`
- `work_items`
- `work_jobs`
- `work_events`
- `work_notes`

## 현재 런타임 메모

World model은 ONNX CPU와 RKNN NPU 모두 실행 검증이 끝났다. 현재 작은 모델에서는 ONNX CPU가 더 빠르고 안정적이다. NPU는 모델이 커지거나 batch 처리 이득이 생길 때 다시 비교한다.
