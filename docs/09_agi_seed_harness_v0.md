# AGI Seed Harness v0

이 문서는 NeuroKernel Core를 Discord, Codex 언어기관, Codex 개발 워커, Redis 큐, 승인형 장착 파이프라인으로 묶는 실행 하네스 기준이다.

현재 목표는 "그럴듯한 챗봇"이 아니다. 실패를 기록하고, 없는 능력을 후보로 만들고, 격리된 작업 공간에서 구현/테스트한 뒤, 승인받은 패치만 실제 몸에 장착하는 자기개선 루프를 만드는 것이다.

## 지금 되는 것

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
- SelfPatchWorker v1
- Activation Pipeline v1
- Dynamic Action Registry v1

## 아직 안 되는 것

- 장착 후 서비스 자동 reload/restart
- 새 action을 Discord에서 자동 검증하는 승격 게이트
- 대형 프로젝트 전담 워커
- 논문/자료 수집과 데이터셋 변환 워커
- 학습 자동화 워커
- 승인 없는 파일 쓰기, 배포, git push
- 비밀값 조회

## 기본 흐름

```text
자연어 요청
-> Codex 언어기관
-> TaskSpec / WorkRoute / CapabilityIntent
-> Core validator
-> Safety Gate
-> Runtime executor 또는 Redis work queue
-> SelfPatchWorker
-> patch/test/summary 산출
-> Discord 승인
-> Activation Pipeline
-> live repo 적용
-> live repo 테스트
-> proposal active 전환
```

## Runtime Task

이미 존재하는 low-risk action은 TaskSpec으로 변환해서 바로 실행할 수 있다.

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
  "success_criteria": ["상태 정보를 반환한다"],
  "risk_level": "low",
  "requires_approval": false,
  "timeout_seconds": 30,
  "mode": "readonly"
}
```

## SelfPatchWorker

없는 능력은 바로 실행하지 않는다. 먼저 후보를 만들고, 승인되면 격리된 workspace에서 Codex 개발 세션을 실행한다.

```text
사용자: cpu 사용률 알려줘
-> 언어기관: 현재 action catalog에 없음
-> Core: capability proposal 생성
-> Discord: 개발 후보 승인 버튼 표시
-> 승인
-> Redis queue: self_patch job enqueue
-> WorkDispatcher
-> SelfPatchWorker
-> artifacts/self_patch/<job_id>/workspace 복사
-> Codex 개발 세션 실행
-> pytest 실행
-> proposal.patch / summary.json 저장
-> work item 상태를 waiting_approval / reviewing / blocked로 전환
```

SelfPatchWorker는 live repo를 직접 수정하지 않는다. 패치 산출물만 만든다.

## Activation Pipeline

Activation Pipeline은 승인된 self-patch 결과를 실제 프로젝트에 붙이는 단계다.

조건:

- work item 상태가 `waiting_approval`이어야 한다.
- patch-ready self-patch 결과가 있어야 한다.
- patch 파일은 `artifacts/self_patch/**/proposal.patch` 아래에 있어야 한다.
- live repo는 깨끗한 git tree여야 한다.
- `git apply --check`가 통과해야 한다.
- live repo에서 테스트가 통과해야 한다.

성공 시:

- patch를 live repo에 적용한다.
- live repo 테스트를 실행한다.
- fresh action catalog를 다시 로드한다.
- proposal의 `action_id`가 catalog에 실제로 등록됐는지 확인한다.
- smoke-test 가능한 action이면 즉시 실행해서 성공 여부를 확인한다.
- 성공한 변경을 git commit으로 남긴다.
- activation manifest와 patch를 `artifacts/activations/<work_id>/`에 보관한다.
- `NEUROKERNEL_ACTIVATION_RELOAD_COMMAND`가 설정되어 있으면 지연 restart를 예약한다.
- work item을 `completed`로 전환한다.
- 연결된 capability proposal을 `active`로 전환한다.
- reload 명령이 설정되어 있지 않으면 `service_reload_required=true`를 반환한다.

실패 시:

- apply 실패면 live repo를 건드리지 않는다.
- test 실패면 patch를 reverse apply로 되돌린다.
- catalog/smoke 검증 실패면 commit 전에 patch를 되돌린다.
- work item을 `reviewing`으로 돌린다.
- 실패 stage와 command tail을 work event에 남긴다.

CLI:

```bash
python -m neurokernel_seed.cli activate-work-item <work_id> --db data/harness.db --project-root .
```

## SelfPatchWorker v2 Contract

SelfPatchWorker v2 is the Codex development-worker adapter.

Runtime contract:

- It creates an isolated workspace before Codex runs.
- `NEUROKERNEL_SELF_PATCH_ISOLATION=auto` uses `git worktree` for git repos and copy mode for non-git projects.
- It never writes directly into the live repo.
- It records a reproducible artifact bundle under `artifacts/self_patch/<job_id>/`.

Required artifacts:

- `proposal.patch`
- `summary.json`
- `contract.json`
- `evidence.json`
- `summary.md`
- `codex_stdout.txt`
- `codex_stderr.txt`
- `test_stdout.txt`
- `test_stderr.txt`
- `diff_check_stdout.txt`
- `diff_check_stderr.txt`

Status contract:

- `patch_ready`: patch exists, tests pass, and `git diff --check` passes.
- `test_failed`: patch exists but tests fail.
- `diff_check_failed`: patch exists but patch formatting/whitespace validation fails.
- `no_patch`: Codex produced no repository diff.

API:

```http
POST /work-items/{work_id}/activate
```

Discord:

```text
work show <work_id>
```

상태가 `waiting_approval`이면 `패치 장착 승인` 버튼이 붙는다.
버튼으로 장착하면 Discord 응답에 catalog 등록과 smoke-test 통과 여부가 함께 표시된다.

Orange Pi 지연 재시작 기본 명령:

```bash
bash tools/restart_orangepi_stack_deferred.sh
```

이 스크립트는 `systemd-run --user --on-active=2s`로 현재 API 응답이 끝난 뒤 `neurokernel-stack.service`를 재시작하게 예약한다.

## Dynamic Action Registry

새 action은 Python 코드에 직접 박지 않고 registry 파일로 등록할 수 있다.

기본 registry:

```text
registry/actions.json
```

환경값:

```bash
NEUROKERNEL_ACTION_REGISTRY=registry/actions.json
```

검증:

```bash
python -m neurokernel_seed.cli validate-action-registry registry/actions.json
```

registry entry 조건:

- `schema_version`은 `neurokernel-action-registry-v1`이어야 한다.
- `action_id`는 lower snake case여야 한다.
- 동적 executor는 현재 `readonly_system`, `benchmark`, `readonly_command`만 허용한다.
- `readonly_command`는 low-risk/read-only action만 허용한다.
- shell 문자열 실행은 금지하고 argv list만 허용한다.
- 동적 action은 `test_plan`이 없으면 로드되지 않는다.

현재 포함된 registry action:

- `get_cpu_usage`: `/proc/stat` 기반 CPU 사용률 조회. Orange Pi/Linux에서 동작하고, `/proc/stat`이 없는 환경에서는 `available=false`를 반환한다.

## Core API

```bash
python -m neurokernel_seed.cli serve-core-api --host 127.0.0.1 --port 8765 --db data/harness.db --project-root .
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
- `POST /work-items/{work_id}/activate`

## Worker

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

## 운영 원칙

- fallback으로 조용히 성공한 척하지 않는다.
- 실패하면 실패 stage와 이유를 남긴다.
- live repo 수정은 activation 승인 뒤에만 한다.
- 테스트 통과 전에는 proposal을 active로 올리지 않는다.
- 비밀값 조회, 배포, git push는 별도 승인 게이트가 필요하다.
