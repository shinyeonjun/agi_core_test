# NeuroKernel Agent Harness

이 저장소의 구현 워커는 즉흥 코딩 세션이 아니라 검증 가능한 하네스 안에서 동작한다.
사람에게 보여주는 설명과 구현 워커에게 주는 지시는 분리한다.

## 기본 원칙

- live repo를 직접 고치지 않는다. 구현 워커는 격리 workspace에서 patch를 만든다.
- 승인된 work item 범위만 구현한다.
- fallback, 가짜 성공, 조용한 조건부 우회, 하드코딩으로 능력을 흉내 내지 않는다.
- 실패하면 실패 stage, 원인, 다음 재시도 근거를 artifact로 남긴다.
- 테스트가 통과하고 `git diff --check`가 통과하기 전에는 장착 후보가 아니다.
- world 모델과 runtime_action 모델의 책임을 섞지 않는다.

## 워커 문서

구현 워커는 아래 문서를 함께 따라야 한다.

- `docs/worker_harness/implementation_worker.md`
- `docs/worker_harness/self_patch_contract.md`
- `docs/worker_harness/world_runtime_usage.md`
- `docs/worker_harness/retry_and_failure.md`

## 승인 경계

다음 작업은 사람이 승인하기 전까지 수행하지 않는다.

- live repo patch 장착
- git push
- OrangePi 서비스 재시작 또는 배포
- 시스템 패키지 설치
- 시크릿, 토큰, 쿠키, SSH key 조회 또는 출력
- destructive filesystem 변경
