# World/Runtime 2슬롯 데이터 계약

목표는 `모델에 필요한 데이터 = 사용하면서 축적되는 데이터`를 맞추는 것이다. 없는 데이터를 조용히 추정하거나, fallback 경로로 성공처럼 포장하지 않는다.

## 모델 슬롯

| 슬롯 | 활성 파일 | 역할 | 학습 데이터 |
|---|---|---|---|
| world | `artifacts/current_world_model.onnx` | 미니월드/slot transition에서 action이 다음 상태, 보상, 종료, 진행도, 정보이득에 어떤 영향을 주는지 예측 | `data/model_ready/features*.jsonl` |
| runtime_action | `artifacts/current_runtime_action_model.pt` | 실제 harness 실행 환경에서 후보 action의 성공, 보상, 실행 시간, 실패 여부를 예측 | `data/model_ready/runtime_features*.jsonl` |

world 슬롯은 환경 전이 모델이다. runtime_action 슬롯은 오렌지파이/harness 운영 action 모델이다. 둘은 같은 모델이 아니고, 같은 데이터로 학습하지 않는다.

## 사용 중 축적되는 원천 데이터

`data/harness.db`에 아래 데이터가 쌓인다.

| 테이블 | 의미 | runtime 학습 사용 |
|---|---|---|
| `tasks` | 사용자 요청을 검증된 task로 바꾼 기록 | 입력 context |
| `action_decisions` | 후보 action, 선택 action, 안전 판단 | 후보 집합/선택 정책 |
| `execution_results` | 실제 실행 action의 성공/실패/result/error | 선택 action의 target |
| `traces` | 실패 bucket과 원인 분석 | 실패 target/진단 보조 |
| `experiences` | 한 decision/run 단위의 before/after/outcome/학습 mask | runtime 학습 계약의 중심 |
| `experience_candidates` | 후보 action별 selected/executed/result_known/target_mask | 후보별 feature row와 target mask |

핵심은 `experience_candidates.target_mask_json`이다. 실행한 후보만 결과를 안다. 실행하지 않은 후보는 실패로 라벨링하지 않고 target mask를 0으로 둔다.

## Runtime 학습 흐름

```text
사용 task 실행
-> experiences / experience_candidates 기록
-> runtime_replay*.jsonl 추출
-> runtime_features*.jsonl 후보별 feature row 생성
-> target_mask가 있는 값만 loss에 반영
-> runtime_action_model.pt 후보 모델 학습
-> `nk deploy-runtime`으로 current_runtime_action_model.pt 슬롯 별도 활성화
```

현재 decision policy가 runtime model을 쓰지 않으면 `model_used=false`, `model_unavailable_reason=no_current_runtime_action_model`로 명시한다. 이것은 fallback이 아니라 상태 기록이다.

활성화 명령:

```powershell
nk deploy-runtime --model artifacts/runtime_action_model.pt
nk current
nk status
```

## 초기 모델 구성

초기 world 모델은 기존 slot transition 데이터(`neurokernel-slot-transition-v2`)로 구성한다. 이 데이터는 사용 중 harness DB에서 직접 생기는 데이터가 아니라 미니월드/slot 시뮬레이션 데이터다.

초기 runtime_action 모델은 `data/harness.db`에 실제 실행 경험이 충분히 쌓인 뒤 생성한다. 데이터가 부족하면 training-ready gate가 실패해야 한다. smoke/candidate checkpoint는 활성 슬롯으로 간주하지 않는다.

## 아직 부족한 데이터

현재 runtime 사용 데이터가 모두 성공이면 실패 예측이 약하다. 실패 케이스는 숨기지 말고 `execution_results`, `traces`, `experience_candidates`에 그대로 남겨야 한다.

후보 action 전체를 완전히 랭킹하려면 같은 상태에서 여러 action 결과를 아는 counterfactual 데이터가 필요하다. 지금 구조는 실행한 후보의 결과만 정답으로 쓰고, 미실행 후보는 mask 0으로 둔다.

## 완료 기준

- `nk current`/`nk status`가 world와 runtime_action 슬롯을 분리해서 보여준다.
- runtime replay row가 `lineage.experience_id`, `experience`, `candidate_outcomes`를 가진다.
- runtime feature row는 후보 action별로 생성된다.
- 실행 결과를 모르는 후보는 target vector를 학습 loss에 반영하지 않는다.
- runtime model manifest는 `model_slot=runtime_action_model`, `data_origin=runtime_experience_log`를 가진다.
