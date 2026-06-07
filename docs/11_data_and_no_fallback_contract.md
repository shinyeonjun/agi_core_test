# 데이터와 무조용 대체 금지 계약

NeuroKernel의 학습 데이터는 대화 원문이 아니라 검증된 상태, 후보 행동, 선택 행동, 실제 결과, 실패 원인, 검증 지표다.

목표는 실패를 감추지 않고 다음 개선 재료로 바꾸는 것이다. 데이터가 부족하면 부족하다고 기록하고, 검증을 통과하지 못하면 학습 후보로 올리지 않는다.

## 금지

- 조용한 대체 경로: 실패했는데 다른 경로로 몰래 바꿔 성공처럼 기록하지 않는다.
- 문장별 하드코딩 라우팅: 특정 문장이나 단어를 `if text == ...` 형태로 기능에 직접 연결하지 않는다.
- 키워드 패턴 때우기: 단어 하나만 보고 결론, 선호, 능력, 모델 위치를 추정하지 않는다.
- 데이터 없는 추정: 모델 위치, 벤치 결과, 작업 상태, 배포 상태를 실제 조회 없이 말하지 않는다.
- 기존 모델 자동 덮어쓰기: 새 모델은 항상 release 후보로 만들고, 활성화는 명시적으로 한다.
- raw secret 저장: token, password, API key, `.env`, SSH key를 학습 데이터에 넣지 않는다.

## 허용

- schema 검증
- manifest 검증
- action id 검증
- feature dimension 검증
- split 검증
- 안전 게이트
- 상태 전이
- 데이터 품질 gate
- 실패 전파와 failure trace 기록

조건문 자체가 문제가 아니다. 조건문이 데이터 검증, 안전 경계, 상태 전이를 위해 쓰이면 허용된다. 조건문이 결정을 숨기거나 성공처럼 꾸미기 위해 쓰이면 금지다.

## 저장 데이터

### Runtime Memory

경로:

```text
data/harness.db
```

주요 테이블:

| 테이블 | 의미 | 학습 사용 |
|---|---|---|
| `tasks` | 사용자 요청을 task 명세로 바꾼 기록 | 후보 |
| `task_events` | task 상태 전이 기록 | 후보 |
| `action_decisions` | 후보 행동, 선택 행동, 안전 판단, gate trace | 사용 |
| `execution_results` | 실제 실행 결과, 성공/실패, redacted 출력 | 사용 |
| `traces` | 실패 bucket, 실제/기대 결과, 분석 | 사용 |
| `work_items` | 개발/외부 작업 단위 | 후보 |
| `work_jobs` | worker 실행 job 상태 | 후보 |
| `work_events` | work item 상태 기록 | 후보 |
| `work_notes` | worker가 남긴 요약 note | 후보 |
| `user_preferences` | 명시적으로 저장된 출력 선호 | Core 학습 아님 |
| `conversation_messages` | redacted 대화 기록 | Core 학습 아님 |

### Model Ready Data

경로:

```text
data/model_ready/
```

slot world model 학습 데이터:

| 파일 | 의미 |
|---|---|
| `features*.jsonl` | slot transition 모델 입력/타깃 벡터 |
| `features*.jsonl.manifest.json` | schema, dim, vocab, split, mask |
| `counterfactual*.jsonl` | 같은 상태에서 여러 후보 행동을 비교한 결과 |

runtime 학습 후보 데이터:

| 파일 | 의미 |
|---|---|
| `runtime_replay*.jsonl` | harness DB에서 검증 추출한 실제 task/action/result 기록 |
| `runtime_replay*.jsonl.manifest.json` | runtime replay schema, row/action 통계, lineage |
| `runtime_features*.jsonl` | runtime replay를 학습 가능한 벡터 row로 바꾼 데이터 |
| `runtime_features*.jsonl.manifest.json` | runtime feature schema, vocab, dim, split, target layout |

slot 데이터와 runtime 데이터는 바로 섞지 않는다. 서로 다른 모델 목적을 가진다.

## Runtime Replay ETL

스키마:

```text
neurokernel-runtime-action-v1
```

row 핵심 구조:

```json
{
  "schema_version": "neurokernel-runtime-action-v1",
  "task": {
    "task_id": "...",
    "goal_redacted": "...",
    "target": "orangepi5",
    "status": "completed",
    "risk_level": "low",
    "allowed_actions": ["list_artifacts"]
  },
  "decision": {
    "candidate_actions": [{"action_id": "list_artifacts"}],
    "chosen_action": {"action_id": "list_artifacts"},
    "safety_decision": {"decision": "allow"},
    "model_score": {},
    "gate_trace": {}
  },
  "execution": {
    "action_id": "list_artifacts",
    "success": true,
    "result": {},
    "error_type": null
  },
  "outcome": {
    "success": true,
    "reward": 1.0,
    "failure_bucket": null,
    "failure_reason": null
  }
}
```

명령:

```powershell
python -m neurokernel_seed.cli run-runtime-replay-etl --db data/harness.db --out data/model_ready/runtime_replay.jsonl
python -m neurokernel_seed.cli validate-runtime-replay data/model_ready/runtime_replay.jsonl
python -m neurokernel_seed.cli check-runtime-replay-gates data/model_ready/runtime_replay.jsonl
```

짧은 nk 명령:

```powershell
nk 사용데이터
```

ETL 계약:

- 원천 DB 오류와 테이블 row count를 manifest에 남긴다.
- 각 row에 `row_id`와 `lineage`를 남긴다.
- JSONL은 임시 파일에 먼저 쓴다.
- schema validation과 dataset gate가 통과해야 최종 파일로 교체한다.
- 실패하면 기존 최종 파일을 건드리지 않고 `*.failed.<run_id>` 격리 파일로 남긴다.
- 최종 파일에는 `sha256`과 byte size를 manifest에 기록한다.

## Runtime Feature Export

스키마:

```text
neurokernel-runtime-action-feature-v1
```

입력은 자연어 원문이 아니라 구조화된 실행 문맥과 제한된 통계값이다.

입력 구성:

```text
action onehot
+ source onehot
+ target onehot
+ task status onehot
+ risk onehot
+ safety decision onehot
+ numeric context features
```

numeric context features:

```text
requires_approval
candidate_count
allowed_action_count
chosen_in_candidates
step
goal_chars
goal_words
goal_digits
goal_non_ascii_ratio
has_model_score
has_gate_trace
```

타깃:

```text
success
reward
duration_seconds_log1p
failure_present
```

명령:

```powershell
python -m neurokernel_seed.cli export-runtime-features data/model_ready/runtime_replay.jsonl --out data/model_ready/runtime_features.jsonl
python -m neurokernel_seed.cli validate-runtime-features data/model_ready/runtime_features.jsonl
python -m neurokernel_seed.cli check-runtime-feature-gates data/model_ready/runtime_features.jsonl
```

짧은 nk 명령:

```powershell
nk 사용특징
```

현재 runtime feature exporter는 선택된 실제 행동의 결과를 학습 후보 row로 만든다. 아직 같은 상태에서 여러 후보 행동을 모두 비교하는 counterfactual ranking 데이터는 아니다. 그 단계는 다음 exporter에서 다룬다.

학습 준비 gate:

- schema가 유효해야 한다.
- 최소 row 수를 만족해야 한다.
- 최소 action 종류 수를 만족해야 한다.
- `train`/`test` split이 둘 다 있어야 한다.
- target vector가 모두 존재해야 한다.

성공/실패 다양성 부족이나 action 다양성 부족은 경고로 남긴다. 경고는 데이터를 꾸미지 않고 현재 부족한 점을 드러내기 위한 신호다.

## Runtime Action Model

스키마:

```text
neurokernel-runtime-action-model-v1
```

입력:

```text
runtime_features.jsonl input_vector
```

타깃:

```text
success
reward
duration_seconds_log1p
failure_present
```

손실:

```text
success: BCEWithLogits
reward: SmoothL1
duration_seconds_log1p: SmoothL1
failure_present: BCEWithLogits
```

명령:

```powershell
python -m neurokernel_seed.cli train-runtime-action-model --features data/model_ready/runtime_features.jsonl --out artifacts/runtime_action_model.pt
python -m neurokernel_seed.cli eval-runtime-action-model --checkpoint artifacts/runtime_action_model.pt --features data/model_ready/runtime_features.jsonl
```

짧은 nk 명령:

```powershell
nk 사용학습
```

현재 runtime action model은 실제로 선택된 행동의 결과를 예측한다. 아직 후보 행동 전체를 랭킹하는 모델은 아니다. 따라서 이 모델은 "실제 하네스 실행 결과 예측기"로 잠그고, action ranking은 별도 `runtime ranking dataset exporter`에서 다룬다.

## 선호 저장 계약

선호는 raw 문장 키워드로 추출하지 않는다.

허용 경로:

```text
사용자 자연어
-> LanguageOrgan preference_update intent
-> preference intent schema 검증
-> 허용 key/scope/source/confidence 검증
-> user_preferences 저장
```

허용 key:

- `response_length`
- `tone`
- `technical_depth`
- `avoid_internal_terms`
- `avoid_emoji`
- `language`
- `avoid_phrases`

선호 데이터는 언어기관 출력 스타일에만 쓴다. 안전 판단, action 허용 여부, 승인 필요 여부를 바꾸지 않는다.

## 학습 데이터 승격 흐름

```text
raw event
-> redacted event
-> validated replay
-> feature export
-> dataset gate
-> train
-> benchmark
-> compare
-> release
-> activate
```

각 단계가 실패하면 다음 단계로 넘어가지 않는다.

## 현재 모델별 데이터

### Slot World Model

입력:

```text
task/visibility features
+ slot features
+ current state/belief
+ candidate action
```

타깃:

```text
next_state
+ reward
+ done
+ local_success
+ progress_delta
+ information_gain
```

용도:

- 미니월드에서 후보 행동 결과 예측
- Action Gate에서 model_only/prior_only/hybrid/hybrid_veto 비교
- model-needed probe에서 모델 기여 검증

### Runtime Action Model 후보

아직 학습 모델은 아니다. 지금 단계는 데이터 형식과 gate를 만든 것이다.

입력 후보:

```text
task metadata
+ allowed actions
+ safety decision
+ decision context
+ chosen action
+ execution context
```

타깃 후보:

```text
success
+ reward
+ failure present
+ duration
```

이 모델의 목적은 실제 하네스에서 어떤 작업을 바로 실행할지, 개발 작업으로 넘길지, 진단을 먼저 할지 판단하는 것이다.

## 다음 과제

- runtime counterfactual/ranking dataset exporter
- work pipeline dataset exporter
- failure trace dataset exporter
- research/data worker dataset exporter
- release 비교표에 runtime benchmark 추가
- 활성 모델 변경 후 runtime replay 성능 추적
