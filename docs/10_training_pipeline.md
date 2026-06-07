# 모델 학습/배포 파이프라인

이 문서는 노트북에서 모델을 학습하고, 벤치마크로 검증한 뒤, Orange Pi에 배포하는 흐름을 정의한다.

핵심 원칙은 세 가지다.

- 기존 운영 모델을 조용히 덮어쓰지 않는다.
- 학습 결과는 벤치마크를 통과한 뒤에만 배포 후보가 된다.
- 현재 운영 모델, 후보 모델, 실패한 모델을 모두 추적 가능하게 남긴다.

## 가장 짧은 사용법

`D:\agi_seed`에서:

```cmd
nk
```

또는 직접:

```cmd
nk 확인
nk 학습
nk 순위
nk 벤치 <run_name>
nk 현재벤치
nk 배포적용 <run_name>
nk 현재
```

## 기본 흐름

```text
학습 전 확인
-> 데이터 계약 확인
-> 학습
-> 평가
-> ONNX 변환
-> gate ablation 벤치
-> 후보 비교
-> Orange Pi 배포
-> 현재 모델 포인터 갱신
```

`nk 학습`은 배포까지 자동으로 가지 않는다. 학습 모델이 기존 운영 모델보다 나쁜데 바로 붙는 사고를 막기 위해서다.

## run name이 필요한 이유

`slot_v2_mn_v3_004` 같은 이름은 모델 릴리스 이름이다.

이름이 필요한 이유:

- 기존 모델을 덮어쓰지 않기 위해
- 여러 후보 모델을 나란히 비교하기 위해
- 문제가 생기면 이전 모델로 되돌리기 위해
- 데이터, 모델, 벤치마크 결과를 한 묶음으로 추적하기 위해

`nk 학습`에서 이름을 생략하면 자동 생성된다.

## 기본 경로

기본 환경값은 `D:\agi_seed\nk.cmd` 또는 환경변수로 설정한다.

```cmd
NEUROKERNEL_TRAIN_FEATURES=D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl
NEUROKERNEL_TRAIN_RUN_DIR=D:\agi_seed\artifacts\training_runs
NEUROKERNEL_EDGE_HOST=orangepi5
NEUROKERNEL_EDGE_PROJECT=/home/ubuntu/projects/neurokernel-agi-seed
```

## 학습 데이터 계약

현재 world model은 raw 대화, raw 로그, raw Discord 메시지를 직접 학습하지 않는다.

학습 입력은 `features*.jsonl`과 그 옆의 `.manifest.json`이다.

학습에 넣을 수 있는 데이터:

- MicroWorld transition feature
- counterfactual action candidate feature
- action 결과가 명확히 라벨링된 runtime transition
- 실패 trace에서 원인/결과가 구조화된 데이터

학습에 넣으면 안 되는 데이터:

- `.env`, 토큰, 비밀값
- 개인정보가 포함된 raw 로그
- 성공/실패 라벨이 불명확한 데이터
- schema version이 없는 JSONL
- train/test/heldout split이 섞인 데이터
- 실패를 성공처럼 바꾼 synthetic label

자세한 계약은 [docs/11_data_and_no_fallback_contract.md](11_data_and_no_fallback_contract.md)를 따른다.

## 모델 비교 기준

`nk 순위`는 단순 성공률만 보지 않는다.

우선순위:

1. `hybrid_veto` 성공률
2. model-needed signal group 수
3. `hybrid_veto`가 prior-only보다 이긴 정도

즉 “규칙만으로 풀린 모델”보다 “모델 예측이 실제 행동 선택에 기여한 모델”을 더 높게 본다.

## 배포와 활성화

학습 후보를 Orange Pi에 보내기:

```cmd
nk 배포 <run_name>
```

보내고 바로 현재 모델로 적용하기:

```cmd
nk 배포적용 <run_name>
```

이미 배포된 모델로 현재 포인터만 바꾸기:

```cmd
nk 적용 <run_name>
```

현재 포인터는 Orange Pi의 `artifacts/current_world_model.onnx`와 `artifacts/current_world_model.manifest.json`이다.

## 실패 처리

실패하면 다른 모델이나 경로로 조용히 넘어가지 않는다.

예:

- feature manifest가 없으면 실패
- dataset gate가 떨어지면 실패
- benchmark가 통과하지 못하면 실패
- remote release가 이미 있으면 실패
- current model 파일이 없으면 실패

이 실패는 숨기지 않고 보고한다. 실패를 데이터로 남겨 다음 개선에 쓴다.
