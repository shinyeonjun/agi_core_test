# Training Pipeline v1

모델 학습은 노트북에서 돌리고, 결과 모델은 오렌지파이에 버전 릴리즈로 보낸다.
기존 모델은 덮어쓰지 않는다. 새 모델은 `model_releases/<run_name>` 아래에 저장되고, 검증 후 원할 때만 현재 모델로 활성화한다.

## 가장 짧은 사용법

`D:\agi_seed`에서 아래처럼 쓴다.

```cmd
nk
```

그러면 메뉴가 뜬다.

```text
1. 학습 전 확인
2. 학습 + 냉정 벤치 저장
3. 벤치 상위 모델 비교
4. 냉정 벤치만 다시 실행
5. 현재 코어 모델 벤치 저장
6. 선택 모델 오렌지파이 배포
7. 선택 모델 배포 후 현재 모델로 활성화
8. 오렌지파이 모델 목록
9. 현재 활성 모델 확인
10. 기존 배포 모델 활성화
```

바로 실행하고 싶으면 아래처럼 쓴다.

```cmd
nk check
nk train
nk bench <run_name>
nk bench-current
nk top
nk deploy <run_name>
nk deploy-use <run_name>
nk list
nk current
nk use <run_name>
```

`run_name`을 생략하면 feature 파일 이름과 UTC 시간을 섞어서 자동 생성한다.

## run name이 필요한 이유

`slot_v2_mn_v3_004` 같은 이름은 모델 릴리즈 이름이다.

필요한 이유는 세 가지다.

- 기존 모델을 덮어쓰지 않기 위해서
- 여러 모델을 나란히 보관하고 비교하기 위해서
- 새 모델이 이상하면 이전 모델로 되돌리기 위해서

평소에는 직접 안 지어도 된다. `nk train`이 자동으로 만든다.
특정 실험을 사람이 구분하고 싶을 때만 `nk train 내실험이름`처럼 지정하면 된다.

## 기본 경로

`D:\agi_seed\nk.cmd`는 기본값을 자동으로 잡는다.

```cmd
NEUROKERNEL_TRAIN_FEATURES=D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl
NEUROKERNEL_TRAIN_RUN_DIR=D:\agi_seed\artifacts\training_runs
NEUROKERNEL_EDGE_HOST=orangepi5
NEUROKERNEL_EDGE_PROJECT=/home/ubuntu/projects/neurokernel-agi-seed
```

다른 데이터셋으로 학습하고 싶으면 환경변수나 옵션으로 바꾸면 된다.

## 학습이 하는 일

`nk train`은 내부적으로 아래 단계를 한 번에 실행한다.

1. feature 파일과 manifest 확인
2. dataset gate 확인
3. CUDA 학습
4. world model 평가
5. action ranking 평가
6. ONNX export와 검증
7. gate ablation 냉정 벤치

`nk train`은 여기서 멈춘다. 바로 오렌지파이에 배포하지 않는다.
이유는 새 모델이 기존 모델보다 나은지 비교하기 전에는 운영 모델을 건드리지 않기 위해서다.

벤치가 통과하지 못하면 기본적으로 학습 run을 실패 처리한다.
실험용으로만 실패 결과까지 보고 싶을 때는 내부 CLI의 `--allow-benchmark-failure`를 명시적으로 써야 한다.

## 모델 비교

최근 학습 결과 중 상위 5개를 보려면:

```cmd
nk top
```

정렬 기준은 다음 순서다.

1. `hybrid_veto` macro success rate
2. model-needed signal group count
3. `hybrid_veto`의 prior 대비 이득

즉 단순히 “성공률 1.0”만 보는 게 아니라, prior만으로 풀린 건지 모델이 실제로 보탰는지도 같이 본다.

현재 오렌지파이에 물려 있는 core 모델도 같은 방식으로 벤치 기록을 남길 수 있다.

```cmd
nk bench-current
```

이 명령은 오렌지파이의 `current_world_model.onnx`를 노트북으로 가져와서 냉정 벤치를 돌리고, `training_runs/current_core_<release>_<time>` 폴더에 `gate_ablation_result.json`을 저장한다.
따라서 `nk top`에서 새 모델들과 같은 기준으로 비교할 수 있다.

## 배포와 활성화

학습과 비교가 끝난 뒤 선택한 모델을 오렌지파이에 배포한다.

```cmd
nk deploy <run_name>
```

배포까지 하고 바로 현재 모델로 바꾸려면:

```cmd
nk deploy-use <run_name>
```

`deploy`와 `deploy-use`는 로컬 `training_runs/<run_name>` 폴더를 기준으로 배포한다.
같은 이름의 릴리즈가 오렌지파이에 이미 있으면 실패한다.

## 현재 모델 확인과 기존 릴리즈 활성화

오렌지파이에 저장된 모델 목록:

```cmd
nk list
```

현재 활성 모델:

```cmd
nk current
```

기존 릴리즈를 현재 모델로 바꾸기:

```cmd
nk use <run_name>
```

`nk use`는 모델 파일을 새로 만들지 않고 `current.json`, `current_world_model.onnx`, `current_world_model.manifest.json` 포인터만 바꾼다.
