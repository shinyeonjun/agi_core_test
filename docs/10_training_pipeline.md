# Training Pipeline v1

`run-training-pipeline`은 모델 실험 하나를 한 번에 실행하는 명령이다.
데이터 검증, slot dataset gate, 학습, 모델 평가, action ranking, ONNX export, gate ablation을 같은 run 폴더에 기록한다.

## 기본 실행

```powershell
cd /d D:\agi_seed\agi_core_test
$env:PYTHONPATH = "src"
python -m neurokernel_seed.cli run-training-pipeline `
  --features D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl `
  --out-dir D:\agi_seed\artifacts\training_runs `
  --run-name slot_v2_model_needed_v3_auto_001 `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10 `
  --strict
```

## 학습 후 오렌지파이 릴리즈 배포

노트북에서 CUDA로 학습하고, 기존 모델 파일을 덮어쓰지 않고 오렌지파이에 새 릴리즈로 보낼 때는 `train-deploy-model`을 쓴다.

먼저 터미널에 기본 경로를 한 번 잡아둔다.

```cmd
set PYTHONPATH=src
set NEUROKERNEL_TRAIN_FEATURES=D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl
set NEUROKERNEL_TRAIN_RUN_DIR=D:\agi_seed\artifacts\training_runs
set NEUROKERNEL_EDGE_HOST=orangepi5
set NEUROKERNEL_EDGE_PROJECT=/home/ubuntu/projects/neurokernel-agi-seed
```

이후 실사용 명령은 짧다.

```cmd
python -m neurokernel_seed.cli train-deploy-model slot_v2_mn_v3_004
```

원격에는 아래처럼 새 폴더가 생긴다.

```text
/home/ubuntu/projects/neurokernel-agi-seed/artifacts/model_releases/slot_v2_mn_v3_004/
```

같은 `run-name` 폴더가 이미 있으면 학습 전에 실패한다. 기존 모델은 덮어쓰지 않는다.

학습 없이 경로와 원격 release 중복만 확인하려면 dry-run을 쓴다.

```cmd
python -m neurokernel_seed.cli train-deploy-model slot_v2_mn_v3_004 --dry-run
```

현재 모델 포인터까지 바꾸고 싶을 때만 명시적으로 `--activate`를 붙인다.

```cmd
python -m neurokernel_seed.cli train-deploy-model slot_v2_mn_v3_004 --activate
```

`--activate`도 모델 파일을 덮어쓰지 않고 `current.json`, `current_world_model.onnx` 심볼릭 링크만 갱신한다.

배포는 원격의 `.incoming` 임시 폴더에서 먼저 압축 해제와 필수 파일 검증을 끝낸 뒤, 마지막에 `model_releases/<run-name>`으로 이동한다.
따라서 최종 release 폴더에는 완성된 모델만 남는다.

## 빠른 smoke 실행

게이트/학습 배선만 확인할 때는 gate ablation을 생략할 수 있다.

```powershell
python -m neurokernel_seed.cli run-training-pipeline `
  --features D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl `
  --out-dir D:\agi_seed\artifacts\training_runs `
  --run-name smoke_cuda `
  --epochs 1 `
  --batch-size 1024 `
  --device cuda `
  --skip-gate-ablation
```

## 산출물

run 폴더에는 아래 파일들이 생긴다.

- `training_pipeline_manifest.json`: 전체 진행 상태와 단계별 요약
- `feature_validation.json`: feature schema 검증 결과
- `dataset_gates.json`: slot dataset gate 결과
- `world_model.pt`: PyTorch checkpoint
- `train_result.json`: 학습 결과
- `world_model_eval_test.json`: world model test split 평가
- `action_ranking_test.json`: 행동 랭킹 평가
- `world_model.onnx`: ONNX 모델
- `onnx_export.json`: ONNX export 및 검증 결과
- `gate_ablation/`: model_only, prior_only, hybrid, hybrid_veto 비교 결과

## 실패 정책

기본 정책은 명확한 실패다.

- feature manifest가 없으면 중단한다.
- slot dataset gate가 실패하면 중단한다.
- gate ablation에서 `hybrid` 또는 `hybrid_veto`가 승격되지 않으면 중단한다.

실험 목적으로 실패한 데이터도 끝까지 흘려보고 싶을 때만 아래 옵션을 명시적으로 쓴다.

- `--allow-gate-failure`
- `--allow-benchmark-failure`

이 옵션들은 fallback이 아니라 실패 조건을 의식적으로 완화하는 실험 스위치다.
