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
