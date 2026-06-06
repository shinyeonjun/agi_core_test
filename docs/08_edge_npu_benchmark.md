# Edge NPU Benchmark Protocol

이 문서는 Orange Pi 5에서 ONNX CPU와 RK3588 NPU 실행을 같은 기준으로 비교하기 위한 절차다.

## 현재 기준선

- 모델: `artifacts/world_model_slot_v2_model_needed_v3.onnx`
- 구조: `Gemm x3`, `Relu x2`
- 입력: `253`
- 출력: `9`
- Orange Pi ONNX CPU predictor 중앙값: 약 `0.14ms`
- Orange Pi raw ONNX batch=1 중앙값: 약 `0.015ms`

현재 모델은 매우 작다. 그래서 NPU가 실제로 이기려면 RKNN 호출 오버헤드까지 포함해도 위 기준보다 빨라야 한다.

## 1. ONNX CPU 기준 측정

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
python -m neurokernel_seed.cli benchmark-runtime \
  --model artifacts/world_model_slot_v2_model_needed_v3.onnx \
  --backend onnx_cpu \
  --iterations 2000 \
  --samples 256 \
  --batches 1 8 32 128 \
  --out artifacts/benchmark/runtime_speed_orangepi_onnx_cpu.json
```

## 2. RKNN 변환용 static ONNX 생성

RKNN 변환은 동적 batch ONNX보다 정적 batch ONNX가 안정적이다.

```cmd
cd /d D:\agi_seed\neurokernel-agi-seed
venv\Scripts\python.exe -m neurokernel_seed.cli export-world-model-onnx ^
  --checkpoint D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3.pt ^
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3_static.onnx ^
  --static-batch
```

## 3. ONNX to RKNN 변환

RKNN 변환은 Rockchip RKNN-Toolkit2가 설치된 Linux/Python 환경에서 실행한다.

```bash
python tools/convert_onnx_to_rknn.py \
  --onnx artifacts/world_model_slot_v2_model_needed_v3_static.onnx \
  --out artifacts/world_model_slot_v2_model_needed_v3_static.rknn \
  --target-platform rk3588
```

첫 검증에서는 `--quantize`를 사용하지 않는다. 정확도/행동 선택 일치가 확인된 뒤 INT8 양자화를 별도로 비교한다.

## 4. RKNN NPU 측정

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
python -m neurokernel_seed.cli benchmark-runtime \
  --model artifacts/world_model_slot_v2_model_needed_v3_static.rknn \
  --backend rknn_npu \
  --iterations 2000 \
  --samples 256 \
  --out artifacts/benchmark/runtime_speed_orangepi_rknn_npu.json
```

## 5. 채택 기준

NPU를 기본 런타임으로 바꾸려면 아래 기준을 모두 만족해야 한다.

- `hybrid_veto` benchmark 결과 유지
- ONNX CPU 대비 action 선택 일치율 `99%` 이상
- predictor loop 중앙값이 ONNX CPU보다 빠름
- 전체 `run-gate-ablation` 실행 시간이 실제로 감소
- RKNN 런타임 설치/배포가 자동화 가능

지금 모델 크기에서는 NPU가 필수 최적화가 아닐 수 있다. 모델이 커지거나 planning rollout이 늘어날 때 NPU 이득이 커진다.
