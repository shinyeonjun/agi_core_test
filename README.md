# NeuroKernel AGI Seed

Orange Pi 5에서 24시간 돌리는 AGI seed 하네스입니다. 목표는 "LLM 느낌의 챗봇"이 아니라, 언어기관(LLM)과 실행 코어를 분리하고, 작업 명세, 안전 게이트, 기억, 평가, 워커, 학습 루프를 하나의 검증 가능한 시스템으로 묶는 것입니다.

## 현재 상태

- Core API: 로컬 FastAPI 서버로 작업 생성, 실행, 상태 조회를 제공합니다.
- Discord bot: 지정 채널/허용 사용자에게 대화형으로 반응하고 Core API에 연결됩니다.
- Memory/Trace: SQLite에 작업, 실행 결과, 실패 기록을 남깁니다.
- Safety Gate: 읽기 작업은 바로 실행하고, 쓰기/위험 작업은 승인 흐름으로 보냅니다.
- Capability Proposal: 없는 능력은 바로 거짓 실행하지 않고, 개발 후보로 제안합니다.
- Queue/Worker: Redis 기반 큐로 self-patch/external-work 작업을 분리할 수 있습니다.
- World Model: MicroWorld 벤치마크에서 ONNX/RKNN 모델을 연결하고 gate ablation으로 모델 기여를 검증했습니다.
- Orange Pi 배포: user systemd service로 Core API와 Discord bot을 상시 구동합니다.

## 아직 남은 것

- Self-Patch worker: 후보 능력을 코드 변경, 테스트, 패치 제안까지 연결하는 워커.
- Activation pipeline: 승인된 패치를 배포, 리로드, 검증하는 절차.
- Research/Data worker: 논문/자료 수집, 요약, 데이터셋 변환 파이프라인.
- Training worker: 노트북 학습 산출물을 Orange Pi 런타임 모델로 승격하는 자동화.
- 장기 평가: 실제 Discord 사용 로그 기반 실패 원인 분석과 회귀 벤치마크.

## 개발 환경

Windows:

```powershell
cd D:\agi_seed\agi_core_test
python -m venv venv
venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev,model,api,discord,queue]"
python -m pytest -q
```

Linux/Orange Pi:

```bash
cd ~/projects/neurokernel-agi-seed
python3 -m venv venv
. venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,api,discord,queue]"
python -m pytest -q
```

## 환경 변수

`.env.example`을 복사해서 `.env`를 만들고 실제 토큰/채널/사용자 ID를 넣습니다.

```bash
cp .env.example .env
```

주의: `.env`, `data/`, `artifacts/`, 모델 파일(`*.pt`, `*.onnx`, `*.rknn`)은 Git에 올리지 않습니다.

## Orange Pi 실행

수동 실행:

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
bash tools/run_orangepi_stack.sh
```

user systemd 서비스 설치:

```bash
cd ~/projects/neurokernel-agi-seed
bash tools/install_orangepi_user_service.sh
systemctl --user status neurokernel-stack.service --no-pager
```

로그 확인:

```bash
journalctl --user -u neurokernel-stack.service -f
```

## Discord 사용

봇은 허용된 사용자 메시지만 처리합니다. 현재 목표는 명령어 중심 CLI가 아니라 자연어 대화형 인터페이스입니다. 내부적으로는 자연어를 Core가 이해하는 작업 명세로 바꾸고, 실행 결과를 다시 사람 말로 요약합니다.

읽기/조회 작업은 바로 실행할 수 있고, 파일 쓰기, 배포, 삭제, 외부 전송, 비밀값 조회 같은 작업은 승인 흐름을 타야 합니다.

## 저장소 운영 원칙

- 공개 저장소에는 코드, 문서, 스키마, 테스트만 올립니다.
- 학습 데이터, 런타임 DB, 로그, 모델 산출물은 로컬/Orange Pi에만 둡니다.
- fallback으로 성공한 척하지 않습니다. 실패하면 실패 원인과 필요한 다음 액션을 기록합니다.
- 하드코딩 규칙을 늘리기보다 schema, evaluator, trace, worker contract로 일반화합니다.
