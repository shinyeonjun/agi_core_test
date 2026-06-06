# NeuroKernel AGI Seed

Orange Pi 5에서 24시간 돌리는 AGI seed 하네스입니다. 목표는 LLM 흉내 챗봇이 아니라, 언어기관(LLM)과 실행 코어를 분리하고, 작업 명세, 안전 게이트, 기억, 큐, 개발 워커, 검증, 장착 루프를 하나의 검증 가능한 시스템으로 묶는 것입니다.

## 현재 상태

- Core API: FastAPI로 task, memory, work, capability, activation을 노출합니다.
- Discord bot: prefix 없는 대화형 입구로 Core API에 연결됩니다.
- Memory/Trace: SQLite에 작업, 실행 결과, 실패 기록, 대화 기억, 선호를 저장합니다.
- Safety Gate: read-only 작업은 실행하고, 위험 작업은 승인 흐름으로 보냅니다.
- Redis Queue/Worker: self-patch와 external-work 작업을 durable queue로 처리합니다.
- SelfPatchWorker: 격리 workspace에서 Codex 개발 세션을 실행하고 `proposal.patch`를 만듭니다.
- Activation Pipeline: 승인된 patch를 live repo에 적용하고, 테스트하고, commit/archive 후 active로 승격합니다.
- Activation Verification: active 승격 전에 fresh catalog reload와 action smoke-test를 수행합니다.
- Dynamic Action Registry: 새 action을 `registry/actions.json`으로 등록하고 executor adapter에 연결합니다.
- World Model: MicroWorld에서 ONNX/RKNN 모델과 gate ablation 검증이 연결되어 있습니다.

## 아직 남은 것

- Activation 후 Orange Pi service 자동 restart/reload.
- 새 action을 Discord에서 자동 smoke-test한 뒤 최종 승격.
- Project/Research/Data worker.
- Training worker.
- registry update를 self-patch artifact에서 자동 promote하는 루프.
- GitHub push/PR 자동화 승인 게이트.

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

## 환경값

```bash
cp .env.example .env
```

`.env`에는 Discord token, channel id, allowed user ids, Redis URL, activation 설정을 넣습니다. `.env`, `data/`, `artifacts/`, 모델 파일(`*.pt`, `*.onnx`, `*.rknn`)은 Git에 올리지 않습니다.

## Action Registry

기본 registry:

```text
registry/actions.json
```

검증:

```bash
python -m neurokernel_seed.cli validate-action-registry registry/actions.json
```

현재 포함된 동적 action:

- `get_cpu_usage`: `/proc/stat` 기반 CPU 사용률 조회.

동적 action은 `test_plan`이 없으면 로드되지 않고, `readonly_command`는 shell 없이 argv list만 실행합니다.

## Orange Pi 실행

수동 실행:

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
bash tools/run_orangepi_stack.sh
```

user systemd service:

```bash
cd ~/projects/neurokernel-agi-seed
bash tools/install_orangepi_user_service.sh
systemctl --user status neurokernel-stack.service --no-pager
```

로그:

```bash
journalctl --user -u neurokernel-stack.service -f
```

## Activation 흐름

```text
Discord 요청
-> Capability Proposal
-> 개발 후보 승인
-> Redis self_patch job
-> SelfPatchWorker가 proposal.patch 생성
-> work show <work_id>
-> 패치 장착 승인
-> git apply --check
-> git apply
-> pytest
-> fresh catalog reload
-> action smoke-test
-> git commit
-> activation archive 저장
-> proposal active
```

CLI:

```bash
python -m neurokernel_seed.cli activate-work-item <work_id> --db data/harness.db --project-root .
```

## 운영 원칙

- fallback으로 성공한 척하지 않습니다.
- 실패하면 stage, command tail, rollback 결과를 기록합니다.
- live repo 수정은 activation 승인 뒤에만 합니다.
- 테스트 통과 전에는 proposal을 active로 올리지 않습니다.
- 비밀값 조회, 배포, git push는 별도 승인 게이트가 필요합니다.
