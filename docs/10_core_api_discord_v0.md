# Core API + Discord Bot v0

Discord 채널을 NeuroKernel Harness 입구로 붙이는 구성이다.

```text
Discord channel
  -> Discord bot
  -> Core API
  -> HarnessService
  -> Safety Gate
  -> Read-only executor / safe benchmark executor
  -> SQLite memory + traces
```

Discord bot은 시스템 명령을 직접 실행하지 않는다. 모든 실행은 Core API를 거쳐 Harness 안전 게이트와 SQLite 기록을 통과한다.

## 설치

Orange Pi:

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
python -m pip install -e ".[api,discord]"
```

Windows 개발 환경:

```powershell
cd /d D:\agi_seed\neurokernel-agi-seed
venv\Scripts\activate
python -m pip install -e ".[api,discord]"
```

## Core API 실행

Orange Pi:

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
bash tools/run_core_api.sh
```

직접 실행:

```bash
python -m neurokernel_seed.cli serve-core-api \
  --host 127.0.0.1 \
  --port 8765 \
  --db data/harness.db \
  --project-root .
```

확인:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/status
```

## Discord Bot 실행

토큰과 실제 채널/사용자 ID는 `.env`나 현재 셸 환경변수에만 둔다.

```bash
cd ~/projects/neurokernel-agi-seed
. venv/bin/activate
export DISCORD_BOT_TOKEN="put_discord_bot_token_here"
export DISCORD_CHANNEL_ID="put_discord_channel_id_here"
export DISCORD_ALLOWED_USER_IDS="put_user_id_here,put_another_user_id_here"
export NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX="1"
export NEUROKERNEL_BOT_AUTO_DO_LOW_RISK="1"
bash tools/run_discord_bot.sh
```

직접 실행:

```bash
python -m neurokernel_seed.cli serve-discord-bot \
  --core-url http://127.0.0.1:8765 \
  --channel-id "$DISCORD_CHANNEL_ID"
```

Discord Developer Portal에서는 bot의 `MESSAGE CONTENT INTENT`가 켜져 있어야 일반 메시지를 읽을 수 있다.

## 대화 모드

기본 목표는 `!nk` 명령어 봇이 아니라 자연어 대화형 인터페이스다.

예시:

```text
오렌지파이 메모리 상태 어때?
모델 파일 잘 있어?
최근 실패 trace 있어?
cpu 사용률 보는 능력 후보로 올려줘
```

봇은 지정 채널과 허용된 사용자에게만 반응한다. `NEUROKERNEL_BOT_REPLY_WITHOUT_PREFIX=1`이면 허용된 사용자의 일반 메시지에도 반응한다. `NEUROKERNEL_BOT_AUTO_DO_LOW_RISK=1`이면 메모리, 디스크, 온도, artifact, trace 같은 low-risk read-only 조회는 자동 실행한다.

벤치마크, 재부팅, 파일 삭제, 배포, 개발 작업은 자동 실행하지 않는다. 이런 작업은 승인 흐름을 타야 한다. DM은 기본 차단이다.

## 현재 안전 경계

- 허용: 상태 조회, 디스크/메모리/온도 조회, artifact 목록, 최근 trace, 안전 벤치마크
- 승인 필요: 파일 쓰기, 재부팅, 삭제, 스크립트 실행, git push, self-patch
- 금지: 비밀값 조회, 토큰 출력, 임의 외부 전송

## 다음 단계

1. LLM TaskSpec adapter: 자연어를 구조화된 TaskSpec으로 변환
2. Approval UI: 중간 이상 위험 작업을 Discord 버튼으로 승인/거절
3. Failure trace reporter: 실패 원인, 재현 명령, 다음 실험 제안 자동 요약
4. Codex worker bridge: 승인된 개발 task만 Codex 작업자로 전달
