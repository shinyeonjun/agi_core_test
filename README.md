# agent-core

Orange Pi 5에서 돌아가는 Local Stateful Agent Core입니다.

이 프로젝트는 LLM이나 딥러닝을 Core 본체로 쓰지 않습니다. Core는 SQLite, 상태 파일, 규칙 기반 점수, 목표 큐, fallback renderer로 동작합니다.

## 실행

```bash
source /home/ubuntu/agent_core/venv/bin/activate
agentctl init
agentctl state
agentctl talk "테스트"
agentctl tick
agentctl events
agentctl goals
```

## v0.1 범위

- Event log
- State manager
- Memory add/search
- Goal create/list/done
- Simple drive scoring
- Decision object
- Fallback renderer
- Basic validator
- `agentctl talk/tick/state/events/goals/memory`

## v0.2-pre 범위

Approval Queue + Policy Engine을 추가했습니다. 위험한 행동은 실행하지 않고, 분류하거나 승인 큐에 저장만 합니다.

```bash
agentctl policy-check "df -h"
agentctl policy-check "sudo systemctl enable agentd"
agentctl approvals
agentctl approve <id>
agentctl reject <id>
```

## 보안 원칙

- sudo 자동 실행 없음
- 파일 삭제/OS 변경 자동 실행 없음
- Codex renderer 미연결
- 민감 정보는 `.env`에 두고 renderer input에 넣지 않음
- `policy-check`는 명령을 실행하지 않고 분류만 수행