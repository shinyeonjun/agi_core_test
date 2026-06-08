# World Runtime Usage Contract

world 모델과 runtime_action 모델은 같은 목적의 모델이 아니다.
구현 워커는 둘을 섞어서 하나의 만능 점수기로 만들면 안 된다.

## world 모델

world 모델은 상태 전이를 예측한다.

- 현재 상태 인코딩
- 후보 행동 이후의 다음 상태
- 위험 변화
- 보상 또는 품질 변화
- multi-step rollout의 중간 근거

world 모델은 미래 시뮬레이터다.
벤치 전용 장식물이 아니라 planner가 후보를 평가할 때 쓰는 예측기여야 한다.

## runtime_action 모델

runtime_action 모델은 지금 실행 가능한 행동 후보를 랭킹한다.

- 실제 OrangePi 사용 중 누적된 행동 데이터
- candidate_set 안의 실행/비실행 후보 outcome
- dry-run, readonly probe, 안전한 counterfactual 결과
- 성공률, 보상, 실패 복구 가능성

runtime_action 모델은 정책 선택기다.
실제로 누적 가능한 데이터만 학습 데이터로 사용한다.

## planner 연결 원칙

1. 후보 행동을 만든다.
2. 안전 gate로 위험 후보를 제거한다.
3. world 모델로 다음 상태, 위험, 보상을 예측한다.
4. runtime_action 모델로 후보를 랭킹한다.
5. 실행 또는 승인 요청을 결정한다.
6. 결과를 replay, feature, world 후보 데이터로 기록한다.

학습 데이터 범위를 늘릴 때는 OrangePi 쪽에서 같은 데이터를 실제로 누적할 수 있어야 한다.
