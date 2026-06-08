# Retry And Failure Contract

실패는 정상적인 신호다.
문제는 실패를 숨기거나 같은 실패를 반복하는 것이다.

## 재시도 입력

재시도 워커는 아래 정보를 먼저 읽는다.

- 이전 status
- 실패한 test tail
- `git diff --check` 결과
- 이전 patch excerpt
- changed files
- failure analysis
- retry directives

## 재시도 원칙

- 실패한 테스트에서 시작한다.
- 이전 patch 모양을 그대로 반복하지 않는다.
- 실패 원인이 테스트 노후화라는 근거가 있을 때만 테스트 기대값을 바꾼다.
- 재시도는 불확실성을 줄여야 한다.
- 통과하지 못하면 더 작은 blocker와 정확한 missing contract를 남긴다.

## 실패 artifact 기준

실패 artifact에는 최소한 아래가 들어가야 한다.

- primary failure
- 실패 stage
- 관련 파일
- 테스트 또는 diff check tail
- 다음 재시도에서 먼저 볼 지점
- 사람이 승인해야 할지, 워커가 재시도해도 되는지
