# Self-Patch Contract

self-patch 워커는 live repo를 바로 수정하지 않는다.
항상 격리 workspace에서 patch 후보를 만들고, 하네스가 검증한 뒤 사람이 장착을 승인한다.

## 입력

- 승인된 work item
- queue payload
- retry context
- worker harness documents
- repository files in the isolated workspace

## 필수 산출물

- `proposal.patch`
- `summary.json`
- `contract.json`
- `evidence.json`
- `failure_analysis.json`
- `summary.md`
- `worker_harness.md`
- `worker_harness.json`
- `codex_stdout.txt`
- `codex_stderr.txt`
- `test_stdout.txt`
- `test_stderr.txt`
- `diff_check_stdout.txt`
- `diff_check_stderr.txt`

## 상태 의미

- `patch_ready`: patch가 있고 테스트와 diff check가 통과했다.
- `test_failed`: patch는 있지만 테스트가 실패했다.
- `diff_check_failed`: patch는 있지만 whitespace 또는 diff 형식 검증이 실패했다.
- `codex_failed_no_patch`: Codex 워커가 실패했고 patch가 없다.
- `no_patch`: 워커가 끝났지만 변경이 없다.

## 장착 기준

`patch_ready`가 아니면 장착하지 않는다.
장착은 live repo에서 별도 activation pipeline이 `git apply --check`, 테스트, smoke 검증을 다시 통과한 뒤에만 진행한다.
