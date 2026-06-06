from __future__ import annotations

from pathlib import Path
from typing import Any

from neurokernel_seed.harness.executors.base import ExecutionResult, TimedExecution
from neurokernel_seed.harness.trace import redact_text


class BenchmarkExecutor:
    def __init__(self, *, project_root: str | Path = "."):
        self.project_root = Path(project_root).resolve()

    def execute(self, action_id: str, params: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ExecutionResult:
        params = params or {}
        timer = TimedExecution()
        if action_id != "run_safe_benchmark":
            return timer.finish(action_id, success=False, error_type="unsupported_action")
        try:
            from neurokernel_seed.eval.benchmark import BenchmarkConfig, run_benchmark

            model = self._resolve_project_path(str(params.get("model") or "artifacts/world_model_slot_v2_model_needed_v3.onnx"))
            episodes = int(params.get("episodes") or 5)
            if episodes < 1 or episodes > 50:
                raise ValueError("episodes must be between 1 and 50")
            out_dir = self._resolve_project_path(str(params.get("out_dir") or "artifacts/benchmark/harness_safe_benchmark"))
            result = run_benchmark(
                model,
                out_dir=out_dir,
                config=BenchmarkConfig(episodes=episodes, trace_episodes=min(3, episodes), max_failures_per_env=3, strict=True, gate_mode="hybrid_veto"),
            )
            return timer.finish(action_id, success=bool(result.get("passed")), result=result)
        except Exception as exc:
            return timer.finish(action_id, success=False, result={"error": str(exc)}, stderr=redact_text(str(exc)), error_type=exc.__class__.__name__)

    def _resolve_project_path(self, path: str) -> Path:
        target = (self.project_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc
        return target

