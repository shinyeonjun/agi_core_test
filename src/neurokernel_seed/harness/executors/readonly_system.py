from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil

from neurokernel_seed.harness.code_structure import CodeStructureThresholds, inspect_code_structure
from neurokernel_seed.harness.executors.base import ExecutionResult, TimedExecution
from neurokernel_seed.harness.trace import redact_text


class ReadOnlyExecutor:
    def __init__(self, *, project_root: str | Path = ".", memory_path: str | Path = "data/harness.db"):
        self.project_root = Path(project_root).resolve()
        self.memory_path = Path(memory_path)

    def execute(self, action_id: str, params: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> ExecutionResult:
        params = params or {}
        timer = TimedExecution()
        try:
            if action_id == "get_uptime":
                return timer.finish(action_id, success=True, result=self._get_uptime())
            if action_id == "get_disk_usage":
                return timer.finish(action_id, success=True, result=self._get_disk_usage(str(params.get("path") or ".")))
            if action_id == "get_memory_usage":
                return timer.finish(action_id, success=True, result=self._get_memory_usage())
            if action_id == "get_cpu_temp":
                return timer.finish(action_id, success=True, result=self._get_cpu_temp())
            if action_id == "get_cpu_per_core_usage":
                return timer.finish(action_id, success=True, result=self._get_cpu_per_core_usage())
            if action_id == "get_service_status":
                return timer.finish(action_id, success=True, result=self._get_service_status(str(params.get("service") or "")))
            if action_id == "tail_logs":
                return timer.finish(action_id, success=True, result=self._tail_logs(str(params.get("path") or ""), int(params.get("lines") or 80)))
            if action_id == "list_artifacts":
                return timer.finish(action_id, success=True, result=self._list_artifacts(str(params.get("path") or "artifacts")))
            if action_id == "get_recent_trace":
                return timer.finish(action_id, success=True, result=self._get_recent_trace(int(params.get("limit") or 5)))
            if action_id == "diagnose_system_symptoms":
                return timer.finish(action_id, success=True, result=self._diagnose_system_symptoms(params))
            if action_id == "inspect_code_structure":
                return timer.finish(action_id, success=True, result=self._inspect_code_structure(params))
            if action_id == "inspect_work_pipeline":
                return timer.finish(action_id, success=True, result=self._inspect_work_pipeline(params))
            return timer.finish(action_id, success=False, result={}, error_type="unsupported_action")
        except Exception as exc:
            return timer.finish(action_id, success=False, result={"error": str(exc)}, stderr=redact_text(str(exc)), error_type=exc.__class__.__name__)

    def _get_uptime(self) -> dict[str, Any]:
        if platform.system().lower() == "windows":
            return {"platform": platform.platform(), "boot_time_unavailable": True}
        uptime_text = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        return {"platform": platform.platform(), "uptime_seconds": float(uptime_text)}

    def _get_disk_usage(self, path: str) -> dict[str, Any]:
        target = self._resolve_project_path(path)
        usage = shutil.disk_usage(target)
        return {
            "path": str(target),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_percent": round(usage.used / usage.total * 100.0, 3) if usage.total else 0.0,
        }

    def _get_memory_usage(self) -> dict[str, Any]:
        if platform.system().lower() == "windows":
            return {"platform": platform.platform(), "memory_unavailable": True}
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, rest = line.split(":", 1)
            value = int(rest.strip().split()[0]) * 1024
            info[key] = value
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        return {"total_bytes": total, "available_bytes": available, "used_bytes": total - available, "used_percent": round((total - available) / total * 100.0, 3) if total else 0.0}

    def _get_cpu_temp(self) -> dict[str, Any]:
        candidates = [Path("/sys/class/thermal/thermal_zone0/temp"), Path("/sys/class/thermal/thermal_zone1/temp")]
        for path in candidates:
            if path.exists():
                raw = path.read_text(encoding="utf-8").strip()
                return {"path": str(path), "celsius": round(float(raw) / 1000.0, 2)}
        return {"available": False}

    def _get_cpu_per_core_usage(self) -> dict[str, Any]:
        percentages = psutil.cpu_percent(interval=0.2, percpu=True)
        per_core = [{"core": index, "used_percent": round(float(value), 3)} for index, value in enumerate(percentages)]
        return {"per_core_percent": per_core, "core_count": len(per_core)}

    def _get_service_status(self, service: str) -> dict[str, Any]:
        if not service:
            return {"available": False, "reason": "service param is required"}
        if platform.system().lower() == "windows":
            return {"available": False, "reason": "systemctl is not available on Windows"}
        result = subprocess.run(["systemctl", "is-active", service], text=True, capture_output=True, timeout=5, check=False)
        return {"service": service, "active": result.stdout.strip(), "returncode": result.returncode, "stderr": redact_text(result.stderr)}

    def _tail_logs(self, path: str, lines: int) -> dict[str, Any]:
        if lines < 1 or lines > 300:
            raise ValueError("lines must be between 1 and 300")
        target = self._resolve_project_path(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        data = target.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        return {"path": str(target), "lines": lines, "text": redact_text("\n".join(data))}

    def _list_artifacts(self, path: str) -> dict[str, Any]:
        target = self._resolve_project_path(path)
        if not target.exists():
            return {"path": str(target), "exists": False, "items": []}
        items = []
        for child in sorted(target.iterdir(), key=lambda item: item.name)[:200]:
            stat = child.stat()
            items.append({"name": child.name, "type": "dir" if child.is_dir() else "file", "size_bytes": stat.st_size})
        return {"path": str(target), "exists": True, "items": items}

    def _get_recent_trace(self, limit: int) -> dict[str, Any]:
        from neurokernel_seed.harness.memory import HarnessMemory

        with HarnessMemory(self.memory_path) as memory:
            return {"traces": memory.recent_traces(max(1, min(limit, 50)))}

    def _diagnose_system_symptoms(self, params: dict[str, Any]) -> dict[str, Any]:
        symptoms = str(params.get("symptoms") or "").strip()
        log_paths = _string_list(params.get("log_paths"))
        log_lines = max(1, min(int(params.get("log_lines") or 80), 300))
        artifact_path = str(params.get("artifact_path") or "artifacts")
        trace_limit = max(1, min(int(params.get("trace_limit") or 5), 50))
        service = str(params.get("service") or "").strip()

        observations: dict[str, Any] = {
            "symptoms": symptoms,
            "uptime": self._get_uptime(),
            "disk": self._get_disk_usage("."),
            "memory": self._get_memory_usage(),
            "cpu_temperature": self._get_cpu_temp(),
            "cpu_per_core": self._get_cpu_per_core_usage(),
            "artifacts": self._list_artifacts(artifact_path),
            "recent_trace": self._get_recent_trace(trace_limit),
            "logs": [],
        }
        if service:
            observations["service"] = self._get_service_status(service)
        for path in log_paths[:5]:
            observations["logs"].append(self._safe_log_snapshot(path, log_lines))

        candidates = self._diagnosis_candidates(observations)
        next_actions = self._diagnosis_next_actions(candidates, observations)
        status = "attention_needed" if candidates else "no_clear_issue_detected"
        return {
            "request": {
                "symptoms": symptoms,
                "log_paths": log_paths[:5],
                "log_lines": log_lines,
                "artifact_path": artifact_path,
                "trace_limit": trace_limit,
                "service": service,
            },
            "summary": {
                "status": status,
                "headline": _diagnosis_headline(status, candidates),
                "cause_candidates": candidates,
                "next_actions": next_actions,
            },
            "observations": observations,
        }

    def _safe_log_snapshot(self, path: str, lines: int) -> dict[str, Any]:
        try:
            snapshot = self._tail_logs(path, lines)
        except Exception as exc:
            return {"path": path, "available": False, "reason": redact_text(str(exc))}
        text = str(snapshot.get("text") or "")
        hits = _log_signal_counts(text)
        return {
            "path": snapshot["path"],
            "available": True,
            "lines": snapshot["lines"],
            "signal_counts": hits,
            "tail_excerpt": _last_nonempty_lines(text, 8),
        }

    def _diagnosis_candidates(self, observations: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        memory = observations.get("memory", {})
        memory_used = _as_float(memory.get("used_percent"))
        if memory_used is not None and memory_used >= 90:
            candidates.append(_candidate("memory_pressure", "high", f"메모리 사용률이 {memory_used:.1f}%입니다.", ["동시 실행 작업 수와 학습 배치 크기를 확인합니다."]))

        disk = observations.get("disk", {})
        disk_used = _as_float(disk.get("used_percent"))
        if disk_used is not None and disk_used >= 90:
            candidates.append(_candidate("disk_pressure", "high", f"디스크 사용률이 {disk_used:.1f}%입니다.", ["checkpoint, 로그, 임시 artifact 용량을 확인합니다."]))

        cores = observations.get("cpu_per_core", {}).get("per_core_percent") or []
        core_values = [_as_float(item.get("used_percent")) for item in cores if isinstance(item, dict)]
        core_values = [value for value in core_values if value is not None]
        if core_values:
            avg_cpu = sum(core_values) / len(core_values)
            max_cpu = max(core_values)
            if avg_cpu >= 90 or max_cpu >= 98:
                candidates.append(_candidate("cpu_saturation", "medium", f"CPU 평균 {avg_cpu:.1f}%, 최대 코어 {max_cpu:.1f}%입니다.", ["CPU를 많이 쓰는 학습/변환/벤치마크 작업이 겹쳤는지 확인합니다."]))

        temp = observations.get("cpu_temperature", {})
        celsius = _as_float(temp.get("celsius"))
        if celsius is not None and celsius >= 80:
            candidates.append(_candidate("thermal_throttling", "medium", f"CPU 온도가 {celsius:.1f}도입니다.", ["쿨링 상태와 장시간 부하 작업을 확인합니다."]))

        log_hits = _combined_log_hits(observations.get("logs", []))
        if log_hits.get("error", 0) or log_hits.get("exception", 0) or log_hits.get("oom", 0):
            candidates.append(_candidate("log_errors", "medium", f"최근 로그에서 오류 신호가 발견되었습니다: {log_hits}", ["오류 직전 단계와 변경된 입력 파일을 확인합니다."]))

        traces = observations.get("recent_trace", {}).get("traces") or []
        failure_count = sum(1 for trace in traces if isinstance(trace, dict) and _trace_looks_failed(trace))
        if failure_count:
            candidates.append(_candidate("recent_failures", "medium", f"최근 trace {failure_count}건이 실패로 보입니다.", ["가장 최근 실패 trace의 원인 분류와 실행 결과를 확인합니다."]))

        artifacts = observations.get("artifacts", {})
        items = artifacts.get("items") or []
        if artifacts.get("exists") is False:
            candidates.append(_candidate("artifact_path_missing", "medium", f"artifact 경로가 없습니다: {artifacts.get('path')}", ["모델/벤치마크 산출물 저장 경로와 현재 프로젝트 루트를 확인합니다."]))
        elif isinstance(items, list) and not items:
            candidates.append(_candidate("artifact_path_empty", "medium", "artifact 목록이 비어 있습니다.", ["학습/변환/벤치마크 산출물이 다른 경로에 저장됐는지 확인합니다."]))

        return candidates

    def _diagnosis_next_actions(self, candidates: list[dict[str, Any]], observations: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        for candidate in candidates:
            for action in candidate.get("recommended_actions", []):
                if action not in actions:
                    actions.append(str(action))
        if observations.get("logs") == []:
            actions.append("관련 로그 경로가 있으면 log_paths에 넣고 다시 진단합니다.")
        if not actions:
            actions.append("명확한 병목은 보이지 않습니다. 증상 직후 로그 경로와 관련 서비스를 지정해 다시 확인합니다.")
        return actions[:6]

    def _inspect_code_structure(self, params: dict[str, Any]) -> dict[str, Any]:
        raw_paths = params.get("paths")
        if raw_paths is None:
            paths = None
        elif isinstance(raw_paths, list) and all(isinstance(item, str) for item in raw_paths):
            paths = raw_paths
        else:
            raise ValueError("paths must be a list of strings")
        thresholds = CodeStructureThresholds(
            long_file_lines=int(params.get("long_file_lines") or 300),
            very_long_file_lines=int(params.get("very_long_file_lines") or 700),
            large_function_lines=int(params.get("large_function_lines") or 80),
            many_functions=int(params.get("many_functions") or 18),
            many_classes=int(params.get("many_classes") or 8),
        )
        return inspect_code_structure(
            self.project_root,
            paths=paths,
            max_files=int(params.get("max_files") or 250),
            max_results=int(params.get("max_results") or 20),
            thresholds=thresholds,
        )

    def _inspect_work_pipeline(self, params: dict[str, Any]) -> dict[str, Any]:
        from neurokernel_seed.harness.work_service import WorkItemService

        service = WorkItemService(db_path=self.memory_path)
        return service.pipeline_status(
            limit=int(params.get("limit") or 20),
            stale_after_seconds=int(params.get("stale_after_seconds") or 300),
        )

    def _resolve_project_path(self, path: str) -> Path:
        target = (self.project_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc
        return target


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate(candidate_id: str, severity: str, evidence: str, recommended_actions: list[str]) -> dict[str, Any]:
    return {"id": candidate_id, "severity": severity, "evidence": evidence, "recommended_actions": recommended_actions}


def _log_signal_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        "error": lowered.count("error"),
        "exception": lowered.count("exception"),
        "oom": lowered.count("out of memory") + lowered.count("oom"),
        "stalled": lowered.count("stalled") + lowered.count("hang") + lowered.count("timeout"),
    }


def _combined_log_hits(logs: list[Any]) -> dict[str, int]:
    combined = {"error": 0, "exception": 0, "oom": 0, "stalled": 0}
    for log in logs:
        if not isinstance(log, dict):
            continue
        for key, value in (log.get("signal_counts") or {}).items():
            if key in combined:
                combined[key] += int(value)
    return combined


def _last_nonempty_lines(text: str, limit: int) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _trace_looks_failed(trace: dict[str, Any]) -> bool:
    text = json.dumps(trace, ensure_ascii=False).lower()
    return any(token in text for token in ("failed", "failure", "error", "exception", "execution_failed"))


def _diagnosis_headline(status: str, candidates: list[dict[str, Any]]) -> str:
    if status == "no_clear_issue_detected":
        return "수집된 지표만으로는 뚜렷한 원인 후보가 없습니다."
    top = ", ".join(str(candidate.get("id")) for candidate in candidates[:3])
    return f"우선 확인할 원인 후보: {top}"
