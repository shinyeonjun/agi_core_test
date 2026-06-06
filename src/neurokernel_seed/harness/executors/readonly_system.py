from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil

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

    def _resolve_project_path(self, path: str) -> Path:
        target = (self.project_root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        try:
            target.relative_to(self.project_root)
        except ValueError as exc:
            raise ValueError(f"path escapes project root: {path}") from exc
        return target
