from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from neurokernel_seed.harness.training_worker import TrainingWorkerError, normalize_training_command
from neurokernel_seed.model.release import ModelReleaseError


OPEN_TRAINING_WORK_STATUSES = {"proposed", "accepted", "planned", "reviewing", "blocked", "failed"}
GATED_TRAINING_STATUSES = {"blocked_by_quality_gate", "blocked_by_benchmark_compare"}
FAILED_TRAINING_STATUSES = {"failed", "error"}


@dataclass(frozen=True)
class TrainingRunCallbacks:
    parse_args: Callable[[list[str]], argparse.Namespace]
    normalize_action: Callable[[str], str]
    run_action: Callable[[argparse.Namespace], dict[str, Any]]


def run_training_pending_action(args: argparse.Namespace) -> dict[str, Any]:
    items = fetch_training_work_items(args)
    return {
        "status": "completed",
        "source": "edge-core",
        "remote": remote_training_target(args),
        "count": len(items),
        "items": [summarize_training_work_item(item, index=index) for index, item in enumerate(items, start=1)],
    }


def run_training_run_action(args: argparse.Namespace, callbacks: TrainingRunCallbacks) -> dict[str, Any]:
    items = fetch_training_work_items(args)
    selected = select_training_work_item(items, work_id=getattr(args, "work_id", None), index=int(getattr(args, "index", 1) or 1))
    work_id = str(selected.get("work_id") or "")
    detail = remote_core_request(args, "GET", f"/work-items/{work_id}")
    work = detail.get("work_item") if isinstance(detail.get("work_item"), dict) else selected
    action, command_args = normalize_training_command(training_command_spec(work))
    if getattr(args, "device", None):
        command_args = replace_cli_option(command_args, "--device", str(args.device))
    command = [action, *command_args]

    report_path = _manual_training_report_path(work_id)
    if bool(getattr(args, "dry_run", False)):
        report = {
            "status": "dry_run",
            "work_id": work_id,
            "title": work.get("title"),
            "remote": remote_training_target(args),
            "command": command,
            "report": str(report_path),
        }
        _write_json(report_path, report)
        return report

    if str(work.get("status") or "") != "running":
        transition_remote_work(args, work_id, "running", reason="manual laptop training started by nk")
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        parsed_args = callbacks.parse_args(command)
        parsed_args.action = callbacks.normalize_action(parsed_args.action)
        result = callbacks.run_action(parsed_args)
    except (Exception, SystemExit) as exc:
        return record_manual_training_failure(args, work, work_id, command, started_at, report_path, exc)

    outcome = classify_training_run_result(result)
    report = {
        "status": outcome.report_status,
        "remote_status": outcome.remote_status,
        "work_id": work_id,
        "title": work.get("title"),
        "remote": remote_training_target(args),
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "nk_result": result,
        "report": str(report_path),
    }
    _write_json(report_path, report)
    add_remote_training_note(args, work_id, f"{outcome.note}: report={report_path} nk_status={result.get('status')}")
    transition_remote_work(args, work_id, outcome.remote_status, reason=outcome.reason)
    return report


def record_manual_training_failure(
    args: argparse.Namespace,
    work: dict[str, Any],
    work_id: str,
    command: list[str],
    started_at: str,
    report_path: Path,
    exc: BaseException,
) -> dict[str, Any]:
    report = {
        "status": "failed",
        "work_id": work_id,
        "title": work.get("title"),
        "remote": remote_training_target(args),
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "report": str(report_path),
    }
    _write_json(report_path, report)
    add_remote_training_note(args, work_id, f"노트북 수동 학습 실패: report={report_path} error={type(exc).__name__}: {str(exc)[:500]}")
    transition_remote_work(args, work_id, "reviewing", reason="manual laptop training failed; inspect local report")
    return report


@dataclass(frozen=True)
class TrainingRunOutcome:
    report_status: str
    remote_status: str
    reason: str
    note: str


def classify_training_run_result(nk_result: dict[str, Any]) -> TrainingRunOutcome:
    nk_status = str(nk_result.get("status") or "unknown")
    if nk_status in GATED_TRAINING_STATUSES:
        return TrainingRunOutcome(
            report_status=nk_status,
            remote_status="reviewing",
            reason=f"manual laptop training finished but deployment was gated: {nk_status}",
            note="노트북 수동 학습 보류",
        )
    if nk_status in FAILED_TRAINING_STATUSES:
        return TrainingRunOutcome(
            report_status=nk_status,
            remote_status="reviewing",
            reason=f"manual laptop training returned failure status: {nk_status}",
            note="노트북 수동 학습 확인 필요",
        )
    return TrainingRunOutcome(
        report_status="completed",
        remote_status="completed",
        reason="manual laptop training completed",
        note="노트북 수동 학습 완료",
    )


def fetch_training_work_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = remote_core_request(
        args,
        "GET",
        "/work-items",
        query={"work_type": "training_pipeline", "limit": max(1, int(getattr(args, "limit", 10) or 10))},
    )
    raw_items = payload.get("items") or payload.get("work_items") or []
    items = [item for item in raw_items if isinstance(item, dict)]
    allowed_statuses = set(getattr(args, "include_status", None) or OPEN_TRAINING_WORK_STATUSES)
    return [item for item in items if str(item.get("status") or "") in allowed_statuses]


def select_training_work_item(items: list[dict[str, Any]], *, work_id: str | None, index: int) -> dict[str, Any]:
    if work_id:
        for item in items:
            if str(item.get("work_id") or "") == work_id:
                return item
        raise ModelReleaseError(f"training work item not found: {work_id}")
    if not items:
        raise ModelReleaseError("실행 가능한 training_pipeline 후보가 없습니다. OrangePi watchdog 제안을 기다리거나 nk 학습대기를 확인하세요.")
    if index < 1 or index > len(items):
        raise ModelReleaseError(f"training work index out of range: {index} / {len(items)}")
    return items[index - 1]


def training_command_spec(work: dict[str, Any]) -> dict[str, Any]:
    metadata = work.get("metadata_json") if isinstance(work.get("metadata_json"), dict) else {}
    command_spec = metadata.get("nk_command") if isinstance(metadata.get("nk_command"), dict) else {}
    if not command_spec:
        raise ModelReleaseError(f"training work item has no nk_command: {work.get('work_id')}")
    return command_spec


def summarize_training_work_item(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    try:
        action, command_args = normalize_training_command(training_command_spec(item))
        command = [action, *command_args]
    except (TrainingWorkerError, ModelReleaseError) as exc:
        command = [f"invalid: {exc}"]
    metadata = item.get("metadata_json") if isinstance(item.get("metadata_json"), dict) else {}
    return {
        "index": index,
        "work_id": item.get("work_id"),
        "status": item.get("status"),
        "slot": metadata.get("slot") or metadata.get("model_slot") or "unknown",
        "title": item.get("title"),
        "priority": item.get("priority"),
        "risk_level": item.get("risk_level"),
        "command": command,
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def replace_cli_option(args: list[str], option: str, value: str) -> list[str]:
    updated: list[str] = []
    skip_next = False
    replaced = False
    for current in args:
        if skip_next:
            skip_next = False
            continue
        if current == option:
            updated.extend([option, value])
            skip_next = True
            replaced = True
            continue
        if current.startswith(f"{option}="):
            updated.append(f"{option}={value}")
            replaced = True
            continue
        updated.append(current)
    if not replaced:
        updated.extend([option, value])
    return updated


def remote_training_target(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "remote_host": getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5"),
        "remote_project": (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/"),
        "remote_core_url": str(getattr(args, "remote_core_url", None) or os.getenv("NEUROKERNEL_EDGE_CORE_URL", "http://127.0.0.1:8765")).rstrip("/"),
        "ssh_connect_timeout": int(getattr(args, "ssh_connect_timeout", 10)),
    }


def remote_core_request(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = remote_training_target(args)
    url = f"{target['remote_core_url']}{path}"
    if query:
        compact = {key: value for key, value in query.items() if value is not None}
        if compact:
            url = f"{url}?{urlencode(compact)}"
    command_parts = ["cd", _sh_quote(str(target["remote_project"])), "&&", "curl", "-fsS"]
    method = method.upper()
    if method != "GET":
        command_parts.extend(["-X", method, "-H", _sh_quote("Content-Type: application/json")])
        command_parts.extend(["--data-binary", _sh_quote(json.dumps(payload or {}, ensure_ascii=False))])
    command_parts.append(_sh_quote(url))
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(target['ssh_connect_timeout'])}", str(target["remote_host"]), " ".join(command_parts)],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModelReleaseError(f"OrangePi Core API request failed: {method} {path}: {detail}")
    try:
        result = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ModelReleaseError(f"OrangePi Core API returned invalid json: {(completed.stdout or '').strip()[:500]}") from exc
    if not isinstance(result, dict):
        raise ModelReleaseError(f"OrangePi Core API returned non-object json: {method} {path}")
    return result


def transition_remote_work(args: argparse.Namespace, work_id: str, status: str, *, reason: str) -> dict[str, Any]:
    return remote_core_request(
        args,
        "POST",
        f"/work-items/{work_id}/status",
        payload={"status": status, "actor": "nk-manual-training", "reason": reason},
    )


def add_remote_training_note(args: argparse.Namespace, work_id: str, note: str) -> dict[str, Any]:
    return remote_core_request(
        args,
        "POST",
        f"/work-items/{work_id}/note",
        payload={"actor": "nk-manual-training", "note": note},
    )


def _manual_training_report_path(work_id: str) -> Path:
    report_dir = Path(os.getenv("NEUROKERNEL_MANUAL_TRAINING_JOB_DIR", "artifacts/training_jobs"))
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"manual_{_safe_file_stem(work_id)}_{_utc_stamp()}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _safe_file_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(value)).strip("._")
    return safe or "training_work"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"
