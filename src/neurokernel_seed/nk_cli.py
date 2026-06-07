from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurokernel_seed.eval.gate_ablation import GateAblationConfig, run_gate_ablation
from neurokernel_seed.model.pipeline import TrainingPipelineConfig, run_training_pipeline
from neurokernel_seed.model.release import (
    ModelReleaseError,
    ModelReleaseRemoteConfig,
    TrainDeployModelConfig,
    activate_model_release,
    current_model_release,
    deploy_training_run,
    list_model_releases,
    train_deploy_model,
)
from neurokernel_seed.model.runtime_action import train_runtime_action_model
from neurokernel_seed.nk_console import menu as nk_menu
from neurokernel_seed.nk_console.dashboard import DashboardController
from neurokernel_seed.replay.runtime_dataset import RuntimeReplayEtlConfig, run_runtime_replay_etl
from neurokernel_seed.replay.runtime_features import check_runtime_feature_gates, export_runtime_features, validate_runtime_features
from neurokernel_seed.replay.slot_dataset import check_slot_dataset_gates, validate_slot_features


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action is None:
        return _run_dashboard(args)
    args.action = _normalize_action(args.action)
    try:
        result = _run_action(args)
    except ModelReleaseError as exc:
        print(f"실패 {exc}", file=sys.stderr)
        return 1
    _print_result(args.action, result, json_mode=args.json)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nk",
        description="NeuroKernel 모델 콘솔. `nk`만 입력하면 반복 TUI가 열립니다.",
    )
    sub = parser.add_subparsers(dest="action")

    for name in ("runtime-auto", "auto", "자동", "학습", "파이프라인", "사용자동"):
        item = sub.add_parser(name)
        _add_runtime_auto_options(item)
        item.set_defaults(deploy_runtime=False)

    for name in ("runtime-cycle", "auto-deploy", "learn-deploy"):
        item = sub.add_parser(name)
        _add_runtime_auto_options(item)
        item.set_defaults(deploy_runtime=True)

    for name in ("data", "dataset", "features", "데이터"):
        item = sub.add_parser(name)
        item.add_argument("--features")
        item.add_argument("--json", action="store_true")

    for name in ("runtime-data", "사용데이터", "현실데이터", "실사용"):
        item = sub.add_parser(name)
        item.add_argument("--source", choices=["edge", "local"], default=os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
        item.add_argument("--db", default=os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
        item.add_argument("--remote-db", default=os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db"))
        item.add_argument("--cache-db", default=os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE"))
        _add_remote_options(item)
        item.add_argument("--out", default=os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
        item.add_argument("--limit", type=int)
        item.add_argument("--min-rows", type=int, default=1)
        item.add_argument("--allow-no-execution", action="store_true")
        item.add_argument("--json", action="store_true")

    for name in ("runtime-features", "사용특징", "현실특징"):
        item = sub.add_parser(name)
        item.add_argument("--replay", default=os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
        item.add_argument("--out", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
        item.add_argument("--test-ratio", type=float, default=0.2)
        item.add_argument("--min-rows", type=int, default=10)
        item.add_argument("--min-actions", type=int, default=1)
        item.add_argument("--json", action="store_true")

    for name in ("runtime-train", "사용학습", "현실학습"):
        item = sub.add_parser(name)
        _add_runtime_train_options(item)

    for name in ("deploy-runtime", "runtime-deploy", "runtime-use"):
        item = sub.add_parser(name)
        _add_runtime_deploy_options(item)

    for name in ("check", "확인", "train"):
        item = sub.add_parser(name)
        item.add_argument("run_name", nargs="?")
        _add_train_options(item)

    for name in ("deploy", "deploy-use", "ship", "배포적용"):
        item = sub.add_parser(name)
        item.add_argument("run_name")
        _add_remote_options(item)
        item.add_argument("--run-dir")
        item.add_argument("--json", action="store_true")

    for name in ("deploy-best", "배포", "동기화", "배포동기화", "배포최고"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--run-dir")
        item.add_argument("--limit", type=int, default=5)
        item.add_argument("--json", action="store_true")

    for name in ("bench", "벤치"):
        item = sub.add_parser(name)
        item.add_argument("run_name", nargs="?")
        item.add_argument("--model")
        item.add_argument("--out-dir")
        item.add_argument("--episodes", type=int, default=50)
        item.add_argument("--trace-episodes", type=int, default=10)
        item.add_argument("--max-failures-per-env", type=int, default=10)
        item.add_argument("--no-strict", dest="strict", action="store_false")
        item.set_defaults(strict=True)
        item.add_argument("--json", action="store_true")

    for name in ("bench-current", "현재벤치"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--out-dir")
        item.add_argument("--episodes", type=int, default=50)
        item.add_argument("--trace-episodes", type=int, default=10)
        item.add_argument("--max-failures-per-env", type=int, default=10)
        item.add_argument("--no-strict", dest="strict", action="store_false")
        item.set_defaults(strict=True)
        item.add_argument("--json", action="store_true")

    for name in ("compare", "비교", "벤치비교"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--run-dir")
        item.add_argument("--limit", type=int, default=5)
        item.add_argument("--json", action="store_true")

    for name in ("top", "순위"):
        item = sub.add_parser(name)
        item.add_argument("--run-dir")
        item.add_argument("--limit", type=int, default=5)
        item.add_argument("--json", action="store_true")

    for name in ("list", "models", "목록"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--json", action="store_true")

    for name in ("current", "edge", "현재"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--json", action="store_true")

    for name in ("use", "switch", "적용"):
        item = sub.add_parser(name)
        item.add_argument("run_name")
        _add_remote_options(item)
        item.add_argument("--json", action="store_true")

    for name in ("status", "상태"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--run-dir")
        item.add_argument("--limit", type=int, default=5)
        item.add_argument("--json", action="store_true")

    for name in ("advanced", "고급"):
        item = sub.add_parser(name)
        item.add_argument("--json", action="store_true")
    return parser


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features")
    parser.add_argument("--out-dir")
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-project")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--skip-gate-ablation", action="store_true")
    parser.add_argument("--overwrite-local-run", action="store_true")
    parser.add_argument("--json", action="store_true")


def _add_runtime_auto_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["edge", "local"], default=os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
    parser.add_argument("--db", default=os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
    parser.add_argument("--remote-db", default=os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db"))
    parser.add_argument("--cache-db", default=os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE"))
    _add_remote_options(parser)
    parser.add_argument("--replay-out", default=os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    parser.add_argument("--features-out", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--model-out", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-actions", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_AUTO_EPOCHS", "50")))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--deploy", dest="deploy_runtime", action="store_true")
    parser.add_argument("--no-deploy", dest="deploy_runtime", action="store_false")
    parser.add_argument("--json", action="store_true")


def _add_runtime_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--out", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-actions", type=int, default=1)
    parser.add_argument("--json", action="store_true")


def _add_runtime_deploy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    _add_remote_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-project")
    parser.add_argument("--ssh-connect-timeout", type=int, default=10)


def _run_dashboard(args: argparse.Namespace) -> int:
    controller = DashboardController(
        run_action=_run_action,
        print_result=lambda action, result: _print_result(action, result, json_mode=False),
        make_status_args=_status_args,
        model_error_type=ModelReleaseError,
        short_current=_short_current,
        short_best=_short_best,
        style=_style,
        kv=_kv,
    )
    return controller.run(args)


def _normalize_action(action: str) -> str:
    return nk_menu.normalize_action(action)


def _probe_current(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        return _run_action(_status_args(args, "current"))
    except ModelReleaseError:
        return None


def _probe_top(args: argparse.Namespace) -> dict[str, Any] | None:
    try:
        return _run_action(_status_args(args, "top"))
    except ModelReleaseError:
        return None


def _status_args(base: argparse.Namespace, action: str) -> argparse.Namespace:
    data = vars(base).copy()
    data.update({"action": action, "run_dir": getattr(base, "run_dir", None), "limit": getattr(base, "limit", 5), "json": False})
    return argparse.Namespace(**data)


def _run_action(args: argparse.Namespace) -> dict[str, Any]:
    args.action = _normalize_action(args.action)
    if args.action in {"runtime-auto", "runtime-cycle"}:
        return _run_runtime_auto_action(args)
    if args.action == "data":
        return _run_data_action(args)
    if args.action == "runtime-data":
        return _run_runtime_data_action(args)
    if args.action == "runtime-features":
        return _run_runtime_features_action(args)
    if args.action == "runtime-train":
        return _run_runtime_training_action(args)
    if args.action == "deploy-runtime":
        return _run_deploy_runtime_action(args)
    if args.action == "check":
        return train_deploy_model(
            TrainDeployModelConfig(
                features=getattr(args, "features", None),
                out_dir=getattr(args, "out_dir", None),
                run_name=getattr(args, "run_name", None),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                dry_run=True,
                activate=False,
                epochs=getattr(args, "epochs", 50),
                batch_size=getattr(args, "batch_size", 1024),
                device=getattr(args, "device", "cuda"),
                patience=getattr(args, "patience", 10),
                run_gate_ablation=not getattr(args, "skip_gate_ablation", False),
                overwrite_local_run=getattr(args, "overwrite_local_run", False),
            )
        )
    if args.action == "train":
        return _run_local_training_action(args)
    if args.action in {"deploy", "deploy-use"}:
        return _run_deploy_action(args, activate=args.action == "deploy-use")
    if args.action == "deploy-best":
        return _run_deploy_best_action(args)
    if args.action == "bench":
        return _run_benchmark_action(args)
    if args.action == "bench-current":
        return _run_benchmark_current_action(args)
    if args.action == "top":
        return _run_top_models(args)
    if args.action == "compare":
        return _run_compare(args)
    if args.action == "status":
        return _run_status(args)
    if args.action == "advanced":
        return _run_advanced(args)

    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    if args.action == "list":
        return list_model_releases(remote_config)
    if args.action == "current":
        return _run_current(args)
    if args.action == "use":
        return activate_model_release(args.run_name, remote_config)
    raise ModelReleaseError(f"알 수 없는 명령: {args.action}")


def _run_data_action(args: argparse.Namespace) -> dict[str, Any]:
    features = getattr(args, "features", None) or os.getenv("NEUROKERNEL_TRAIN_FEATURES")
    if not features:
        raise ModelReleaseError("features 없음: --features 또는 NEUROKERNEL_TRAIN_FEATURES 필요")
    feature_path = Path(features)
    validation = validate_slot_features(feature_path)
    gates = check_slot_dataset_gates(feature_path)
    return {
        "status": "passed" if gates.get("passed") else "failed",
        "features": str(feature_path),
        "manifest": str(feature_path.with_suffix(feature_path.suffix + ".manifest.json")),
        "validation": validation,
        "gates": gates,
        "ready_for_training": bool(gates.get("passed")),
    }


def _run_runtime_auto_action(args: argparse.Namespace) -> dict[str, Any]:
    replay_out = Path(getattr(args, "replay_out", None) or os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    features_out = Path(getattr(args, "features_out", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    model_out = Path(getattr(args, "model_out", None) or getattr(args, "runtime_model_out", None) or os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))

    data_result = _run_runtime_data_action(
        argparse.Namespace(
            action="runtime-data",
            source=getattr(args, "source", "edge"),
            db=getattr(args, "db", "data/harness.db"),
            remote_host=getattr(args, "remote_host", None),
            remote_project=getattr(args, "remote_project", None),
            remote_db=getattr(args, "remote_db", "data/harness.db"),
            cache_db=getattr(args, "cache_db", None),
            out=str(replay_out),
            limit=getattr(args, "limit", None),
            min_rows=getattr(args, "min_rows", 10),
            allow_no_execution=False,
        )
    )
    _require_stage_ready(data_result, "사용데이터", "ready_for_runtime_training")

    feature_result = _run_runtime_features_action(
        argparse.Namespace(
            action="runtime-features",
            replay=str(replay_out),
            out=str(features_out),
            test_ratio=getattr(args, "test_ratio", 0.2),
            min_rows=getattr(args, "min_rows", 10),
            min_actions=getattr(args, "min_actions", 1),
        )
    )
    _require_stage_ready(feature_result, "사용특징", "ready_for_runtime_model_training")

    train_result = _run_runtime_training_action(
        argparse.Namespace(
            action="runtime-train",
            features=str(features_out),
            out=str(model_out),
            epochs=getattr(args, "epochs", 50),
            batch_size=getattr(args, "batch_size", 128),
            lr=getattr(args, "lr", 1e-3),
            weight_decay=getattr(args, "weight_decay", 1e-4),
            hidden_dim=getattr(args, "hidden_dim", 64),
            hidden_layers=getattr(args, "hidden_layers", 2),
            device=getattr(args, "device", "auto"),
            patience=getattr(args, "patience", 10),
            min_rows=getattr(args, "min_rows", 10),
            min_actions=getattr(args, "min_actions", 1),
        )
    )
    deploy_result = None
    current_after = None
    if bool(getattr(args, "deploy_runtime", False)):
        deploy_result = _run_deploy_runtime_action(
            argparse.Namespace(
                action="deploy-runtime",
                model=str(train_result.get("checkpoint") or model_out),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
            )
        )
        current_after = _run_current(
            argparse.Namespace(
                action="current",
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
            )
        )

    report = {
        "status": "completed",
        "pipeline": "runtime_action_cycle_v1" if deploy_result else "runtime_action_auto_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "steps": {
            "runtime_data": data_result,
            "runtime_features": feature_result,
            "runtime_train": train_result,
            "runtime_deploy": deploy_result,
            "current_after": current_after,
        },
        "artifacts": {
            "replay": str(replay_out),
            "features": str(features_out),
            "model": train_result.get("checkpoint") or str(model_out),
            "metrics": train_result.get("metrics"),
            "manifest": train_result.get("manifest"),
        },
    }
    report_path = _write_runtime_auto_report(report)
    return {**report, "report": str(report_path)}


def _require_stage_ready(result: dict[str, Any], label: str, ready_key: str) -> None:
    if result.get(ready_key):
        return
    gates = result.get("gates") or {}
    failed = [name for name, item in (gates.get("gates") or {}).items() if isinstance(item, dict) and not item.get("passed")]
    detail = f" 실패 게이트: {', '.join(failed)}" if failed else ""
    raise ModelReleaseError(f"{label} 단계가 학습 준비 상태가 아님.{detail}")


def _write_runtime_auto_report(report: dict[str, Any]) -> Path:
    report_dir = Path(os.getenv("NEUROKERNEL_RUNTIME_PIPELINE_DIR", "artifacts/runtime_pipeline"))
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"runtime_action_auto_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = report_dir / "latest_runtime_action_auto.json"
    latest_path.write_text(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _run_runtime_data_action(args: argparse.Namespace) -> dict[str, Any]:
    source = str(getattr(args, "source", None) or os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
    db = _resolve_runtime_data_db(args, source=source)
    out = Path(getattr(args, "out", None) or os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    try:
        result = run_runtime_replay_etl(
            RuntimeReplayEtlConfig(
                db_path=db,
                out_path=out,
                limit=getattr(args, "limit", None),
                min_rows=getattr(args, "min_rows", 1),
                require_execution=not getattr(args, "allow_no_execution", False),
            )
        )
    except FileNotFoundError as exc:
        raise ModelReleaseError(f"사용데이터 입력 실패: {exc}") from exc
    return {"status": result["status"], "source": source, "db": str(db), **result}


def _resolve_runtime_data_db(args: argparse.Namespace, *, source: str) -> Path:
    if source == "local":
        return Path(getattr(args, "db", None) or os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
    if source != "edge":
        raise ModelReleaseError(f"알 수 없는 사용데이터 출처: {source}")
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    remote_db = getattr(args, "remote_db", None) or os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db")
    cache_db_value = getattr(args, "cache_db", None) or os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE")
    if cache_db_value:
        cache_db = Path(cache_db_value)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cache_db = Path(f"data/runtime_sources/{remote_host}_harness_{stamp}.db")
    remote_path = f"{remote_project}/{str(remote_db).lstrip('/')}"
    _copy_remote_file(str(remote_host), remote_path, cache_db)
    return cache_db


def _run_runtime_features_action(args: argparse.Namespace) -> dict[str, Any]:
    replay = Path(getattr(args, "replay", None) or os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    out = Path(getattr(args, "out", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    exported = export_runtime_features(replay, out, test_ratio=float(getattr(args, "test_ratio", 0.2)))
    validation = validate_runtime_features(out)
    gates = check_runtime_feature_gates(
        out,
        min_rows=int(getattr(args, "min_rows", 10)),
        min_actions=int(getattr(args, "min_actions", 1)),
    )
    return {
        "status": "passed" if gates["passed"] else "failed",
        "replay": str(replay),
        "features": str(out),
        "export": exported,
        "validation": validation,
        "gates": gates,
        "ready_for_runtime_model_training": bool(gates["ready_for_runtime_model_training"]),
    }


def _run_runtime_training_action(args: argparse.Namespace) -> dict[str, Any]:
    features = Path(getattr(args, "features", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    out = Path(getattr(args, "out", None) or os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    try:
        result = train_runtime_action_model(
            features,
            out,
            epochs=int(getattr(args, "epochs", 100)),
            batch_size=int(getattr(args, "batch_size", 128)),
            lr=float(getattr(args, "lr", 1e-3)),
            weight_decay=float(getattr(args, "weight_decay", 1e-4)),
            hidden_dim=int(getattr(args, "hidden_dim", 64)),
            hidden_layers=int(getattr(args, "hidden_layers", 2)),
            device=str(getattr(args, "device", "auto")),
            patience=int(getattr(args, "patience", 20)),
            min_rows=int(getattr(args, "min_rows", 10)),
            min_actions=int(getattr(args, "min_actions", 1)),
        )
    except ValueError as exc:
        raise ModelReleaseError(f"사용학습 입력 실패: {exc}") from exc
    return {"status": "completed", "features": str(features), "out": str(out), **result}


def _run_deploy_runtime_action(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(getattr(args, "model", None) or os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    manifest = model.with_suffix(".manifest.json")
    if not model.exists():
        raise ModelReleaseError(f"runtime 모델 없음: {model}")
    if not manifest.exists():
        raise ModelReleaseError(f"runtime manifest 없음: {manifest}")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("model_slot") != "runtime_action_model":
        raise ModelReleaseError(f"runtime manifest 슬롯 불일치: {manifest_payload.get('model_slot')}")
    if manifest_payload.get("data_origin") != "runtime_experience_log":
        raise ModelReleaseError(f"runtime manifest 데이터 출처 불일치: {manifest_payload.get('data_origin')}")

    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    timeout = int(getattr(args, "ssh_connect_timeout", 10))
    remote_model = f"{remote_project}/artifacts/current_runtime_action_model.pt"
    remote_manifest = f"{remote_project}/artifacts/current_runtime_action_model.manifest.json"
    _copy_local_file(str(remote_host), model, remote_model, timeout=timeout)
    _copy_local_file(str(remote_host), manifest, remote_manifest, timeout=timeout)
    return {
        "status": "deployed_and_activated",
        "slot": "runtime_action",
        "model": str(model),
        "manifest": str(manifest),
        "remote_host": remote_host,
        "remote_model": remote_model,
        "remote_manifest": remote_manifest,
        "manifest_payload": manifest_payload,
        "model_slots": {
            "runtime": {
                "slot": "runtime_action",
                "status": "active",
                "model": remote_model,
                "manifest": remote_manifest,
                "data_origin": "runtime_experience_log",
            }
        },
    }


def _run_local_training_action(args: argparse.Namespace) -> dict[str, Any]:
    features = getattr(args, "features", None) or os.getenv("NEUROKERNEL_TRAIN_FEATURES")
    if not features:
        raise ModelReleaseError("features 없음: --features 또는 NEUROKERNEL_TRAIN_FEATURES 필요")
    out_dir = getattr(args, "out_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs")
    return run_training_pipeline(
        TrainingPipelineConfig(
            features=features,
            out_dir=out_dir,
            run_name=getattr(args, "run_name", None),
            epochs=getattr(args, "epochs", 50),
            batch_size=getattr(args, "batch_size", 1024),
            device=getattr(args, "device", "cuda"),
            patience=getattr(args, "patience", 10),
            run_gate_ablation=not getattr(args, "skip_gate_ablation", False),
            overwrite=getattr(args, "overwrite_local_run", False),
            strict=True,
        )
    )


def _run_deploy_action(args: argparse.Namespace, *, activate: bool) -> dict[str, Any]:
    run_root = Path(getattr(args, "run_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    run_name = getattr(args, "run_name", None)
    if not run_name:
        raise ModelReleaseError("실행 이름 필요")
    run_dir = run_root / run_name
    if not run_dir.exists():
        raise ModelReleaseError(f"실행 없음: {run_dir}")
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")
    release = deploy_training_run(
        run_dir,
        release_name=run_name,
        remote_host=remote_host,
        remote_project=remote_project,
        activate=activate,
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    return {
        "status": "deployed_and_activated" if activate else "deployed",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "release": release,
    }


def _run_deploy_best_action(args: argparse.Namespace) -> dict[str, Any]:
    comparison = _run_compare(args)
    best = comparison.get("best")
    if not best:
        raise ModelReleaseError("배포할 최고 모델 없음")
    best_name = best["run_name"]
    current_name = comparison.get("current_name")
    if current_name == best_name:
        return {
            "status": "already_current",
            "current_name": current_name,
            "best": best,
            "comparison": comparison,
        }

    run_root = Path(getattr(args, "run_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    run_dir = run_root / best_name
    if not run_dir.exists():
        raise ModelReleaseError(f"최고 모델 실행 폴더 없음: {run_dir}")
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")
    release = deploy_training_run(
        run_dir,
        release_name=best_name,
        remote_host=remote_host,
        remote_project=remote_project,
        activate=True,
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    return {
        "status": "deployed_and_activated",
        "previous_current": current_name,
        "run_name": best_name,
        "run_dir": str(run_dir),
        "best": best,
        "comparison": comparison,
        "release": release,
    }


def _run_benchmark_action(args: argparse.Namespace) -> dict[str, Any]:
    model_path = _resolve_benchmark_model(args)
    out_dir = getattr(args, "out_dir", None)
    if out_dir:
        run_dir = Path(out_dir)
        benchmark_dir = run_dir / "gate_ablation"
    else:
        run_dir = model_path.parent
        benchmark_dir = run_dir / f"gate_ablation_nk_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    result = run_gate_ablation(
        model_path,
        out_dir=benchmark_dir,
        config=GateAblationConfig(
            episodes=getattr(args, "episodes", 50),
            trace_episodes=getattr(args, "trace_episodes", 10),
            max_failures_per_env=getattr(args, "max_failures_per_env", 10),
            strict=getattr(args, "strict", True),
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_ablation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _run_benchmark_current_action(args: argparse.Namespace) -> dict[str, Any]:
    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    current = current_model_release(remote_config)
    current_payload = current.get("current") or {}
    release_name = current_payload.get("release_name") or current_payload.get("run_name") or "current_core"
    safe_release_name = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(release_name))
    benchmark_dir = Path(getattr(args, "out_dir", None) or _default_current_bench_dir(safe_release_name))
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    remote_host = current["remote_host"]
    remote_project = current["remote_project"].rstrip("/")
    _copy_remote_file(remote_host, f"{remote_project}/artifacts/current_world_model.onnx", benchmark_dir / "world_model.onnx")
    _copy_remote_file(remote_host, f"{remote_project}/artifacts/current_world_model.manifest.json", benchmark_dir / "world_model.manifest.json")
    result = run_gate_ablation(
        benchmark_dir / "world_model.onnx",
        out_dir=benchmark_dir / "gate_ablation",
        config=GateAblationConfig(
            episodes=getattr(args, "episodes", 50),
            trace_episodes=getattr(args, "trace_episodes", 10),
            max_failures_per_env=getattr(args, "max_failures_per_env", 10),
            strict=getattr(args, "strict", True),
        ),
    )
    result_path = benchmark_dir / "gate_ablation_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "status": "completed",
        "source": "orangepi_current_model",
        "release_name": release_name,
        "remote_host": remote_host,
        "remote_project": remote_project,
        "run_dir": str(benchmark_dir),
        "gate_ablation_result": str(result_path),
    }
    (benchmark_dir / "current_core_benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest | {"benchmark": result}


def _run_top_models(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(getattr(args, "run_dir", None) or os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    limit = max(1, int(getattr(args, "limit", 5)))
    candidates = [_summarize_training_run(path) for path in sorted(run_root.glob("*/gate_ablation_result.json"))]
    ranked = sorted([item for item in candidates if item is not None], key=_top_sort_key, reverse=True)
    return {
        "status": "ok",
        "run_root": str(run_root),
        "limit": limit,
        "count": len(ranked),
        "top": ranked[:limit],
    }


def _run_status(args: argparse.Namespace) -> dict[str, Any]:
    current = _probe_current(args)
    return {"status": "ok", "current": current, "model_slots": _model_slots_from_current(current), "top": _probe_top(args)}


def _run_current(args: argparse.Namespace) -> dict[str, Any]:
    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    world = current_model_release(remote_config)
    runtime = _runtime_model_slot(remote_config)
    return {**world, "model_slots": {"world": _world_slot(world), "runtime": runtime}}


def _model_slots_from_current(current: dict[str, Any] | None) -> dict[str, Any]:
    if not current:
        return {
            "world": {"slot": "world", "status": "unknown"},
            "runtime": {"slot": "runtime_action", "status": "unknown"},
        }
    slots = current.get("model_slots")
    if isinstance(slots, dict):
        return slots
    return {"world": _world_slot(current), "runtime": {"slot": "runtime_action", "status": "unknown"}}


def _world_slot(current: dict[str, Any]) -> dict[str, Any]:
    payload = current.get("current") or {}
    return {
        "slot": "world",
        "status": payload.get("status") or current.get("status") or "unknown",
        "release_name": _current_name(payload),
        "primary_model": payload.get("primary_model"),
        "remote_release_dir": payload.get("remote_release_dir"),
        "data_origin": "simulated_world_pretrain",
    }


def _runtime_model_slot(config: ModelReleaseRemoteConfig) -> dict[str, Any]:
    remote_host = config.remote_host or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (config.remote_project or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    model_path = f"{remote_project}/artifacts/current_runtime_action_model.pt"
    manifest_path = f"{remote_project}/artifacts/current_runtime_action_model.manifest.json"
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(config.ssh_connect_timeout or 10)}"]
    probe = subprocess.run(
        ["ssh", *ssh_options, str(remote_host), f"test -f { _sh_quote(model_path) } && echo present || echo missing"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return {
            "slot": "runtime_action",
            "status": "unknown",
            "remote_host": remote_host,
            "model": model_path,
            "error": (probe.stderr or probe.stdout or "").strip(),
        }
    status = "active" if "present" in (probe.stdout or "") else "missing"
    result: dict[str, Any] = {
        "slot": "runtime_action",
        "status": status,
        "remote_host": remote_host,
        "model": model_path,
        "manifest": manifest_path,
        "data_origin": "runtime_experience_log",
    }
    if status == "active":
        manifest = subprocess.run(
            ["ssh", *ssh_options, str(remote_host), f"cat { _sh_quote(manifest_path) }"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if manifest.returncode == 0 and manifest.stdout.strip():
            try:
                result["manifest_payload"] = json.loads(manifest.stdout)
            except json.JSONDecodeError:
                result["manifest_error"] = "invalid_json"
    return result


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    remote_config = ModelReleaseRemoteConfig(
        remote_host=getattr(args, "remote_host", None),
        remote_project=getattr(args, "remote_project", None),
        ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
    )
    current = current_model_release(remote_config)
    top = _run_top_models(args)
    best = (top.get("top") or [None])[0]
    current_payload = current.get("current") or {}
    current_name = _current_name(current_payload)
    best_name = best.get("run_name") if best else None
    return {
        "status": "ok",
        "current": current,
        "current_name": current_name,
        "top": top,
        "best": best,
        "best_name": best_name,
        "needs_deploy": bool(best_name and current_name != best_name),
    }


def _run_advanced(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "ok",
        "commands": [
            {"name": "사용데이터", "summary": "오렌지파이 사용 로그를 학습 replay로 변환"},
            {"name": "사용특징", "summary": "runtime replay를 학습 feature로 변환"},
            {"name": "사용학습", "summary": "runtime action model 학습"},
            {"name": "학습 <이름>", "summary": "미니월드 world model 학습 + 벤치"},
            {"name": "벤치 <이름>", "summary": "모델 채점"},
            {"name": "배포적용 <이름>", "summary": "선택 모델을 오렌지파이에 전송 후 활성화"},
        ],
    }


def _default_current_bench_dir(release_name: str) -> Path:
    run_root = Path(os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_root / f"current_core_{release_name}_{stamp}"


def _copy_remote_file(remote_host: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f"{local_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    result = subprocess.run(
        ["scp", f"{remote_host}:{remote_path}", str(temp_path)],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if result.returncode != 0:
        if temp_path.exists():
            temp_path.unlink()
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelReleaseError(f"전송 실패: {detail}")
    try:
        temp_path.replace(local_path)
    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise ModelReleaseError(f"전송 파일 교체 실패: {local_path} ({exc})") from exc


def _copy_local_file(remote_host: str, local_path: Path, remote_path: str, *, timeout: int = 10) -> None:
    if not local_path.exists():
        raise ModelReleaseError(f"로컬 파일 없음: {local_path}")
    remote_dir = str(Path(remote_path).parent).replace("\\", "/")
    temp_remote = f"{remote_path}.tmp"
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout)}"]
    mkdir = subprocess.run(
        ["ssh", *ssh_options, remote_host, f"mkdir -p {_sh_quote(remote_dir)}"],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if mkdir.returncode != 0:
        detail = (mkdir.stderr or mkdir.stdout or "").strip()
        raise ModelReleaseError(f"runtime 슬롯 디렉터리 생성 실패: {detail}")
    copied = subprocess.run(
        ["scp", str(local_path), f"{remote_host}:{temp_remote}"],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if copied.returncode != 0:
        detail = (copied.stderr or copied.stdout or "").strip()
        raise ModelReleaseError(f"runtime 슬롯 전송 실패: {detail}")
    moved = subprocess.run(
        ["ssh", *ssh_options, remote_host, f"mv {_sh_quote(temp_remote)} {_sh_quote(remote_path)}"],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if moved.returncode != 0:
        detail = (moved.stderr or moved.stdout or "").strip()
        raise ModelReleaseError(f"runtime 슬롯 활성화 실패: {detail}")


def _resolve_benchmark_model(args: argparse.Namespace) -> Path:
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        model_path = Path(explicit_model)
    else:
        run_name = getattr(args, "run_name", None)
        if not run_name:
            raise ModelReleaseError("모델/실행 필요")
        run_root = Path(os.getenv("NEUROKERNEL_TRAIN_RUN_DIR", "artifacts/training_runs"))
        model_path = run_root / run_name / "world_model.onnx"
    if not model_path.exists():
        raise ModelReleaseError(f"모델 없음: {model_path}")
    return model_path


def _summarize_training_run(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    aggregate = payload.get("aggregate") or {}
    macro = aggregate.get("macro_success_rate") or {}
    interpretation = payload.get("interpretation") or {}
    return {
        "run_name": path.parent.name,
        "report": str(path),
        "verdict": interpretation.get("verdict") or "unknown",
        "hybrid_veto_success": _as_float(macro.get("hybrid_veto")),
        "hybrid_success": _as_float(macro.get("hybrid")),
        "prior_success": _as_float(macro.get("prior_only")),
        "model_only_success": _as_float(macro.get("model_only")),
        "model_needed_signal_group_count": int(aggregate.get("model_needed_signal_group_count") or 0),
        "hybrid_veto_gain_over_prior": _as_float(aggregate.get("hybrid_veto_gain_over_prior")),
        "hybrid_gain_over_prior": _as_float(aggregate.get("hybrid_gain_over_prior")),
        "groups_with_model_needed_signal": aggregate.get("groups_with_model_needed_signal") or [],
    }


def _top_sort_key(item: dict[str, Any]) -> tuple[float, int, float]:
    return (item["hybrid_veto_success"], item["model_needed_signal_group_count"], item["hybrid_veto_gain_over_prior"])


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _print_result(action: str, result: dict[str, Any], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    action = _normalize_action(action)
    if action in {"runtime-auto", "runtime-cycle"}:
        _print_runtime_auto(result)
    elif action == "data":
        _print_data(result)
    elif action == "runtime-data":
        _print_runtime_data(result)
    elif action == "runtime-features":
        _print_runtime_features(result)
    elif action == "runtime-train":
        _print_runtime_train(result)
    elif action == "deploy-runtime":
        _print_deploy_runtime(result)
    elif action == "check":
        _print_check(result)
    elif action == "train":
        _print_train(result)
    elif action in {"deploy", "deploy-use"}:
        _print_deploy(action, result)
    elif action == "deploy-best":
        _print_deploy_best(result)
    elif action == "bench":
        _print_bench(result)
    elif action == "bench-current":
        _print_bench_current(result)
    elif action == "top":
        _print_top(result)
    elif action == "compare":
        _print_compare(result)
    elif action == "list":
        _print_list(result)
    elif action == "current":
        _print_current(result)
    elif action == "use":
        print(_ok("적용", result["release_name"]))
    elif action == "status":
        _print_status(result)
    elif action == "advanced":
        _print_advanced(result)


def _print_data(result: dict[str, Any]) -> None:
    validation = result.get("validation") or {}
    gates = result.get("gates") or {}
    label = "통과" if result.get("ready_for_training") else "실패"
    print(_ok("데이터", label) if result.get("ready_for_training") else _warn("데이터", label))
    print(_kv("파일", result.get("features")))
    print(_kv("행", validation.get("rows")))
    print(_kv("입력", validation.get("input_dim")))
    print(_kv("출력", validation.get("target_dim")))
    print(_kv("스키마", validation.get("schema_version")))
    shortages = gates.get("audit_shortage_report") or []
    if shortages:
        print(_kv("부족", len(shortages)))
        for item in shortages[:5]:
            family = item.get("family") or item.get("gate") or item.get("kind") or "unknown"
            actual = item.get("actual_ratio")
            target = item.get("target_min_ratio") or item.get("threshold")
            print(f"- {family}: actual={actual} target={target}")


def _print_runtime_auto(result: dict[str, Any]) -> None:
    print(_ok("자동", result.get("status", "completed")))
    artifacts = result.get("artifacts") or {}
    steps = result.get("steps") or {}
    data = steps.get("runtime_data") or {}
    features = steps.get("runtime_features") or {}
    train = steps.get("runtime_train") or {}
    print(_kv("출처", data.get("source")))
    print(_kv("행", (data.get("validation") or {}).get("rows")))
    print(_kv("입력", (features.get("validation") or {}).get("input_dim")))
    print(_kv("모델", artifacts.get("model")))
    test = train.get("test") or {}
    if test:
        print(_kv("성공률", f"{float(test.get('success_accuracy', 0.0)):.3f}"))
        print(_kv("보상오차", f"{float(test.get('reward_mae', 0.0)):.3f}"))
    deploy = steps.get("runtime_deploy") or {}
    if deploy:
        print(_kv("runtime", deploy.get("remote_model")))
    print(_kv("리포트", result.get("report")))


def _print_runtime_data(result: dict[str, Any]) -> None:
    validation = result.get("validation") or {}
    gates = result.get("gates") or {}
    label = "통과" if result.get("ready_for_runtime_training") else "실패"
    print(_ok("사용데이터", label) if result.get("ready_for_runtime_training") else _warn("사용데이터", label))
    print(_kv("출처", result.get("source")))
    print(_kv("DB", result.get("db")))
    print(_kv("파일", result.get("out")))
    print(_kv("행", validation.get("rows")))
    print(_kv("성공", validation.get("success_rows")))
    print(_kv("실패", validation.get("failure_rows")))
    print(_kv("스키마", validation.get("schema_version")))
    failed = [name for name, item in (gates.get("gates") or {}).items() if not item.get("passed")]
    if failed:
        print(_kv("막힘", ", ".join(failed)))


def _print_runtime_features(result: dict[str, Any]) -> None:
    validation = result.get("validation") or {}
    gates = result.get("gates") or {}
    label = "통과" if result.get("ready_for_runtime_model_training") else "실패"
    print(_ok("사용특징", label) if result.get("ready_for_runtime_model_training") else _warn("사용특징", label))
    print(_kv("원본", result.get("replay")))
    print(_kv("파일", result.get("features")))
    print(_kv("행", validation.get("rows")))
    print(_kv("입력", validation.get("input_dim")))
    print(_kv("출력", validation.get("target_dim")))
    print(_kv("스키마", validation.get("schema_version")))
    warnings = gates.get("warnings") or []
    for warning in warnings[:3]:
        print(_kv("주의", warning.get("message") or warning.get("kind")))


def _print_runtime_train(result: dict[str, Any]) -> None:
    print(_ok("사용학습", result.get("status", "completed")))
    print(_kv("특징", result.get("features")))
    print(_kv("모델", result.get("checkpoint") or result.get("out")))
    print(_kv("장치", result.get("device")))
    print(_kv("최적", result.get("best_epoch")))
    test = result.get("test") or {}
    if test:
        print(_kv("성공률", f"{float(test.get('success_accuracy', 0.0)):.3f}"))
        print(_kv("보상오차", f"{float(test.get('reward_mae', 0.0)):.3f}"))


def _print_deploy_runtime(result: dict[str, Any]) -> None:
    print(_ok("runtime", result.get("status", "deployed_and_activated")))
    print(_kv("모델", result.get("remote_model")))
    print(_kv("manifest", result.get("remote_manifest")))


def _print_check(result: dict[str, Any]) -> None:
    preflight = result["preflight"]
    print(_ok("확인", result["run_name"]))
    print(_kv("로컬", preflight["local_run_dir"]))
    print(_kv("엣지", preflight["remote_release_dir"]))


def _print_train(result: dict[str, Any]) -> None:
    print(_ok("학습", result["run_name"]))
    print(_kv("실행", result["run_dir"]))
    gate = result.get("artifacts", {}).get("gate_ablation")
    if gate:
        print(_kv("벤치", gate))


def _print_deploy(action: str, result: dict[str, Any]) -> None:
    label = "배포적용" if action == "deploy-use" else "배포"
    print(_ok(label, result["run_name"]))
    print(_kv("엣지", result["release"]["remote_release_dir"]))


def _print_deploy_best(result: dict[str, Any]) -> None:
    if result.get("status") == "already_current":
        print(_ok("배포", "변경 없음"))
        print(_kv("현재", result.get("current_name")))
        return
    print(_ok("배포", result.get("run_name")))
    print(_kv("이전", result.get("previous_current")))
    print(_kv("엣지", (result.get("release") or {}).get("remote_release_dir")))


def _print_bench(result: dict[str, Any]) -> None:
    interpretation = result.get("interpretation", {})
    print(_ok("벤치", interpretation.get("verdict") or "완료"))
    _print_score_line(result)
    if result.get("out"):
        print(_kv("결과", result["out"]))


def _print_bench_current(result: dict[str, Any]) -> None:
    print(_ok("현재벤치", result["release_name"]))
    _print_score_line(result.get("benchmark", {}))
    print(_kv("실행", result["run_dir"]))


def _print_top(result: dict[str, Any]) -> None:
    top = result.get("top") or []
    if not top:
        print(_warn("순위", "비어있음"))
        if result.get("run_root"):
            print(_kv("위치", result["run_root"]))
        return
    print(_style("순위", "cyan"))
    for index, item in enumerate(top, start=1):
        print(
            f"{index:<2} {item['run_name']:<42} "
            f"혼합={item['hybrid_veto_success']:.3f} "
            f"규칙={item['prior_success']:.3f} "
            f"상승={item['hybrid_veto_gain_over_prior']:.3f} "
            f"신호={item['model_needed_signal_group_count']} "
            f"{_verdict_text(item['verdict'])}"
        )


def _print_compare(result: dict[str, Any]) -> None:
    print(_style("비교", "cyan"))
    print(_kv("현재", result.get("current_name")))
    best = result.get("best") or {}
    if best:
        print(_kv("최고", f"{best.get('run_name')} 혼합={_as_float(best.get('hybrid_veto_success')):.3f} 상승={_as_float(best.get('hybrid_veto_gain_over_prior')):.3f}"))
    else:
        print(_kv("최고", "비어있음"))
    print(_kv("배포", "필요" if result.get("needs_deploy") else "불필요"))


def _print_list(result: dict[str, Any]) -> None:
    releases = result.get("releases") or []
    print(_style("목록", "cyan"))
    if not releases:
        print("비어있음")
        return
    for release in releases:
        print(f"- {release}")


def _print_current(result: dict[str, Any]) -> None:
    print(_style("현재", "cyan"))
    current = result.get("current") or {}
    print(_kv("적용", _current_name(current)))
    if current.get("deployed_at"):
        print(_kv("시간", current["deployed_at"]))
    if current.get("remote_release_dir"):
        print(_kv("위치", current["remote_release_dir"]))
    _print_model_slots(result.get("model_slots"))


def _print_status(result: dict[str, Any]) -> None:
    print(_style("상태", "cyan"))
    print(_kv("현재", _short_current(result.get("current"))))
    print(_kv("최고", _short_best(result.get("top"))))
    _print_model_slots(result.get("model_slots"))


def _print_model_slots(slots: dict[str, Any] | None) -> None:
    if not isinstance(slots, dict):
        return
    world = slots.get("world") if isinstance(slots.get("world"), dict) else {}
    runtime = slots.get("runtime") if isinstance(slots.get("runtime"), dict) else {}
    if world:
        print(_kv("world", f"{world.get('status', 'unknown')} {_current_name(world)}".strip()))
    if runtime:
        print(_kv("runtime", f"{runtime.get('status', 'unknown')} {runtime.get('model') or ''}".strip()))


def _print_advanced(result: dict[str, Any]) -> None:
    print(_style("고급", "cyan"))
    for command in result.get("commands") or []:
        print(f"- {command['name']}: {command['summary']}")


def _print_score_line(result: dict[str, Any]) -> None:
    aggregate = result.get("aggregate") or {}
    macro = aggregate.get("macro_success_rate") or {}
    if not macro:
        return
    print(
        _kv(
            "점수",
            f"혼합={_as_float(macro.get('hybrid_veto')):.3f} "
            f"규칙={_as_float(macro.get('prior_only')):.3f} "
            f"상승={_as_float(aggregate.get('hybrid_veto_gain_over_prior')):.3f} "
            f"신호={int(aggregate.get('model_needed_signal_group_count') or 0)}",
        )
    )


def _short_current(result: dict[str, Any] | None) -> str:
    if not result:
        return "확인 실패"
    return _current_name(result.get("current") or {})


def _current_name(current: dict[str, Any]) -> str:
    return str(current.get("release_name") or current.get("run_name") or "모름")


def _short_best(result: dict[str, Any] | None) -> str:
    top = (result or {}).get("top") or []
    if not top:
        return "비어있음"
    item = top[0]
    return f"{item['run_name']} 혼합={item['hybrid_veto_success']:.3f} 상승={item['hybrid_veto_gain_over_prior']:.3f}"


def _verdict_text(verdict: str) -> str:
    return {
        "hybrid_adds_value": "모델기여",
        "prior_dominated": "규칙우세",
        "model_only_dominated": "모델우세",
        "unknown": "모름",
    }.get(verdict, verdict)


def _ok(label: str, value: str) -> str:
    return f"{_style(label, 'green')} {value}"


def _warn(label: str, value: str) -> str:
    return f"{_style(label, 'yellow')} {value}"


def _kv(key: str, value: Any) -> str:
    return f"{key:<5} {value}"


def _style(text: str, color: str) -> str:
    if not sys.stdout.isatty():
        return text
    codes = {"bold": "1", "cyan": "36", "green": "32", "yellow": "33"}
    code = codes.get(color)
    return f"\033[{code}m{text}\033[0m" if code else text


if __name__ == "__main__":
    raise SystemExit(main())
