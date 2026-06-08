from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from neurokernel_seed.eval.gate_ablation import GateAblationConfig, run_gate_ablation
from neurokernel_seed.harness.service import HarnessService
from neurokernel_seed.harness.training_worker import TrainingWorkerError, normalize_training_command
from neurokernel_seed.harness.runtime_policy import RuntimeActionPolicy, RuntimePolicyConfig
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
from neurokernel_seed.model.runtime_action import eval_runtime_action_checkpoint, train_runtime_action_model
from neurokernel_seed.nk_console import menu as nk_menu
from neurokernel_seed.nk_console.dashboard import DashboardController
from neurokernel_seed.replay.real_world_transitions import export_real_world_transitions
from neurokernel_seed.replay.runtime_dataset import RuntimeReplayEtlConfig, run_runtime_replay_etl
from neurokernel_seed.replay.runtime_features import check_runtime_feature_gates, export_runtime_features, validate_runtime_features
from neurokernel_seed.replay.slot_dataset import check_slot_dataset_gates, validate_slot_features


MODEL_BENCHMARK_SCHEMA_VERSION = "neurokernel-model-benchmark-v1"


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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "자주 쓰는 명령:\n"
            "  nk 시드 --source edge            런타임 초기 데이터만 준비\n"
            "  nk runtime-train --device cuda   준비된 runtime_features로 CUDA 학습\n"
            "  nk 런타임배포 --model artifacts/runtime_action_model.pt\n"
            "  nk current                       world/runtime 슬롯 상태 확인\n"
            "\n"
            "영문 명령도 계속 지원합니다: runtime-seed, runtime-probe, runtime-auto, runtime-cycle, runtime-bench, runtime-compare, current-bench-all, deploy-runtime, deploy-all-best, deploy-use, bench-current, top"
        ),
    )
    sub = parser.add_subparsers(dest="action", metavar="명령")

    for name in ("runtime-pipeline", "autopilot", "오토파일럿", "런타임학습", "풀자동", "전체자동"):
        item = sub.add_parser(name)
        _add_runtime_pipeline_options(item)

    for name in ("runtime-auto", "auto", "자동", "학습", "파이프라인", "사용자동"):
        item = sub.add_parser(name)
        _add_runtime_auto_options(item)
        item.set_defaults(deploy_runtime=False)

    for name in ("runtime-cycle", "auto-deploy", "learn-deploy", "배포학습", "학습배포", "순환"):
        item = sub.add_parser(name)
        _add_runtime_auto_options(item)
        item.set_defaults(deploy_runtime=True)

    for name in ("runtime-seed", "seed-runtime", "시드", "초기데이터"):
        item = sub.add_parser(name)
        _add_runtime_seed_options(item)

    for name in ("runtime-probe", "probe-runtime", "런타임프로브", "후보프로브"):
        item = sub.add_parser(name)
        _add_runtime_probe_options(item)

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

    for name in ("data-audit", "audit-data"):
        item = sub.add_parser(name)
        item.add_argument("--source", choices=["edge", "local"], default=os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
        item.add_argument("--db", default=os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
        item.add_argument("--remote-db", default=os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db"))
        item.add_argument("--cache-db", default=os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE"))
        _add_remote_options(item)
        item.add_argument("--json", action="store_true")

    for name in ("training-pending", "train-pending", "학습대기", "학습후보"):
        item = sub.add_parser(name)
        _add_training_work_options(item)
        item.add_argument("--json", action="store_true")

    for name in ("training-run", "train-run", "학습실행", "후보학습"):
        item = sub.add_parser(name)
        _add_training_work_options(item)
        item.add_argument("--work-id")
        item.add_argument("--index", type=int, default=1)
        item.add_argument("--device", choices=["auto", "cpu", "cuda"])
        item.add_argument("--dry-run", action="store_true")
        item.add_argument("--json", action="store_true")

    for name in ("real-world-transitions", "world-real-data"):
        item = sub.add_parser(name)
        item.add_argument("--source", choices=["edge", "local"], default=os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
        item.add_argument("--db", default=os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
        item.add_argument("--remote-db", default=os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db"))
        item.add_argument("--cache-db", default=os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE"))
        _add_remote_options(item)
        item.add_argument("--out", default=os.getenv("NEUROKERNEL_REAL_WORLD_TRANSITIONS_OUT", "data/model_ready/real_world_transitions.jsonl"))
        item.add_argument("--limit", type=int)
        item.add_argument("--json", action="store_true")

    for name in ("runtime-train", "사용학습", "현실학습", "학습만", "런타임학습만"):
        item = sub.add_parser(name)
        _add_runtime_train_options(item)

    for name in ("runtime-bench", "bench-runtime", "런타임벤치"):
        item = sub.add_parser(name)
        _add_runtime_benchmark_options(item)

    for name in ("runtime-bench-current", "bench-current-runtime", "현재런타임벤치"):
        item = sub.add_parser(name)
        _add_runtime_benchmark_options(item)
        _add_remote_options(item)

    for name in ("runtime-compare", "compare-runtime", "런타임비교"):
        item = sub.add_parser(name)
        _add_runtime_compare_options(item)

    for name in ("deploy-runtime", "runtime-deploy", "runtime-use", "런타임배포"):
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

    for name in ("deploy-all-best", "integrated-deploy", "통합배포", "전체배포"):
        item = sub.add_parser(name)
        _add_integrated_deploy_options(item)

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

    for name in ("current-bench-all", "bench-current-all", "현행벤치", "현재모델벤치"):
        item = sub.add_parser(name)
        _add_current_benchmark_all_options(item)

    for name in ("compare", "비교", "벤치비교"):
        item = sub.add_parser(name)
        _add_remote_options(item)
        item.add_argument("--run-dir")
        item.add_argument("--limit", type=int, default=5)
        item.add_argument("--refresh-current-bench", action="store_true")
        item.add_argument("--min-delta", type=float, default=float(os.getenv("NEUROKERNEL_WORLD_BENCH_MIN_DELTA", "0.005")))
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


def _add_runtime_ranking_quality_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-ranking-groups", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_RANKING_GROUPS", "0")))
    parser.add_argument("--min-top1-action-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_TOP1_ACTION_ACCURACY", "0.0")))


def _add_runtime_pipeline_options(parser: argparse.ArgumentParser) -> None:
    _add_runtime_auto_options(parser)
    parser.set_defaults(deploy_runtime=True, device="cuda", epochs=100)
    parser.add_argument("--seed-cycles", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_PIPELINE_SEED_CYCLES", "0")))
    parser.add_argument("--no-seed-failures", dest="include_failures", action="store_false")
    parser.set_defaults(include_failures=True)
    parser.add_argument("--min-success-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_SUCCESS_ACCURACY", "0.75")))
    parser.add_argument("--max-reward-mae", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MAX_REWARD_MAE", "0.35")))
    parser.set_defaults(min_actions=4)
    parser.add_argument("--min-known-success-rows", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_KNOWN_SUCCESS_ROWS", "5")))
    _add_runtime_ranking_quality_options(parser)
    parser.add_argument("--benchmark-split", choices=["train", "test"], default=os.getenv("NEUROKERNEL_RUNTIME_BENCH_SPLIT", "test"))
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_BENCH_MIN_DELTA", "0.01")))
    parser.add_argument("--force-deploy", action="store_true")


def _add_runtime_seed_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["edge", "local"], default=os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
    parser.add_argument("--db", default=os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db"))
    parser.add_argument("--remote-db", default=os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db"))
    parser.add_argument("--cache-db", default=os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE"))
    _add_remote_options(parser)
    parser.add_argument("--remote-python", default=os.getenv("NEUROKERNEL_EDGE_PYTHON"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--profile", choices=["readonly-basic"], default="readonly-basic")
    parser.add_argument("--target", choices=["local", "orangepi5"], default="orangepi5")
    parser.add_argument("--cycles", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_SEED_CYCLES", "8")))
    parser.add_argument("--replay-out", default=os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    parser.add_argument("--features-out", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--min-actions", type=int, default=4)
    parser.add_argument("--failures", dest="include_failures", action="store_true")
    parser.add_argument("--no-failures", dest="include_failures", action="store_false")
    parser.set_defaults(include_failures=True)
    parser.add_argument("--export", dest="export_dataset", action="store_true")
    parser.add_argument("--no-export", dest="export_dataset", action="store_false")
    parser.set_defaults(export_dataset=True)
    parser.add_argument("--json", action="store_true")


def _add_runtime_probe_options(parser: argparse.ArgumentParser) -> None:
    _add_runtime_seed_options(parser)
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_PROBE_MAX_CANDIDATES", "4")))


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


def _add_runtime_benchmark_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    parser.add_argument("--features", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--split", choices=["train", "test"], default=os.getenv("NEUROKERNEL_RUNTIME_BENCH_SPLIT", "test"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default=os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    parser.add_argument("--candidate-kind", default="local_candidate")
    parser.add_argument("--min-success-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_SUCCESS_ACCURACY", "0.75")))
    parser.add_argument("--max-reward-mae", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MAX_REWARD_MAE", "0.35")))
    parser.add_argument("--min-known-success-rows", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_KNOWN_SUCCESS_ROWS", "5")))
    _add_runtime_ranking_quality_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_runtime_compare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    parser.add_argument("--features", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--split", choices=["train", "test"], default=os.getenv("NEUROKERNEL_RUNTIME_BENCH_SPLIT", "test"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default=os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    parser.add_argument("--min-success-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_SUCCESS_ACCURACY", "0.75")))
    parser.add_argument("--max-reward-mae", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MAX_REWARD_MAE", "0.35")))
    parser.add_argument("--min-known-success-rows", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_KNOWN_SUCCESS_ROWS", "5")))
    _add_runtime_ranking_quality_options(parser)
    parser.add_argument("--min-delta", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_BENCH_MIN_DELTA", "0.01")))
    _add_remote_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_current_benchmark_all_options(parser: argparse.ArgumentParser) -> None:
    _add_remote_options(parser)
    parser.add_argument("--features", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--split", choices=["train", "test"], default=os.getenv("NEUROKERNEL_RUNTIME_BENCH_SPLIT", "test"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default=os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--trace-episodes", type=int, default=10)
    parser.add_argument("--max-failures-per-env", type=int, default=10)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    parser.add_argument("--min-success-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_SUCCESS_ACCURACY", "0.75")))
    parser.add_argument("--max-reward-mae", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MAX_REWARD_MAE", "0.35")))
    parser.add_argument("--min-known-success-rows", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_KNOWN_SUCCESS_ROWS", "5")))
    _add_runtime_ranking_quality_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_integrated_deploy_options(parser: argparse.ArgumentParser) -> None:
    _add_remote_options(parser)
    parser.add_argument("--run-dir")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    parser.add_argument("--features", default=os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    parser.add_argument("--split", choices=["train", "test"], default=os.getenv("NEUROKERNEL_RUNTIME_BENCH_SPLIT", "test"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", default=os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--trace-episodes", type=int, default=10)
    parser.add_argument("--max-failures-per-env", type=int, default=10)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True, refresh_current_bench=True)
    parser.add_argument("--skip-refresh-current-bench", dest="refresh_current_bench", action="store_false")
    parser.add_argument("--runtime-min-delta", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_BENCH_MIN_DELTA", "0.01")))
    parser.add_argument("--world-min-delta", type=float, default=float(os.getenv("NEUROKERNEL_WORLD_BENCH_MIN_DELTA", "0.005")))
    parser.add_argument("--min-success-accuracy", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MIN_SUCCESS_ACCURACY", "0.75")))
    parser.add_argument("--max-reward-mae", type=float, default=float(os.getenv("NEUROKERNEL_RUNTIME_MAX_REWARD_MAE", "0.35")))
    parser.add_argument("--min-known-success-rows", type=int, default=int(os.getenv("NEUROKERNEL_RUNTIME_MIN_KNOWN_SUCCESS_ROWS", "5")))
    _add_runtime_ranking_quality_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_runtime_deploy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    _add_remote_options(parser)
    parser.add_argument("--json", action="store_true")


def _add_remote_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote-host")
    parser.add_argument("--remote-project")
    parser.add_argument("--ssh-connect-timeout", type=int, default=10)


def _add_training_work_options(parser: argparse.ArgumentParser) -> None:
    _add_remote_options(parser)
    parser.add_argument("--remote-core-url", default=os.getenv("NEUROKERNEL_EDGE_CORE_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--include-status",
        action="append",
        choices=["proposed", "accepted", "planned", "running", "reviewing", "blocked", "failed"],
        help="여러 번 줄 수 있습니다. 기본값은 실행 가능한 학습 후보만 봅니다.",
    )


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
    if args.action == "runtime-pipeline":
        return _run_runtime_pipeline_action(args)
    if args.action in {"runtime-auto", "runtime-cycle"}:
        return _run_runtime_auto_action(args)
    if args.action == "runtime-seed":
        return _run_runtime_seed_action(args)
    if args.action == "runtime-probe":
        return _run_runtime_probe_action(args)
    if args.action == "data":
        return _run_data_action(args)
    if args.action == "runtime-data":
        return _run_runtime_data_action(args)
    if args.action == "runtime-features":
        return _run_runtime_features_action(args)
    if args.action == "data-audit":
        return _run_data_audit_action(args)
    if args.action == "training-pending":
        return _run_training_pending_action(args)
    if args.action == "training-run":
        return _run_training_run_action(args)
    if args.action == "real-world-transitions":
        return _run_real_world_transitions_action(args)
    if args.action == "runtime-train":
        return _run_runtime_training_action(args)
    if args.action == "runtime-bench":
        return _run_runtime_benchmark_action(args)
    if args.action == "runtime-bench-current":
        return _run_runtime_benchmark_current_action(args)
    if args.action == "runtime-compare":
        return _run_runtime_compare_action(args)
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
    if args.action == "deploy-all-best":
        return _run_deploy_all_best_action(args)
    if args.action == "bench":
        return _run_benchmark_action(args)
    if args.action == "bench-current":
        return _run_benchmark_current_action(args)
    if args.action == "current-bench-all":
        return _run_current_benchmark_all_action(args)
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


def _run_runtime_pipeline_action(args: argparse.Namespace) -> dict[str, Any]:
    replay_out = Path(getattr(args, "replay_out", None) or os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl"))
    features_out = Path(getattr(args, "features_out", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    model_out = Path(getattr(args, "model_out", None) or os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    seed_result = None
    seed_cycles = int(getattr(args, "seed_cycles", 0) or 0)
    if seed_cycles > 0:
        seed_result = _run_runtime_seed_action(
            argparse.Namespace(
                action="runtime-seed",
                source=getattr(args, "source", "edge"),
                db=getattr(args, "db", "data/harness.db"),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                remote_db=getattr(args, "remote_db", "data/harness.db"),
                cache_db=getattr(args, "cache_db", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
                remote_python=getattr(args, "remote_python", None),
                project_root=".",
                profile="readonly-basic",
                target="orangepi5" if getattr(args, "source", "edge") == "edge" else "local",
                cycles=seed_cycles,
                include_failures=getattr(args, "include_failures", True),
                export_dataset=False,
                replay_out=str(replay_out),
                features_out=str(features_out),
                test_ratio=getattr(args, "test_ratio", 0.2),
                min_rows=getattr(args, "min_rows", 10),
                min_actions=getattr(args, "min_actions", 1),
            )
        )

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
    _require_stage_ready(data_result, "런타임 데이터", "ready_for_runtime_training")

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
    _require_stage_ready(feature_result, "런타임 특징", "ready_for_runtime_model_training")

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
            device=getattr(args, "device", "cuda"),
            patience=getattr(args, "patience", 10),
            min_rows=getattr(args, "min_rows", 10),
            min_actions=getattr(args, "min_actions", 1),
        )
    )
    quality = _runtime_quality_gate(train_result, args)
    benchmark_result = None
    compare_result = None
    deploy_result = None
    current_after = None
    deploy_requested = bool(getattr(args, "deploy_runtime", True))
    if deploy_requested:
        compare_result = _run_runtime_compare_action(
            argparse.Namespace(
                action="runtime-compare",
                model=str(train_result.get("checkpoint") or model_out),
                features=str(features_out),
                split=getattr(args, "benchmark_split", "test"),
                device=getattr(args, "device", "auto"),
                out_dir=getattr(args, "benchmark_out_dir", os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
                min_success_accuracy=getattr(args, "min_success_accuracy", 0.75),
                max_reward_mae=getattr(args, "max_reward_mae", 0.35),
                min_known_success_rows=getattr(args, "min_known_success_rows", 5),
                min_ranking_groups=getattr(args, "min_ranking_groups", 0),
                min_top1_action_accuracy=getattr(args, "min_top1_action_accuracy", 0.0),
                min_delta=getattr(args, "min_delta", 0.01),
            )
        )
        benchmark_result = compare_result.get("local")
    else:
        benchmark_result = _run_runtime_benchmark_action(
            argparse.Namespace(
                action="runtime-bench",
                model=str(train_result.get("checkpoint") or model_out),
                features=str(features_out),
                split=getattr(args, "benchmark_split", "test"),
                device=getattr(args, "device", "auto"),
                out_dir=getattr(args, "benchmark_out_dir", os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")),
                candidate_kind="local_candidate",
                min_success_accuracy=getattr(args, "min_success_accuracy", 0.75),
                max_reward_mae=getattr(args, "max_reward_mae", 0.35),
                min_known_success_rows=getattr(args, "min_known_success_rows", 5),
                min_ranking_groups=getattr(args, "min_ranking_groups", 0),
                min_top1_action_accuracy=getattr(args, "min_top1_action_accuracy", 0.0),
            )
        )
    deploy_allowed = bool(((compare_result or {}).get("decision") or {}).get("local_wins")) or bool(getattr(args, "force_deploy", False))
    if deploy_requested and (quality["passed"] or bool(getattr(args, "force_deploy", False))) and deploy_allowed:
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

    status = "deployed" if deploy_result else "trained"
    if deploy_requested and not deploy_result:
        status = "blocked_by_quality_gate" if not quality["passed"] and not bool(getattr(args, "force_deploy", False)) else "blocked_by_benchmark_compare"
    report = {
        "status": status,
        "pipeline": "runtime_action_autopilot_v2_benchmark_gated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality": quality,
        "steps": {
            "runtime_seed": seed_result,
            "runtime_data": data_result,
            "runtime_features": feature_result,
            "runtime_train": train_result,
            "runtime_benchmark": benchmark_result,
            "runtime_compare": compare_result,
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
    report_path = _write_runtime_pipeline_report(report)
    return {**report, "report": str(report_path)}


def _runtime_quality_gate(train_result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    test = train_result.get("test") if isinstance(train_result.get("test"), dict) else {}
    success_accuracy = _as_float(test.get("success_accuracy"))
    reward_mae = _as_float(test.get("reward_mae"))
    known_success_rows = _as_float(test.get("known_success_rows"))
    ranking_evaluable_groups = _as_float(test.get("ranking_evaluable_groups"))
    top1_action_accuracy = _as_float(test.get("top1_action_accuracy", test.get("ranking_top1_accuracy")))
    min_success = float(getattr(args, "min_success_accuracy", 0.75))
    max_reward = float(getattr(args, "max_reward_mae", 0.35))
    min_known = int(getattr(args, "min_known_success_rows", 5))
    min_ranking_groups = int(getattr(args, "min_ranking_groups", 0))
    min_top1 = float(getattr(args, "min_top1_action_accuracy", 0.0))
    checks = {
        "success_accuracy": {"passed": success_accuracy >= min_success, "actual": success_accuracy, "threshold": min_success},
        "reward_mae": {"passed": reward_mae <= max_reward, "actual": reward_mae, "threshold": max_reward},
        "known_success_rows": {"passed": known_success_rows >= min_known, "actual": known_success_rows, "threshold": min_known},
        "ranking_evaluable_groups": {"passed": ranking_evaluable_groups >= min_ranking_groups, "actual": ranking_evaluable_groups, "threshold": min_ranking_groups},
        "top1_action_accuracy": {"passed": top1_action_accuracy >= min_top1, "actual": top1_action_accuracy, "threshold": min_top1},
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "passed": passed,
        "forced": bool(getattr(args, "force_deploy", False)),
        "summary": "품질 게이트 통과" if passed else "품질 게이트 미통과",
        "checks": checks,
    }


def _write_runtime_pipeline_report(report: dict[str, Any]) -> Path:
    report_dir = Path(os.getenv("NEUROKERNEL_RUNTIME_PIPELINE_DIR", "artifacts/runtime_pipeline"))
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"runtime_action_autopilot_{stamp}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    report_path.write_text(payload, encoding="utf-8")
    latest_path = report_dir / "latest_runtime_action_autopilot.json"
    latest_path.write_text(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _run_runtime_seed_action(args: argparse.Namespace) -> dict[str, Any]:
    source = str(getattr(args, "source", "edge"))
    if source == "edge":
        return _run_runtime_seed_edge_action(args)
    if source != "local":
        raise ModelReleaseError(f"unknown runtime seed source: {source}")
    seed = _run_runtime_seed_local_action(args)
    result: dict[str, Any] = {
        "status": "completed",
        "source": "local",
        "seed": seed,
        "artifacts": {},
    }
    if bool(getattr(args, "export_dataset", True)):
        replay_out = Path(getattr(args, "replay_out", "data/model_ready/runtime_replay.jsonl"))
        features_out = Path(getattr(args, "features_out", "data/model_ready/runtime_features.jsonl"))
        data_result = _run_runtime_data_action(
            argparse.Namespace(
                action="runtime-data",
                source="local",
                db=getattr(args, "db", "data/harness.db"),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                remote_db=getattr(args, "remote_db", "data/harness.db"),
                cache_db=getattr(args, "cache_db", None),
                out=str(replay_out),
                limit=None,
                min_rows=getattr(args, "min_rows", 10),
                allow_no_execution=False,
            )
        )
        feature_result = _run_runtime_features_action(
            argparse.Namespace(
                action="runtime-features",
                replay=str(replay_out),
                out=str(features_out),
                test_ratio=getattr(args, "test_ratio", 0.2),
                min_rows=getattr(args, "min_rows", 10),
                min_actions=getattr(args, "min_actions", 4),
            )
        )
        result["runtime_data"] = data_result
        result["runtime_features"] = feature_result
        result["artifacts"] = {"replay": str(replay_out), "features": str(features_out)}
        result["ready_for_runtime_model_training"] = bool(feature_result.get("ready_for_runtime_model_training"))
    return result


def _run_runtime_seed_edge_action(args: argparse.Namespace) -> dict[str, Any]:
    remote_seed = _run_remote_runtime_seed_command(args)
    result: dict[str, Any] = {
        "status": "completed",
        "source": "edge",
        "remote_seed": remote_seed,
        "artifacts": {},
    }
    if bool(getattr(args, "export_dataset", True)):
        replay_out = Path(getattr(args, "replay_out", "data/model_ready/runtime_replay.jsonl"))
        features_out = Path(getattr(args, "features_out", "data/model_ready/runtime_features.jsonl"))
        data_result = _run_runtime_data_action(
            argparse.Namespace(
                action="runtime-data",
                source="edge",
                db=getattr(args, "db", "data/harness.db"),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                remote_db=getattr(args, "remote_db", "data/harness.db"),
                cache_db=getattr(args, "cache_db", None),
                out=str(replay_out),
                limit=None,
                min_rows=getattr(args, "min_rows", 10),
                allow_no_execution=False,
            )
        )
        feature_result = _run_runtime_features_action(
            argparse.Namespace(
                action="runtime-features",
                replay=str(replay_out),
                out=str(features_out),
                test_ratio=getattr(args, "test_ratio", 0.2),
                min_rows=getattr(args, "min_rows", 10),
                min_actions=getattr(args, "min_actions", 4),
            )
        )
        result["runtime_data"] = data_result
        result["runtime_features"] = feature_result
        result["artifacts"] = {"replay": str(replay_out), "features": str(features_out)}
        result["ready_for_runtime_model_training"] = bool(feature_result.get("ready_for_runtime_model_training"))
    return result


def _run_runtime_probe_action(args: argparse.Namespace) -> dict[str, Any]:
    source = str(getattr(args, "source", "edge"))
    if source == "edge":
        return _run_runtime_probe_edge_action(args)
    if source != "local":
        raise ModelReleaseError(f"unknown runtime probe source: {source}")
    probe = _run_runtime_probe_local_action(args)
    result: dict[str, Any] = {
        "status": "completed",
        "source": "local",
        "probe": probe,
        "artifacts": {},
    }
    if bool(getattr(args, "export_dataset", True)):
        replay_out = Path(getattr(args, "replay_out", "data/model_ready/runtime_replay.jsonl"))
        features_out = Path(getattr(args, "features_out", "data/model_ready/runtime_features.jsonl"))
        data_result = _run_runtime_data_action(
            argparse.Namespace(
                action="runtime-data",
                source="local",
                db=getattr(args, "db", "data/harness.db"),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                remote_db=getattr(args, "remote_db", "data/harness.db"),
                cache_db=getattr(args, "cache_db", None),
                out=str(replay_out),
                limit=None,
                min_rows=getattr(args, "min_rows", 10),
                allow_no_execution=False,
            )
        )
        feature_result = _run_runtime_features_action(
            argparse.Namespace(
                action="runtime-features",
                replay=str(replay_out),
                out=str(features_out),
                test_ratio=getattr(args, "test_ratio", 0.2),
                min_rows=getattr(args, "min_rows", 10),
                min_actions=getattr(args, "min_actions", 4),
            )
        )
        result["runtime_data"] = data_result
        result["runtime_features"] = feature_result
        result["artifacts"] = {"replay": str(replay_out), "features": str(features_out)}
        result["ready_for_runtime_model_training"] = bool(feature_result.get("ready_for_runtime_model_training"))
    return result


def _run_runtime_probe_edge_action(args: argparse.Namespace) -> dict[str, Any]:
    remote_probe = _run_remote_runtime_probe_command(args)
    result: dict[str, Any] = {
        "status": "completed",
        "source": "edge",
        "remote_probe": remote_probe,
        "artifacts": {},
    }
    if bool(getattr(args, "export_dataset", True)):
        replay_out = Path(getattr(args, "replay_out", "data/model_ready/runtime_replay.jsonl"))
        features_out = Path(getattr(args, "features_out", "data/model_ready/runtime_features.jsonl"))
        data_result = _run_runtime_data_action(
            argparse.Namespace(
                action="runtime-data",
                source="edge",
                db=getattr(args, "db", "data/harness.db"),
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                remote_db=getattr(args, "remote_db", "data/harness.db"),
                cache_db=getattr(args, "cache_db", None),
                out=str(replay_out),
                limit=None,
                min_rows=getattr(args, "min_rows", 10),
                allow_no_execution=False,
            )
        )
        feature_result = _run_runtime_features_action(
            argparse.Namespace(
                action="runtime-features",
                replay=str(replay_out),
                out=str(features_out),
                test_ratio=getattr(args, "test_ratio", 0.2),
                min_rows=getattr(args, "min_rows", 10),
                min_actions=getattr(args, "min_actions", 4),
            )
        )
        result["runtime_data"] = data_result
        result["runtime_features"] = feature_result
        result["artifacts"] = {"replay": str(replay_out), "features": str(features_out)}
        result["ready_for_runtime_model_training"] = bool(feature_result.get("ready_for_runtime_model_training"))
    return result


def _run_runtime_seed_local_action(args: argparse.Namespace) -> dict[str, Any]:
    cycles = int(getattr(args, "cycles", 8))
    if cycles < 1:
        raise ModelReleaseError("runtime seed cycles must be >= 1")
    service = HarnessService(
        db_path=getattr(args, "db", "data/harness.db"),
        project_root=getattr(args, "project_root", "."),
        runtime_policy=RuntimeActionPolicy(RuntimePolicyConfig(model_path=None)),
    )
    specs = _runtime_seed_task_specs(
        profile=str(getattr(args, "profile", "readonly-basic")),
        target=str(getattr(args, "target", "orangepi5")),
        include_failures=bool(getattr(args, "include_failures", True)),
    )
    if not specs:
        raise ModelReleaseError("runtime seed profile produced no tasks")

    results = []
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    success_rows = 0
    failure_rows = 0
    for cycle in range(cycles):
        for template_index, spec in enumerate(specs):
            task_spec = {
                **spec,
                "context": {
                    **(spec.get("context") or {}),
                    "runtime_seed": {
                        "profile": getattr(args, "profile", "readonly-basic"),
                        "cycle": cycle,
                        "template_index": template_index,
                        "version": "runtime-seed-readonly-basic-v1",
                    },
                },
            }
            created = service.create_task(task_spec, created_by="runtime_seed", source="runtime_seed")
            task_id = created["task"]["task_id"]
            run_result = service.run(task_id)
            status = str(run_result.get("status") or "unknown")
            status_counts[status] += 1
            action = str(run_result.get("action") or "none")
            if action != "none":
                action_counts[action] += 1
            execution = run_result.get("execution_result") if isinstance(run_result.get("execution_result"), dict) else {}
            if execution.get("success") is True:
                success_rows += 1
            elif execution.get("success") is False:
                failure_rows += 1
            results.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "action": action,
                    "success": execution.get("success") if execution else None,
                    "error_type": execution.get("error_type") if execution else None,
                }
            )

    return {
        "profile": getattr(args, "profile", "readonly-basic"),
        "db": str(getattr(args, "db", "data/harness.db")),
        "project_root": str(getattr(args, "project_root", ".")),
        "target": str(getattr(args, "target", "orangepi5")),
        "cycles": cycles,
        "templates": len(specs),
        "tasks_created": len(results),
        "success_rows": success_rows,
        "failure_rows": failure_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "sample_results": results[:20],
    }


def _run_runtime_probe_local_action(args: argparse.Namespace) -> dict[str, Any]:
    cycles = int(getattr(args, "cycles", 8))
    if cycles < 1:
        raise ModelReleaseError("runtime probe cycles must be >= 1")
    service = HarnessService(
        db_path=getattr(args, "db", "data/harness.db"),
        project_root=getattr(args, "project_root", "."),
    )
    specs = _runtime_seed_task_specs(
        profile=str(getattr(args, "profile", "readonly-basic")),
        target=str(getattr(args, "target", "orangepi5")),
        include_failures=bool(getattr(args, "include_failures", True)),
    )
    if not specs:
        raise ModelReleaseError("runtime probe profile produced no tasks")

    results = []
    action_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    known_candidate_rows = 0
    candidate_groups = 0
    max_candidates = int(getattr(args, "max_candidates", 4))
    for cycle in range(cycles):
        for template_index, spec in enumerate(specs):
            task_spec = {
                **spec,
                "context": {
                    **(spec.get("context") or {}),
                    "runtime_probe": {
                        "profile": getattr(args, "profile", "readonly-basic"),
                        "cycle": cycle,
                        "template_index": template_index,
                        "version": "runtime-counterfactual-probe-v1",
                    },
                },
            }
            created = service.create_task(task_spec, created_by="runtime_probe", source="runtime_probe")
            task_id = created["task"]["task_id"]
            probe_result = service.probe_counterfactual_candidates(task_id, max_candidates=max_candidates)
            status = str(probe_result.get("status") or "unknown")
            status_counts[status] += 1
            known_count = int(probe_result.get("known_candidate_count") or 0)
            known_candidate_rows += known_count
            if known_count >= 2:
                candidate_groups += 1
            for action in probe_result.get("probed_actions") or []:
                action_counts[str(action)] += 1
            results.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "chosen_action": probe_result.get("chosen_action"),
                    "probed_actions": probe_result.get("probed_actions") or [],
                    "known_candidate_count": known_count,
                }
            )

    return {
        "profile": getattr(args, "profile", "readonly-basic"),
        "db": str(getattr(args, "db", "data/harness.db")),
        "project_root": str(getattr(args, "project_root", ".")),
        "target": str(getattr(args, "target", "orangepi5")),
        "cycles": cycles,
        "templates": len(specs),
        "tasks_created": len(results),
        "candidate_groups": candidate_groups,
        "known_candidate_rows": known_candidate_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "sample_results": results[:20],
    }


def _runtime_seed_task_specs(*, profile: str, target: str, include_failures: bool) -> list[dict[str, Any]]:
    if profile != "readonly-basic":
        raise ModelReleaseError(f"unknown runtime seed profile: {profile}")
    specs = [
        _runtime_seed_task("List project artifacts", target, ["list_artifacts", "get_disk_usage"], {"path": "."}, "artifact listing succeeds"),
        _runtime_seed_task("Check project disk usage", target, ["get_disk_usage", "list_artifacts"], {"path": "."}, "disk usage succeeds"),
        _runtime_seed_task("Check memory usage", target, ["get_memory_usage", "get_cpu_temp"], {}, "memory usage succeeds"),
        _runtime_seed_task("Check CPU temperature", target, ["get_cpu_temp", "get_cpu_per_core_usage"], {}, "temperature check completes"),
        _runtime_seed_task("Check per-core CPU usage", target, ["get_cpu_per_core_usage", "get_memory_usage"], {}, "per-core usage succeeds"),
        _runtime_seed_task("Read recent harness trace", target, ["get_recent_trace", "list_artifacts"], {"limit": 5}, "recent trace query succeeds"),
        _runtime_seed_task(
            "Diagnose seed health signals",
            target,
            ["diagnose_system_symptoms", "get_memory_usage"],
            {"symptoms": "runtime seed health check", "artifact_path": "artifacts", "trace_limit": 5},
            "diagnosis completes",
        ),
        _runtime_seed_task(
            "Inspect code structure for runtime seed",
            target,
            ["inspect_code_structure", "list_artifacts"],
            {"paths": ["src", "tests"], "max_files": 80, "max_results": 10},
            "code structure inspection completes",
        ),
        _runtime_seed_task("Inspect work pipeline state", target, ["inspect_work_pipeline", "get_recent_trace"], {"limit": 10, "stale_after_seconds": 300}, "pipeline inspection completes"),
        _runtime_seed_task("Check absent seed service status", target, ["get_service_status", "get_uptime"], {"service": "neurokernel-runtime-seed-missing.service"}, "service status query completes"),
    ]
    if include_failures:
        specs.extend(
            [
                _runtime_seed_task("Tail missing runtime seed log", target, ["tail_logs", "list_artifacts"], {"path": "missing-runtime-seed.log", "lines": 20}, "missing log failure is recorded"),
                _runtime_seed_task("Reject escaped runtime seed log path", target, ["tail_logs", "get_recent_trace"], {"path": "../outside-runtime-seed.log", "lines": 20}, "path escape failure is recorded"),
                _runtime_seed_task("Reject invalid runtime seed log line count", target, ["tail_logs", "get_recent_trace"], {"path": "README.md", "lines": 301}, "invalid line count failure is recorded"),
                _runtime_seed_task("Check missing disk path", target, ["get_disk_usage", "list_artifacts"], {"path": "missing-runtime-seed-dir"}, "missing disk path failure is recorded"),
            ]
        )
    return specs


def _runtime_seed_task(goal: str, target: str, actions: list[str], params: dict[str, Any], criterion: str) -> dict[str, Any]:
    return {
        "goal": goal,
        "target": target,
        "allowed_actions": actions,
        "context": {"params": params},
        "success_criteria": [criterion],
        "risk_level": "low",
        "requires_approval": False,
        "timeout_seconds": 30,
        "mode": "readonly",
    }


def _run_remote_runtime_seed_command(args: argparse.Namespace) -> dict[str, Any]:
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    timeout = int(getattr(args, "ssh_connect_timeout", 10))
    remote_db = getattr(args, "remote_db", None) or os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db")
    remote_python = getattr(args, "remote_python", None) or os.getenv("NEUROKERNEL_EDGE_PYTHON")
    python_cmd = _sh_quote(str(remote_python)) if remote_python else "$(test -x venv/bin/python && printf %s venv/bin/python || printf %s python3)"
    command_parts = [
        "cd",
        _sh_quote(remote_project),
        "&&",
        "PYTHONPATH=src",
        python_cmd,
        "-m",
        "neurokernel_seed.nk_cli",
        "runtime-seed",
        "--source",
        "local",
        "--db",
        _sh_quote(str(remote_db)),
        "--project-root",
        ".",
        "--profile",
        _sh_quote(str(getattr(args, "profile", "readonly-basic"))),
        "--target",
        "orangepi5",
        "--cycles",
        str(int(getattr(args, "cycles", 8))),
        "--no-export",
        "--json",
    ]
    if not bool(getattr(args, "include_failures", True)):
        command_parts.insert(-2, "--no-failures")
    remote_command = " ".join(command_parts)
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"]
    result = subprocess.run(
        ["ssh", *ssh_options, str(remote_host), remote_command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelReleaseError(f"runtime seed edge command failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelReleaseError(f"runtime seed edge command returned invalid json: {(result.stdout or '').strip()[:500]}") from exc


def _run_remote_runtime_probe_command(args: argparse.Namespace) -> dict[str, Any]:
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    timeout = int(getattr(args, "ssh_connect_timeout", 10))
    remote_db = getattr(args, "remote_db", None) or os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db")
    remote_python = getattr(args, "remote_python", None) or os.getenv("NEUROKERNEL_EDGE_PYTHON")
    python_cmd = _sh_quote(str(remote_python)) if remote_python else "$(test -x venv/bin/python && printf %s venv/bin/python || printf %s python3)"
    command_parts = [
        "cd",
        _sh_quote(remote_project),
        "&&",
        "PYTHONPATH=src",
        python_cmd,
        "-m",
        "neurokernel_seed.nk_cli",
        "runtime-probe",
        "--source",
        "local",
        "--db",
        _sh_quote(str(remote_db)),
        "--project-root",
        ".",
        "--profile",
        _sh_quote(str(getattr(args, "profile", "readonly-basic"))),
        "--target",
        "orangepi5",
        "--cycles",
        str(int(getattr(args, "cycles", 8))),
        "--max-candidates",
        str(int(getattr(args, "max_candidates", 4))),
        "--no-export",
        "--json",
    ]
    if not bool(getattr(args, "include_failures", True)):
        command_parts.insert(-2, "--no-failures")
    remote_command = " ".join(command_parts)
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"]
    result = subprocess.run(
        ["ssh", *ssh_options, str(remote_host), remote_command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ModelReleaseError(f"runtime probe edge command failed: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ModelReleaseError(f"runtime probe edge command returned invalid json: {(result.stdout or '').strip()[:500]}") from exc


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


def _run_data_audit_action(args: argparse.Namespace) -> dict[str, Any]:
    source = str(getattr(args, "source", None) or os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
    db = _resolve_runtime_data_db(args, source=source)
    if not db.exists():
        raise ModelReleaseError(f"harness db not found: {db}")
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        table_counts = _audit_table_counts(conn)
        linked_messages = _audit_count(conn, "conversation_messages", "linked_task_id IS NOT NULL")
        total_messages = table_counts.get("conversation_messages") or 0
        known_candidates = _audit_count(conn, "experience_candidates", "execution_result_known=1")
        total_candidates = table_counts.get("experience_candidates") or 0
        qualities = _audit_group_counts(conn, "interaction_outcomes", "answer_quality")
        task_sources = _audit_group_counts(conn, "tasks", "source")
        actions = _audit_group_counts(conn, "experience_candidates", "action_id")
    return {
        "status": "completed",
        "source": source,
        "db": str(db),
        "currently_accumulated": table_counts,
        "runtime_training_ready_sources": {
            "tasks": table_counts.get("tasks", 0),
            "action_decisions": table_counts.get("action_decisions", 0),
            "execution_results": table_counts.get("execution_results", 0),
            "experiences": table_counts.get("experiences", 0),
            "experience_candidates": total_candidates,
            "known_candidate_outcomes": known_candidates,
        },
        "new_contract_sources": {
            "conversation_messages": total_messages,
            "conversation_messages_linked_to_task": linked_messages,
            "interaction_outcomes": table_counts.get("interaction_outcomes", 0),
            "interaction_answer_quality": qualities,
        },
        "available_for_world_real_transitions": {
            "known_candidate_transitions": known_candidates,
            "interaction_aligned_transitions": table_counts.get("interaction_outcomes", 0),
        },
        "task_sources": task_sources,
        "candidate_actions": actions,
        "gaps": _audit_gaps(table_counts, total_messages, linked_messages, total_candidates, known_candidates),
    }


OPEN_TRAINING_WORK_STATUSES = {"proposed", "accepted", "planned", "reviewing", "blocked", "failed"}


def _run_training_pending_action(args: argparse.Namespace) -> dict[str, Any]:
    items = _fetch_training_work_items(args)
    return {
        "status": "completed",
        "source": "edge-core",
        "remote": _remote_training_target(args),
        "count": len(items),
        "items": [_summarize_training_work_item(item, index=index) for index, item in enumerate(items, start=1)],
    }


def _run_training_run_action(args: argparse.Namespace) -> dict[str, Any]:
    items = _fetch_training_work_items(args)
    selected = _select_training_work_item(items, work_id=getattr(args, "work_id", None), index=int(getattr(args, "index", 1) or 1))
    work_id = str(selected.get("work_id") or "")
    detail = _remote_core_request(args, "GET", f"/work-items/{work_id}")
    work = detail.get("work_item") if isinstance(detail.get("work_item"), dict) else selected
    command_spec = _training_command_spec(work)
    action, command_args = normalize_training_command(command_spec)
    if getattr(args, "device", None):
        command_args = _replace_cli_option(command_args, "--device", str(args.device))
    command_preview = [action, *command_args]

    report_dir = Path(os.getenv("NEUROKERNEL_MANUAL_TRAINING_JOB_DIR", "artifacts/training_jobs"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"manual_{_safe_file_stem(work_id)}_{_utc_stamp()}.json"

    if bool(getattr(args, "dry_run", False)):
        report = {
            "status": "dry_run",
            "work_id": work_id,
            "title": work.get("title"),
            "remote": _remote_training_target(args),
            "command": command_preview,
            "report": str(report_path),
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return report

    if str(work.get("status") or "") != "running":
        _transition_remote_work(args, work_id, "running", reason="manual laptop training started by nk")
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        parsed_args = _build_parser().parse_args(command_preview)
        parsed_args.action = _normalize_action(parsed_args.action)
        result = _run_action(parsed_args)
    except (ModelReleaseError, TrainingWorkerError, SystemExit) as exc:
        failure = {
            "status": "failed",
            "work_id": work_id,
            "title": work.get("title"),
            "remote": _remote_training_target(args),
            "command": command_preview,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
            "report": str(report_path),
        }
        report_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        _add_remote_training_note(args, work_id, f"노트북 수동 학습 실패: report={report_path} error={str(exc)[:500]}")
        _transition_remote_work(args, work_id, "reviewing", reason="manual laptop training failed; inspect local report")
        return failure

    report = {
        "status": "completed",
        "work_id": work_id,
        "title": work.get("title"),
        "remote": _remote_training_target(args),
        "command": command_preview,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "nk_result": result,
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    _add_remote_training_note(args, work_id, f"노트북 수동 학습 완료: report={report_path} status={result.get('status')}")
    _transition_remote_work(args, work_id, "completed", reason="manual laptop training completed")
    return report


def _fetch_training_work_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    payload = _remote_core_request(
        args,
        "GET",
        "/work-items",
        query={"work_type": "training_pipeline", "limit": max(1, int(getattr(args, "limit", 10) or 10))},
    )
    raw_items = payload.get("items") or payload.get("work_items") or []
    items = [item for item in raw_items if isinstance(item, dict)]
    allowed_statuses = set(getattr(args, "include_status", None) or OPEN_TRAINING_WORK_STATUSES)
    return [item for item in items if str(item.get("status") or "") in allowed_statuses]


def _select_training_work_item(items: list[dict[str, Any]], *, work_id: str | None, index: int) -> dict[str, Any]:
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


def _training_command_spec(work: dict[str, Any]) -> dict[str, Any]:
    metadata = work.get("metadata_json") if isinstance(work.get("metadata_json"), dict) else {}
    command_spec = metadata.get("nk_command") if isinstance(metadata.get("nk_command"), dict) else {}
    if not command_spec:
        raise ModelReleaseError(f"training work item has no nk_command: {work.get('work_id')}")
    return command_spec


def _summarize_training_work_item(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    try:
        command_spec = _training_command_spec(item) if isinstance(item.get("metadata_json"), dict) else {}
        action, command_args = normalize_training_command(command_spec)
        command = [action, *command_args]
    except (TrainingWorkerError, ModelReleaseError) as exc:
        command = [f"invalid: {exc}"]
    metadata = item.get("metadata_json") if isinstance(item.get("metadata_json"), dict) else {}
    slot = metadata.get("slot") or metadata.get("model_slot") or "unknown"
    return {
        "index": index,
        "work_id": item.get("work_id"),
        "status": item.get("status"),
        "slot": slot,
        "title": item.get("title"),
        "priority": item.get("priority"),
        "risk_level": item.get("risk_level"),
        "command": command,
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
    }


def _replace_cli_option(args: list[str], option: str, value: str) -> list[str]:
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


def _run_real_world_transitions_action(args: argparse.Namespace) -> dict[str, Any]:
    source = str(getattr(args, "source", None) or os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge"))
    db = _resolve_runtime_data_db(args, source=source)
    out = Path(getattr(args, "out", None) or os.getenv("NEUROKERNEL_REAL_WORLD_TRANSITIONS_OUT", "data/model_ready/real_world_transitions.jsonl"))
    result = export_real_world_transitions(db, out, limit=getattr(args, "limit", None))
    return {"status": "completed", "source": source, "db": str(db), **result}


def _audit_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "tasks",
        "action_decisions",
        "execution_results",
        "experiences",
        "experience_candidates",
        "conversation_messages",
        "interaction_outcomes",
        "task_references",
        "work_items",
        "work_jobs",
        "agent_events",
        "traces",
    )
    return {table: _audit_count(conn, table) for table in tables}


def _audit_count(conn: sqlite3.Connection, table: str, where: str | None = None) -> int:
    if not _audit_table_exists(conn, table):
        return 0
    sql = f"SELECT COUNT(*) FROM {table}" + (f" WHERE {where}" if where else "")
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def _audit_group_counts(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    if not _audit_table_exists(conn, table):
        return {}
    rows = conn.execute(f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} ORDER BY COUNT(*) DESC").fetchall()
    return {str(row[0] or "unknown"): int(row[1]) for row in rows}


def _audit_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _audit_gaps(table_counts: dict[str, int], total_messages: int, linked_messages: int, total_candidates: int, known_candidates: int) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if total_messages and linked_messages < total_messages:
        gaps.append({"kind": "conversation_task_link_missing", "actual": linked_messages, "total": total_messages, "why_it_matters": "디코 원문 요청을 runtime/world 학습 라벨과 안정적으로 묶기 위해 필요"})
    if table_counts.get("interaction_outcomes", 0) <= 0:
        gaps.append({"kind": "interaction_outcomes_missing", "actual": 0, "why_it_matters": "required/answered/missing/answer_quality 학습 원천이 아직 없음"})
    if total_candidates and known_candidates < total_candidates:
        gaps.append({"kind": "unknown_candidate_outcomes", "actual": known_candidates, "total": total_candidates, "why_it_matters": "runtime top1/ranking 학습은 후보별 outcome이 많을수록 좋아짐"})
    return gaps


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


def _run_runtime_benchmark_action(args: argparse.Namespace) -> dict[str, Any]:
    model = Path(getattr(args, "model", None) or os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt"))
    features = Path(getattr(args, "features", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl"))
    if not model.exists():
        raise ModelReleaseError(f"runtime 모델 없음: {model}")
    if not features.exists():
        raise ModelReleaseError(f"runtime features 없음: {features}")
    stamp = _utc_stamp()
    candidate_kind = str(getattr(args, "candidate_kind", None) or "local_candidate")
    record_dir = _runtime_benchmark_root(args) / f"{_safe_token(candidate_kind)}_{_safe_token(model.stem)}_{stamp}"
    record = _build_runtime_benchmark_record(model=model, features=features, args=args, candidate_kind=candidate_kind)
    return _write_runtime_benchmark_record(record, record_dir)


def _run_runtime_benchmark_current_action(args: argparse.Namespace) -> dict[str, Any]:
    remote_host = getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    remote_project = (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/")
    stamp = _utc_stamp()
    record_dir = _runtime_benchmark_root(args) / f"edge_current_{stamp}"
    record_dir.mkdir(parents=True, exist_ok=True)
    local_model = record_dir / "current_runtime_action_model.pt"
    local_manifest = record_dir / "current_runtime_action_model.manifest.json"
    _copy_remote_file(str(remote_host), f"{remote_project}/artifacts/current_runtime_action_model.pt", local_model)
    _copy_remote_file(str(remote_host), f"{remote_project}/artifacts/current_runtime_action_model.manifest.json", local_manifest)
    record = _build_runtime_benchmark_record(
        model=local_model,
        features=Path(getattr(args, "features", None) or os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl")),
        args=args,
        candidate_kind="edge_current",
        remote={"remote_host": remote_host, "remote_project": remote_project},
    )
    return _write_runtime_benchmark_record(record, record_dir)


def _run_runtime_compare_action(args: argparse.Namespace) -> dict[str, Any]:
    local_benchmark = _run_runtime_benchmark_action(
        argparse.Namespace(
            **{
                **vars(args),
                "action": "runtime-bench",
                "candidate_kind": "local_candidate",
            }
        )
    )
    current_benchmark = _run_runtime_benchmark_current_action(
        argparse.Namespace(
            **{
                **vars(args),
                "action": "runtime-bench-current",
                "candidate_kind": "edge_current",
            }
        )
    )
    decision = _compare_runtime_benchmarks(
        local_benchmark,
        current_benchmark,
        min_delta=float(getattr(args, "min_delta", 0.01)),
    )
    report = {
        "schema_version": MODEL_BENCHMARK_SCHEMA_VERSION,
        "slot": "runtime_action",
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "local": local_benchmark,
        "current": current_benchmark,
        "decision": decision,
        "needs_deploy": bool(decision.get("local_wins")),
    }
    report_path = _write_runtime_compare_report(report, _runtime_benchmark_root(args))
    return {**report, "report": str(report_path)}


def _build_runtime_benchmark_record(
    *,
    model: Path,
    features: Path,
    args: argparse.Namespace,
    candidate_kind: str,
    remote: dict[str, Any] | None = None,
) -> dict[str, Any]:
    feature_fingerprint = _runtime_feature_fingerprint(features)
    split = str(getattr(args, "split", None) or getattr(args, "benchmark_split", None) or "test")
    record: dict[str, Any] = {
        "schema_version": MODEL_BENCHMARK_SCHEMA_VERSION,
        "slot": "runtime_action",
        "candidate_kind": candidate_kind,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model),
        "manifest": str(model.with_suffix(".manifest.json")),
        "features": str(features),
        "feature_fingerprint": feature_fingerprint,
        "split": split,
        "remote": remote or {},
        "environment": _benchmark_environment(),
    }
    try:
        metrics = eval_runtime_action_checkpoint(model, features, split=split, device=str(getattr(args, "device", "auto")))
    except Exception as exc:  # noqa: BLE001 - benchmark records incompatibility instead of hiding it.
        record.update(
            {
                "status": "not_comparable",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "score": None,
                "quality": {"passed": False, "summary": "벤치 불가"},
            }
        )
        return record
    score = _runtime_benchmark_score(metrics)
    record.update(
        {
            "metrics": metrics,
            "score": score["score"],
            "score_components": score,
            "quality": _runtime_benchmark_quality(metrics, args),
        }
    )
    return record


def _runtime_benchmark_score(metrics: dict[str, Any]) -> dict[str, float]:
    success = _as_float(metrics.get("success_accuracy"))
    failure = _as_float(metrics.get("failure_present_accuracy"))
    reward_quality = 1.0 / (1.0 + max(0.0, _as_float(metrics.get("reward_mae"))))
    duration_quality = 1.0 / (1.0 + max(0.0, _as_float(metrics.get("duration_log1p_mae"))))
    ranking_groups = _as_float(metrics.get("ranking_evaluable_groups"))
    ranking_top1 = _as_float(metrics.get("top1_action_accuracy", metrics.get("ranking_top1_accuracy")))
    ranking_pairwise = _as_float(metrics.get("mean_pairwise_ranking_accuracy", metrics.get("ranking_pairwise_accuracy")))
    ranking_regret_quality = 1.0 / (1.0 + max(0.0, _as_float(metrics.get("mean_best_action_regret", metrics.get("ranking_mean_regret")))))
    ranking_quality = (0.50 * ranking_top1) + (0.30 * ranking_pairwise) + (0.20 * ranking_regret_quality) if ranking_groups > 0.0 else 0.0
    if ranking_groups > 0.0:
        score = (0.35 * success) + (0.20 * failure) + (0.20 * reward_quality) + (0.10 * duration_quality) + (0.15 * ranking_quality)
    else:
        score = (0.45 * success) + (0.20 * failure) + (0.25 * reward_quality) + (0.10 * duration_quality)
    return {
        "score": round(score, 6),
        "success_accuracy": success,
        "failure_present_accuracy": failure,
        "reward_quality": reward_quality,
        "duration_quality": duration_quality,
        "ranking_quality": ranking_quality,
        "ranking_evaluable_groups": ranking_groups,
        "top1_action_accuracy": ranking_top1,
        "mean_pairwise_ranking_accuracy": ranking_pairwise,
        "mean_best_action_regret_quality": ranking_regret_quality,
    }


def _runtime_benchmark_quality(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    success_accuracy = _as_float(metrics.get("success_accuracy"))
    reward_mae = _as_float(metrics.get("reward_mae"))
    known_success_rows = _as_float(metrics.get("known_success_rows"))
    ranking_evaluable_groups = _as_float(metrics.get("ranking_evaluable_groups"))
    top1_action_accuracy = _as_float(metrics.get("top1_action_accuracy", metrics.get("ranking_top1_accuracy")))
    min_success = float(getattr(args, "min_success_accuracy", 0.75))
    max_reward = float(getattr(args, "max_reward_mae", 0.35))
    min_known = int(getattr(args, "min_known_success_rows", 5))
    min_ranking_groups = int(getattr(args, "min_ranking_groups", 0))
    min_top1 = float(getattr(args, "min_top1_action_accuracy", 0.0))
    checks = {
        "success_accuracy": {"passed": success_accuracy >= min_success, "actual": success_accuracy, "threshold": min_success},
        "reward_mae": {"passed": reward_mae <= max_reward, "actual": reward_mae, "threshold": max_reward},
        "known_success_rows": {"passed": known_success_rows >= min_known, "actual": known_success_rows, "threshold": min_known},
        "ranking_evaluable_groups": {"passed": ranking_evaluable_groups >= min_ranking_groups, "actual": ranking_evaluable_groups, "threshold": min_ranking_groups},
        "top1_action_accuracy": {"passed": top1_action_accuracy >= min_top1, "actual": top1_action_accuracy, "threshold": min_top1},
    }
    passed = all(item["passed"] for item in checks.values())
    return {"passed": passed, "summary": "벤치 통과" if passed else "벤치 미통과", "checks": checks}


def _compare_runtime_benchmarks(local: dict[str, Any], current: dict[str, Any], *, min_delta: float) -> dict[str, Any]:
    local_passed = bool((local.get("quality") or {}).get("passed"))
    current_passed = bool((current.get("quality") or {}).get("passed"))
    local_score = local.get("score")
    current_score = current.get("score")
    if not local_passed:
        return {
            "local_wins": False,
            "reason": "local_benchmark_failed",
            "local_score": local_score,
            "current_score": current_score,
            "min_delta": min_delta,
        }
    if current.get("status") != "completed":
        return {
            "local_wins": True,
            "reason": "current_not_comparable",
            "local_score": local_score,
            "current_score": current_score,
            "min_delta": min_delta,
        }
    if not current_passed:
        return {
            "local_wins": True,
            "reason": "current_benchmark_failed",
            "local_score": local_score,
            "current_score": current_score,
            "min_delta": min_delta,
        }
    local_value = _as_float(local_score)
    current_value = _as_float(current_score)
    delta = local_value - current_value
    return {
        "local_wins": delta >= min_delta,
        "reason": "local_score_better" if delta >= min_delta else "current_score_not_worse",
        "local_score": local_value,
        "current_score": current_value,
        "delta": delta,
        "min_delta": min_delta,
    }


def _runtime_feature_fingerprint(features: Path) -> dict[str, Any]:
    manifest_path = features.with_suffix(features.suffix + ".manifest.json")
    payload = _read_json(manifest_path)
    action_vocab = payload.get("action_vocab") or []
    comparable = {
        "schema_version": payload.get("schema_version"),
        "input_dim": payload.get("input_dim"),
        "target_dim": payload.get("target_dim"),
        "target_names": payload.get("target_names"),
        "action_vocab": action_vocab,
        "numeric_feature_names": payload.get("numeric_feature_names"),
    }
    digest = hashlib.sha256(json.dumps(comparable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "manifest": str(manifest_path),
        "schema_version": payload.get("schema_version"),
        "rows": payload.get("rows"),
        "input_dim": payload.get("input_dim"),
        "target_dim": payload.get("target_dim"),
        "action_count": len(action_vocab),
        "sha256": digest,
    }


def _runtime_benchmark_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "out_dir", None) or os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")) / "runtime_action"


def _write_runtime_benchmark_record(record: dict[str, Any], record_dir: Path) -> dict[str, Any]:
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "benchmark.json"
    payload = {**record, "benchmark_path": str(record_path)}
    record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = record_dir.parent / f"latest_runtime_action_{_safe_token(str(record.get('candidate_kind') or 'candidate'))}.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_runtime_compare_report(report: dict[str, Any], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    report_path = root / f"runtime_action_compare_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (root / "latest_runtime_action_compare.json").write_text(
        json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _write_world_benchmark_record(
    *,
    candidate_kind: str,
    model_path: Path,
    result: dict[str, Any],
    source_dir: Path,
    release_name: str,
    remote: dict[str, Any] | None = None,
) -> Path:
    aggregate = result.get("aggregate") or {}
    macro = aggregate.get("macro_success_rate") or {}
    score = _as_float(macro.get("hybrid_veto"))
    record = {
        "schema_version": MODEL_BENCHMARK_SCHEMA_VERSION,
        "slot": "world",
        "candidate_kind": candidate_kind,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_name": release_name,
        "model": str(model_path),
        "source_dir": str(source_dir),
        "remote": remote or {},
        "environment": _benchmark_environment(),
        "score": score,
        "score_components": {
            "hybrid_veto_success": score,
            "hybrid_success": _as_float(macro.get("hybrid")),
            "prior_success": _as_float(macro.get("prior_only")),
            "model_only_success": _as_float(macro.get("model_only")),
            "hybrid_veto_gain_over_prior": _as_float(aggregate.get("hybrid_veto_gain_over_prior")),
            "model_needed_signal_group_count": int(aggregate.get("model_needed_signal_group_count") or 0),
        },
        "benchmark": result,
    }
    root = Path(os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")) / "world"
    stamp = _utc_stamp()
    record_dir = root / f"{_safe_token(candidate_kind)}_{_safe_token(release_name)}_{stamp}"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "benchmark.json"
    payload = {**record, "benchmark_path": str(record_path)}
    record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    latest_path = root / f"latest_world_{_safe_token(candidate_kind)}.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return record_path


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
    return _deploy_world_best_from_comparison(args, comparison)


def _deploy_world_best_from_comparison(args: argparse.Namespace, comparison: dict[str, Any]) -> dict[str, Any]:
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


def _run_current_benchmark_all_action(args: argparse.Namespace) -> dict[str, Any]:
    stamp = _utc_stamp()
    out_root = Path(getattr(args, "out_dir", None) or os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    report_root = out_root / "current_models"
    report_root.mkdir(parents=True, exist_ok=True)
    world = _run_guarded_slot(
        "world",
        lambda: _run_benchmark_current_action(
            argparse.Namespace(
                action="bench-current",
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
                out_dir=str(report_root / f"world_current_{stamp}"),
                episodes=getattr(args, "episodes", 50),
                trace_episodes=getattr(args, "trace_episodes", 10),
                max_failures_per_env=getattr(args, "max_failures_per_env", 10),
                strict=getattr(args, "strict", True),
            )
        ),
    )
    runtime = _run_guarded_slot(
        "runtime_action",
        lambda: _run_runtime_benchmark_current_action(
            argparse.Namespace(
                action="runtime-bench-current",
                remote_host=getattr(args, "remote_host", None),
                remote_project=getattr(args, "remote_project", None),
                ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
                features=getattr(args, "features", None),
                split=getattr(args, "split", "test"),
                device=getattr(args, "device", "auto"),
                out_dir=str(out_root),
                min_success_accuracy=getattr(args, "min_success_accuracy", 0.75),
                max_reward_mae=getattr(args, "max_reward_mae", 0.35),
                min_known_success_rows=getattr(args, "min_known_success_rows", 5),
            )
        ),
    )
    report = {
        "schema_version": MODEL_BENCHMARK_SCHEMA_VERSION,
        "action": "current_bench_all",
        "status": "completed" if world.get("status") == "completed" and runtime.get("status") in {"completed", "not_comparable"} else "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": _benchmark_environment(),
        "benchmark_policy": {
            "world": "OrangePi current_world_model.onnx를 로컬로 복사한 뒤 gate ablation을 실행한다.",
            "runtime_action": "OrangePi current_runtime_action_model.pt를 로컬로 복사한 뒤 현재 runtime_features split으로 평가한다.",
        },
        "steps": {"world": world, "runtime_action": runtime},
    }
    report_path = report_root / f"current_model_benchmarks_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (report_root / "latest_current_model_benchmarks.json").write_text(
        json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {**report, "report": str(report_path)}


def _run_deploy_all_best_action(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(getattr(args, "out_dir", None) or os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks"))
    world_current_out = out_root / "current_models" / f"world_current_deploy_{_utc_stamp()}"
    world_compare = _run_guarded_slot(
        "world_compare",
        lambda: _run_compare(
            argparse.Namespace(
                **{
                    **vars(args),
                    "action": "compare",
                    "out_dir": str(world_current_out),
                    "min_delta": getattr(args, "world_min_delta", 0.005),
                    "refresh_current_bench": getattr(args, "refresh_current_bench", True),
                }
            )
        ),
    )
    runtime_compare = _run_guarded_slot(
        "runtime_compare",
        lambda: _run_runtime_compare_action(
            argparse.Namespace(
                **{
                    **vars(args),
                    "action": "runtime-compare",
                    "min_delta": getattr(args, "runtime_min_delta", 0.01),
                }
            )
        ),
    )
    world_deploy = None
    runtime_deploy = None
    if world_compare.get("status") == "ok" and world_compare.get("needs_deploy"):
        world_deploy = _run_guarded_slot("world_deploy", lambda: _deploy_world_best_from_comparison(args, world_compare))
    if runtime_compare.get("status") == "completed" and runtime_compare.get("needs_deploy"):
        runtime_deploy = _run_guarded_slot(
            "runtime_deploy",
            lambda: _run_deploy_runtime_action(
                argparse.Namespace(
                    action="deploy-runtime",
                    model=getattr(args, "model", None),
                    remote_host=getattr(args, "remote_host", None),
                    remote_project=getattr(args, "remote_project", None),
                    ssh_connect_timeout=getattr(args, "ssh_connect_timeout", 10),
                )
            ),
        )
    deployed_slots = []
    if isinstance(world_deploy, dict) and world_deploy.get("status") == "deployed_and_activated":
        deployed_slots.append("world")
    if isinstance(runtime_deploy, dict) and runtime_deploy.get("status") == "deployed_and_activated":
        deployed_slots.append("runtime_action")
    report = {
        "schema_version": MODEL_BENCHMARK_SCHEMA_VERSION,
        "action": "deploy_all_best",
        "status": "deployed" if deployed_slots else "nothing_deployed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": _benchmark_environment(),
        "deployed_slots": deployed_slots,
        "steps": {
            "world_compare": world_compare,
            "runtime_compare": runtime_compare,
            "world_deploy": world_deploy,
            "runtime_deploy": runtime_deploy,
        },
    }
    report_path = _write_integrated_deploy_report(report, out_root)
    return {**report, "report": str(report_path)}


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
    benchmark_record = _write_world_benchmark_record(
        candidate_kind="local_candidate",
        model_path=model_path,
        result=result,
        source_dir=run_dir,
        release_name=getattr(args, "run_name", None) or model_path.stem,
    )
    return {**result, "benchmark_record": str(benchmark_record)}


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
    benchmark_record = _write_world_benchmark_record(
        candidate_kind="edge_current",
        model_path=benchmark_dir / "world_model.onnx",
        result=result,
        source_dir=benchmark_dir,
        release_name=release_name,
        remote={"remote_host": remote_host, "remote_project": remote_project},
    )
    manifest = {
        "status": "completed",
        "source": "orangepi_current_model",
        "release_name": release_name,
        "remote_host": remote_host,
        "remote_project": remote_project,
        "run_dir": str(benchmark_dir),
        "gate_ablation_result": str(result_path),
        "benchmark_record": str(benchmark_record),
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


def _remote_training_target(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "remote_host": getattr(args, "remote_host", None) or os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5"),
        "remote_project": (getattr(args, "remote_project", None) or os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")).rstrip("/"),
        "remote_core_url": str(getattr(args, "remote_core_url", None) or os.getenv("NEUROKERNEL_EDGE_CORE_URL", "http://127.0.0.1:8765")).rstrip("/"),
        "ssh_connect_timeout": int(getattr(args, "ssh_connect_timeout", 10)),
    }


def _remote_core_request(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = _remote_training_target(args)
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
    remote_command = " ".join(command_parts)
    ssh_options = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(target['ssh_connect_timeout'])}"]
    completed = subprocess.run(
        ["ssh", *ssh_options, str(target["remote_host"]), remote_command],
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


def _transition_remote_work(args: argparse.Namespace, work_id: str, status: str, *, reason: str) -> dict[str, Any]:
    return _remote_core_request(
        args,
        "POST",
        f"/work-items/{work_id}/status",
        payload={"status": status, "actor": "nk-manual-training", "reason": reason},
    )


def _add_remote_training_note(args: argparse.Namespace, work_id: str, note: str) -> dict[str, Any]:
    return _remote_core_request(
        args,
        "POST",
        f"/work-items/{work_id}/note",
        payload={"actor": "nk-manual-training", "note": note},
    )


def _safe_file_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(value)).strip("._")
    return safe or "training_work"


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
    current_benchmark = None
    if bool(getattr(args, "refresh_current_bench", False)):
        current_benchmark = _run_benchmark_current_action(args)
    else:
        current_benchmark = _latest_world_current_benchmark(current_name)
    decision = _compare_world_benchmark(best, current_benchmark, min_delta=float(getattr(args, "min_delta", 0.005)))
    if decision.get("comparable"):
        needs_deploy = bool(best_name and current_name != best_name and decision.get("local_wins"))
    else:
        needs_deploy = bool(best_name and current_name != best_name)
    return {
        "status": "ok",
        "current": current,
        "current_name": current_name,
        "current_benchmark": current_benchmark,
        "top": top,
        "best": best,
        "best_name": best_name,
        "benchmark_decision": decision,
        "needs_deploy": needs_deploy,
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
    run_root = Path(os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")) / "current_models"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return run_root / f"world_current_{_safe_token(release_name)}_{stamp}"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_token(value: str) -> str:
    token = "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value))
    return token.strip("._-") or "unknown"


def _run_guarded_slot(slot: str, action) -> dict[str, Any]:
    try:
        result = action()
    except Exception as exc:  # noqa: BLE001 - slot reports must preserve the exact failure.
        return {"slot": slot, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    if isinstance(result, dict):
        return {"slot": slot, **result}
    return {"slot": slot, "status": "completed", "result": result}


def _benchmark_environment() -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "git_commit": _git_rev_parse("HEAD"),
        "git_branch": _git_rev_parse("--abbrev-ref", "HEAD"),
    }


def _git_rev_parse(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _write_integrated_deploy_report(report: dict[str, Any], root: Path) -> Path:
    report_root = root / "deploy"
    report_root.mkdir(parents=True, exist_ok=True)
    stamp = _utc_stamp()
    report_path = report_root / f"deploy_all_best_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (report_root / "latest_deploy_all_best.json").write_text(
        json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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
    if _is_current_world_benchmark_dir(path.parent):
        return None
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


def _is_current_world_benchmark_dir(run_dir: Path) -> bool:
    manifest_path = run_dir / "current_core_benchmark_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = _read_json(manifest_path)
    return manifest is None or manifest.get("source") == "orangepi_current_model"


def _top_sort_key(item: dict[str, Any]) -> tuple[float, int, float]:
    return (item["hybrid_veto_success"], item["model_needed_signal_group_count"], item["hybrid_veto_gain_over_prior"])


def _latest_world_current_benchmark(current_name: str) -> dict[str, Any] | None:
    root = Path(os.getenv("NEUROKERNEL_MODEL_BENCHMARK_DIR", "artifacts/model_benchmarks")) / "world"
    latest = _read_json(root / "latest_world_edge_current.json")
    if latest:
        release_name = str(latest.get("release_name") or "")
        if not current_name or release_name == current_name or current_name == "모름":
            return latest
    return None


def _compare_world_benchmark(best: dict[str, Any] | None, current_benchmark: dict[str, Any] | None, *, min_delta: float) -> dict[str, Any]:
    if not best:
        return {"comparable": False, "local_wins": False, "reason": "no_local_world_candidate", "min_delta": min_delta}
    best_score = _as_float(best.get("hybrid_veto_success"))
    if not current_benchmark:
        return {
            "comparable": False,
            "local_wins": True,
            "reason": "current_world_benchmark_missing",
            "best_score": best_score,
            "current_score": None,
            "min_delta": min_delta,
        }
    current_score = _as_float(current_benchmark.get("score"))
    delta = best_score - current_score
    return {
        "comparable": True,
        "local_wins": delta >= min_delta,
        "reason": "local_score_better" if delta >= min_delta else "current_score_not_worse",
        "best_score": best_score,
        "current_score": current_score,
        "delta": delta,
        "min_delta": min_delta,
    }


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
    if action == "runtime-pipeline":
        _print_runtime_pipeline(result)
    elif action in {"runtime-auto", "runtime-cycle"}:
        _print_runtime_auto(result)
    elif action == "runtime-seed":
        _print_runtime_seed(result)
    elif action == "runtime-probe":
        _print_runtime_probe(result)
    elif action == "data":
        _print_data(result)
    elif action == "runtime-data":
        _print_runtime_data(result)
    elif action == "runtime-features":
        _print_runtime_features(result)
    elif action == "data-audit":
        _print_data_audit(result)
    elif action == "training-pending":
        _print_training_pending(result)
    elif action == "training-run":
        _print_training_run(result)
    elif action == "real-world-transitions":
        _print_real_world_transitions(result)
    elif action == "runtime-train":
        _print_runtime_train(result)
    elif action == "runtime-bench":
        _print_runtime_bench(result)
    elif action == "runtime-bench-current":
        _print_runtime_bench(result)
    elif action == "runtime-compare":
        _print_runtime_compare(result)
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
    elif action == "deploy-all-best":
        _print_deploy_all_best(result)
    elif action == "bench":
        _print_bench(result)
    elif action == "bench-current":
        _print_bench_current(result)
    elif action == "current-bench-all":
        _print_current_bench_all(result)
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


def _print_training_pending(result: dict[str, Any]) -> None:
    print(_ok("학습대기", result.get("status", "completed")))
    remote = result.get("remote") or {}
    print(_kv("원격", f"{remote.get('remote_host')} {remote.get('remote_core_url')}"))
    print(_kv("후보", result.get("count")))
    for item in result.get("items") or []:
        command = " ".join(str(part) for part in (item.get("command") or []))
        print(f"{item.get('index')}. {item.get('status')} {item.get('slot')} {item.get('work_id')}")
        print(f"   {item.get('title')}")
        print(f"   nk {command}")


def _print_training_run(result: dict[str, Any]) -> None:
    status = str(result.get("status") or "unknown")
    print(_ok("학습실행", status) if status in {"completed", "dry_run"} else _warn("학습실행", status))
    print(_kv("work", result.get("work_id")))
    print(_kv("제목", result.get("title")))
    print(_kv("명령", " ".join(str(part) for part in (result.get("command") or []))))
    print(_kv("리포트", result.get("report")))
    nk_result = result.get("nk_result") if isinstance(result.get("nk_result"), dict) else {}
    if nk_result:
        print(_kv("NK결과", nk_result.get("status")))
        artifacts = nk_result.get("artifacts") if isinstance(nk_result.get("artifacts"), dict) else {}
        if artifacts.get("model"):
            print(_kv("모델", artifacts.get("model")))
    if result.get("error"):
        print(_kv("오류", result.get("error")))


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


def _print_runtime_pipeline(result: dict[str, Any]) -> None:
    status = str(result.get("status") or "unknown")
    label = {
        "deployed": "배포 완료",
        "trained": "학습 완료",
        "blocked_by_quality_gate": "품질 보류",
        "blocked_by_benchmark_compare": "벤치 보류",
    }.get(status, status)
    print(_ok("오토파일럿", label) if status not in {"blocked_by_quality_gate", "blocked_by_benchmark_compare"} else _warn("오토파일럿", label))
    steps = result.get("steps") or {}
    data = steps.get("runtime_data") or {}
    features = steps.get("runtime_features") or {}
    train = steps.get("runtime_train") or {}
    quality = result.get("quality") or {}
    artifacts = result.get("artifacts") or {}
    print(_kv("데이터", f"{(data.get('validation') or {}).get('rows')} rows / {data.get('source')}"))
    print(_kv("특징", f"{(features.get('validation') or {}).get('rows')} rows"))
    print(_kv("모델", artifacts.get("model")))
    test = train.get("test") or {}
    if test:
        print(_kv("성공률", f"{float(test.get('success_accuracy', 0.0)):.3f}"))
        print(_kv("보상오차", f"{float(test.get('reward_mae', 0.0)):.3f}"))
        print(_kv("known", int(float(test.get("known_success_rows", 0.0)))))
    print(_kv("품질", quality.get("summary")))
    for name, item in (quality.get("checks") or {}).items():
        mark = "통과" if item.get("passed") else "미통과"
        print(_kv(f" - {name}", f"{mark} actual={item.get('actual')} threshold={item.get('threshold')}"))
    compare = steps.get("runtime_compare") or {}
    if compare:
        decision = compare.get("decision") or {}
        print(_kv("벤치", f"local={decision.get('local_score')} current={decision.get('current_score')} reason={decision.get('reason')}"))
    deploy = steps.get("runtime_deploy") or {}
    if deploy:
        print(_kv("배포", deploy.get("remote_model")))
    elif status == "blocked_by_quality_gate":
        print(_kv("배포", "보류됨. --force-deploy를 쓰면 강제 배포 가능"))
    elif status == "blocked_by_benchmark_compare":
        print(_kv("배포", "보류됨. 현행 모델이 벤치에서 밀리지 않음"))
    print(_kv("리포트", result.get("report")))


def _print_runtime_seed(result: dict[str, Any]) -> None:
    print(_ok("런타임 시드", result.get("status", "completed")))
    seed = result.get("seed") or result.get("remote_seed", {}).get("seed") or {}
    print(_kv("출처", result.get("source")))
    print(_kv("작업", seed.get("tasks_created")))
    print(_kv("성공", seed.get("success_rows")))
    print(_kv("실패", seed.get("failure_rows")))
    action_counts = seed.get("action_counts") or {}
    if action_counts:
        print(_kv("액션", ", ".join(f"{key}:{value}" for key, value in action_counts.items())))
    data = result.get("runtime_data") or {}
    features = result.get("runtime_features") or {}
    if data:
        print(_kv("replay행", (data.get("validation") or {}).get("rows")))
    if features:
        print(_kv("특징행", (features.get("validation") or {}).get("rows")))
        print(_kv("학습준비", result.get("ready_for_runtime_model_training")))
    artifacts = result.get("artifacts") or {}
    if artifacts:
        print(_kv("replay", artifacts.get("replay")))
        print(_kv("특징", artifacts.get("features")))


def _print_runtime_probe(result: dict[str, Any]) -> None:
    print(_ok("런타임프로브", result.get("status", "completed")))
    probe = result.get("probe") or result.get("remote_probe", {}).get("probe") or {}
    print(_kv("출처", result.get("source")))
    print(_kv("작업", probe.get("tasks_created")))
    print(_kv("후보그룹", probe.get("candidate_groups")))
    print(_kv("known후보", probe.get("known_candidate_rows")))
    action_counts = probe.get("action_counts") or {}
    if action_counts:
        print(_kv("액션", ", ".join(f"{key}:{value}" for key, value in action_counts.items())))
    data = result.get("runtime_data") or {}
    features = result.get("runtime_features") or {}
    if data:
        print(_kv("replay행", (data.get("validation") or {}).get("rows")))
    if features:
        print(_kv("feature행", (features.get("validation") or {}).get("rows")))
        print(_kv("학습준비", result.get("ready_for_runtime_model_training")))
    artifacts = result.get("artifacts") or {}
    if artifacts:
        print(_kv("replay", artifacts.get("replay")))
        print(_kv("features", artifacts.get("features")))


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


def _print_data_audit(result: dict[str, Any]) -> None:
    print(_ok("데이터감사", result.get("status", "completed")))
    print(_kv("출처", result.get("source")))
    print(_kv("DB", result.get("db")))
    accumulated = result.get("currently_accumulated") or {}
    runtime = result.get("runtime_training_ready_sources") or {}
    contract = result.get("new_contract_sources") or {}
    world = result.get("available_for_world_real_transitions") or {}
    print(_kv("tasks", accumulated.get("tasks")))
    print(_kv("decisions", accumulated.get("action_decisions")))
    print(_kv("executions", accumulated.get("execution_results")))
    print(_kv("candidates", f"{runtime.get('known_candidate_outcomes')}/{runtime.get('experience_candidates')} known"))
    print(_kv("messages", f"{contract.get('conversation_messages_linked_to_task')}/{contract.get('conversation_messages')} linked"))
    print(_kv("interaction", contract.get("interaction_outcomes")))
    print(_kv("world 후보", world.get("known_candidate_transitions")))
    for gap in (result.get("gaps") or [])[:5]:
        print(_kv("gap", f"{gap.get('kind')} actual={gap.get('actual')} total={gap.get('total')}"))


def _print_real_world_transitions(result: dict[str, Any]) -> None:
    print(_ok("실월드전이", result.get("status", "completed")))
    print(_kv("출처", result.get("source")))
    print(_kv("DB", result.get("db")))
    print(_kv("파일", result.get("out")))
    print(_kv("rows", result.get("rows")))
    print(_kv("actions", ", ".join(result.get("actions") or [])))


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


def _print_runtime_bench(result: dict[str, Any]) -> None:
    passed = bool((result.get("quality") or {}).get("passed"))
    label = "통과" if passed else result.get("status", "미통과")
    print(_ok("런타임벤치", label) if passed else _warn("런타임벤치", label))
    print(_kv("후보", result.get("candidate_kind")))
    print(_kv("점수", result.get("score")))
    metrics = result.get("metrics") or {}
    if metrics:
        print(_kv("성공률", f"{_as_float(metrics.get('success_accuracy')):.3f}"))
        print(_kv("보상오차", f"{_as_float(metrics.get('reward_mae')):.3f}"))
    if result.get("error"):
        print(_kv("이유", result.get("error")))
    print(_kv("기록", result.get("benchmark_path")))


def _print_runtime_compare(result: dict[str, Any]) -> None:
    decision = result.get("decision") or {}
    print(_style("런타임비교", "cyan"))
    print(_kv("로컬", decision.get("local_score")))
    print(_kv("현재", decision.get("current_score")))
    if "delta" in decision:
        print(_kv("차이", f"{_as_float(decision.get('delta')):.6f}"))
    print(_kv("판단", decision.get("reason")))
    print(_kv("배포", "필요" if result.get("needs_deploy") else "보류"))
    print(_kv("기록", result.get("report")))


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


def _print_deploy_all_best(result: dict[str, Any]) -> None:
    slots = result.get("deployed_slots") or []
    print(_ok("통합배포", ", ".join(slots) if slots else "배포 없음"))
    steps = result.get("steps") or {}
    world_compare = steps.get("world_compare") or {}
    runtime_compare = steps.get("runtime_compare") or {}
    print(_kv("world", "배포 필요" if world_compare.get("needs_deploy") else "보류"))
    print(_kv("runtime", "배포 필요" if runtime_compare.get("needs_deploy") else "보류"))
    world_deploy = steps.get("world_deploy") or {}
    runtime_deploy = steps.get("runtime_deploy") or {}
    if world_deploy:
        print(_kv("world배포", world_deploy.get("status")))
    if runtime_deploy:
        print(_kv("runtime배포", runtime_deploy.get("status")))
    print(_kv("리포트", result.get("report")))


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


def _print_current_bench_all(result: dict[str, Any]) -> None:
    print(_ok("현행벤치", result.get("status", "completed")))
    steps = result.get("steps") or {}
    world = steps.get("world") or {}
    runtime = steps.get("runtime_action") or {}
    print(_kv("world", world.get("status")))
    print(_kv("runtime", runtime.get("status")))
    if runtime.get("error"):
        print(_kv("runtime이유", runtime.get("error")))
    print(_kv("리포트", result.get("report")))


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
