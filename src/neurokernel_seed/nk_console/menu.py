from __future__ import annotations

import argparse
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MenuCommand:
    key: str
    action: str
    label: str
    hint: str


ACTION_ALIASES = {
    "autopilot": "runtime-pipeline",
    "오토파일럿": "runtime-pipeline",
    "오토": "runtime-pipeline",
    "풀자동": "runtime-pipeline",
    "전체자동": "runtime-pipeline",
    "auto": "runtime-auto",
    "learn": "runtime-auto",
    "학습": "runtime-auto",
    "자동학습": "runtime-auto",
    "world-train": "train",
    "월드학습": "train",
    "runtime-pipeline": "runtime-pipeline",
    "train-runtime": "runtime-train",
    "학습만": "runtime-train",
    "런타임학습": "runtime-pipeline",
    "런타임학습만": "runtime-train",
    "runtime-bench": "runtime-bench",
    "런타임벤치": "runtime-bench",
    "현재런타임벤치": "runtime-bench-current",
    "runtime-compare": "runtime-compare",
    "런타임비교": "runtime-compare",
    "seed": "runtime-seed",
    "시드": "runtime-seed",
    "초기데이터": "runtime-seed",
    "runtime-seed": "runtime-seed",
    "seed-runtime": "runtime-seed",
    "auto-deploy": "runtime-cycle",
    "learn-deploy": "runtime-cycle",
    "cycle": "runtime-cycle",
    "배포학습": "runtime-cycle",
    "학습배포": "runtime-cycle",
    "순환": "runtime-cycle",
    "runtime-cycle": "runtime-cycle",
    "dataset": "data",
    "features": "data",
    "데이터": "data",
    "runtime-data": "runtime-data",
    "runtime-features": "runtime-features",
    "runtime-train": "runtime-train",
    "runtime-deploy": "deploy-runtime",
    "runtime-use": "deploy-runtime",
    "런타임배포": "deploy-runtime",
    "배포만": "deploy-runtime",
    "edge": "current",
    "models": "list",
    "rank": "top",
    "ship": "deploy-best",
    "compare": "compare",
    "비교": "compare",
    "월드비교": "compare",
    "모델비교": "runtime-compare",
    "deploy": "deploy-best",
    "배포": "deploy-best",
    "월드배포": "deploy-best",
    "통합배포": "deploy-all-best",
    "전체배포": "deploy-all-best",
    "현행벤치": "current-bench-all",
    "현재모델벤치": "current-bench-all",
    "상태": "status",
    "switch": "use",
    "q": "exit",
    "quit": "exit",
    "exit": "exit",
    "종료": "exit",
    "나가기": "exit",
}


DASHBOARD_COMMANDS = (
    MenuCommand("1", "train", "월드학습", "데이터 확인, world 학습, gate 벤치까지 실행"),
    MenuCommand("2", "runtime-pipeline", "런타임학습", "OrangePi 수집, 전처리, CUDA 학습, 벤치 비교 후 승리 시 배포"),
    MenuCommand("3", "runtime-compare", "런타임비교", "현행 OrangePi runtime과 로컬 후보를 같은 데이터로 벤치 비교"),
    MenuCommand("4", "deploy-all-best", "통합배포", "world/runtime 각각 비교해서 이긴 슬롯만 OrangePi에 배포"),
    MenuCommand("5", "status", "상태", "world/runtime 슬롯과 후보 모델 확인"),
    MenuCommand("6", "compare", "월드비교", "현재 world 모델과 최고 후보 벤치 비교"),
    MenuCommand("9", "current-bench-all", "현행벤치", "OrangePi 현행 world/runtime 모델을 캐시하고 벤치 결과 저장"),
    MenuCommand("0", "exit", "종료", "콘솔 닫기"),
)


KNOWN_ACTIONS = {
    "data",
    "runtime-pipeline",
    "runtime-seed",
    "runtime-auto",
    "runtime-cycle",
    "runtime-data",
    "runtime-features",
    "runtime-train",
    "runtime-bench",
    "runtime-bench-current",
    "runtime-compare",
    "deploy-runtime",
    "check",
    "train",
    "deploy",
    "deploy-use",
    "deploy-best",
    "deploy-all-best",
    "bench",
    "bench-current",
    "current-bench-all",
    "compare",
    "top",
    "list",
    "current",
    "use",
    "status",
    "advanced",
    "exit",
}


def normalize_action(action: str) -> str:
    return ACTION_ALIASES.get(action, action)


def dashboard_action(raw: str) -> str | None:
    value = raw.split()[0].lower()
    for command in DASHBOARD_COMMANDS:
        if value in {command.key, command.action, command.label.lower()}:
            return normalize_action(command.action)
    normalized = normalize_action(value)
    return normalized if normalized in KNOWN_ACTIONS else None


def build_menu_args(base: argparse.Namespace, action: str) -> argparse.Namespace:
    menu_args = argparse.Namespace(**vars(base))
    menu_args.action = normalize_action(action)
    runtime_replay = os.getenv("NEUROKERNEL_RUNTIME_REPLAY_OUT", "data/model_ready/runtime_replay.jsonl")
    runtime_features = os.getenv("NEUROKERNEL_RUNTIME_FEATURES_OUT", "data/model_ready/runtime_features.jsonl")
    runtime_model = os.getenv("NEUROKERNEL_RUNTIME_ACTION_MODEL_OUT", "artifacts/runtime_action_model.pt")
    menu_args.run_name = None
    menu_args.model = None
    menu_args.out_dir = None
    menu_args.run_dir = None
    menu_args.limit = 5 if menu_args.action in {"status", "top", "compare", "deploy-best", "deploy-all-best"} else None
    menu_args.episodes = 50
    menu_args.trace_episodes = 10
    menu_args.max_failures_per_env = 10
    menu_args.strict = True
    menu_args.db = os.getenv("NEUROKERNEL_HARNESS_DB", "data/harness.db")
    menu_args.source = os.getenv("NEUROKERNEL_RUNTIME_DATA_SOURCE", "edge")
    menu_args.remote_host = os.getenv("NEUROKERNEL_EDGE_HOST", "orangepi5")
    menu_args.remote_project = os.getenv("NEUROKERNEL_EDGE_PROJECT", "/home/ubuntu/projects/neurokernel-agi-seed")
    menu_args.remote_db = os.getenv("NEUROKERNEL_EDGE_HARNESS_DB", "data/harness.db")
    menu_args.cache_db = os.getenv("NEUROKERNEL_RUNTIME_DB_CACHE")
    menu_args.ssh_connect_timeout = 10
    menu_args.replay_out = runtime_replay
    menu_args.features_out = runtime_features
    menu_args.model_out = runtime_model
    menu_args.replay = runtime_replay
    menu_args.features = runtime_features
    if menu_args.action == "runtime-data":
        menu_args.out = runtime_replay
    elif menu_args.action == "runtime-features":
        menu_args.out = runtime_features
    elif menu_args.action == "runtime-train":
        menu_args.out = runtime_model
    else:
        menu_args.out = None
    if menu_args.action in {"deploy-runtime", "runtime-bench", "runtime-compare", "deploy-all-best"}:
        menu_args.model = runtime_model
    menu_args.test_ratio = 0.2
    menu_args.project_root = "."
    menu_args.profile = "readonly-basic"
    menu_args.target = "orangepi5"
    menu_args.cycles = 8
    menu_args.seed_cycles = 0
    menu_args.epochs = 100 if menu_args.action in {"runtime-pipeline", "runtime-train"} else 50
    menu_args.batch_size = 128
    menu_args.device = "cuda" if menu_args.action in {"runtime-pipeline", "runtime-train"} else "auto"
    menu_args.patience = 20
    menu_args.min_rows = 1
    menu_args.min_actions = 4 if menu_args.action in {"runtime-seed", "runtime-pipeline"} else 1
    menu_args.min_success_accuracy = 0.75
    menu_args.max_reward_mae = 0.35
    menu_args.min_known_success_rows = 5
    menu_args.split = "test"
    menu_args.benchmark_split = "test"
    menu_args.min_delta = 0.01 if menu_args.action in {"runtime-pipeline", "runtime-compare"} else 0.005
    menu_args.runtime_min_delta = 0.01
    menu_args.world_min_delta = 0.005
    menu_args.refresh_current_bench = menu_args.action in {"compare", "deploy-all-best"}
    menu_args.force_deploy = False
    menu_args.lr = 1e-3
    menu_args.weight_decay = 1e-4
    menu_args.hidden_dim = 64
    menu_args.hidden_layers = 2
    menu_args.allow_no_execution = False
    menu_args.include_failures = True
    menu_args.export_dataset = True
    menu_args.runtime_model_out = runtime_model
    menu_args.deploy_runtime = menu_args.action in {"runtime-cycle", "runtime-pipeline"}
    menu_args.json = False
    return menu_args


def apply_inline_value(args: argparse.Namespace, raw: str) -> None:
    parts = raw.split(maxsplit=1)
    if len(parts) == 1:
        return
    value = parts[1].strip()
    if args.action in {"check", "train", "deploy", "deploy-use", "use"}:
        args.run_name = value
    elif args.action == "bench":
        if value.lower().endswith(".onnx"):
            args.model = value
        else:
            args.run_name = value
    elif args.action == "deploy-runtime":
        args.model = value
