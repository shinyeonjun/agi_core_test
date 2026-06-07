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
    "runtime-probe": "runtime-probe",
    "probe-runtime": "runtime-probe",
    "probe": "runtime-probe",
    "런타임프로브": "runtime-probe",
    "후보프로브": "runtime-probe",
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
    "data-audit": "data-audit",
    "audit-data": "data-audit",
    "real-world-transitions": "real-world-transitions",
    "world-real-data": "real-world-transitions",
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


ACTION_PROCESS_STEPS = {
    "train": (
        "world 학습용 features 경로를 확인합니다. NEUROKERNEL_TRAIN_FEATURES가 없으면 직접 입력해야 합니다.",
        "features/manifest 스키마와 dataset gate를 검사합니다.",
        "노트북에서 world model을 학습하고 test split 평가를 저장합니다.",
        "ONNX로 내보내고 gate ablation 벤치를 실행합니다.",
        "후보 run과 benchmark 결과를 artifacts/training_runs에 저장합니다.",
    ),
    "runtime-pipeline": (
        "OrangePi harness DB를 로컬로 캐시합니다.",
        "runtime replay를 만들고 runtime_features로 전처리합니다.",
        "노트북 CUDA/CPU 설정으로 runtime_action 후보 모델을 학습합니다.",
        "로컬 후보와 OrangePi 현행 runtime을 같은 features로 벤치합니다.",
        "로컬 후보가 벤치에서 이기면 runtime 슬롯만 OrangePi에 배포합니다.",
    ),
    "runtime-compare": (
        "로컬 runtime 후보 모델을 현재 runtime_features test split으로 벤치합니다.",
        "OrangePi 현행 runtime 모델을 로컬로 복사해 같은 features로 벤치합니다.",
        "score, 품질 gate, 스키마 호환성을 비교합니다.",
        "비교 리포트를 artifacts/model_benchmarks/runtime_action에 저장합니다.",
    ),
    "deploy-all-best": (
        "OrangePi 현행 world 모델을 캐시하고 world 최고 후보와 벤치 비교합니다.",
        "OrangePi 현행 runtime 모델과 로컬 runtime 후보를 벤치 비교합니다.",
        "world/runtime 중 로컬 후보가 이긴 슬롯만 골라 배포합니다.",
        "통합 배포 리포트를 artifacts/model_benchmarks/deploy에 저장합니다.",
    ),
    "status": (
        "OrangePi의 현재 world/runtime 슬롯을 조회합니다.",
        "로컬 runtime 데이터와 후보 모델 상태를 확인합니다.",
        "다음 추천 작업을 계산해 표시합니다.",
    ),
    "compare": (
        "필요하면 OrangePi 현행 world 모델을 캐시하고 gate ablation 벤치를 실행합니다.",
        "로컬 training_runs에서 최고 world 후보를 찾습니다.",
        "현행 world와 최고 후보의 벤치 점수를 비교합니다.",
    ),
    "current-bench-all": (
        "OrangePi 현행 world 모델을 로컬로 복사하고 gate ablation 벤치를 실행합니다.",
        "OrangePi 현행 runtime 모델을 로컬로 복사하고 현재 runtime_features로 평가합니다.",
        "벤치 환경, 모델 경로, 스키마 호환성, 점수를 JSON 원장에 저장합니다.",
    ),
}


KNOWN_ACTIONS = {
    "data",
    "runtime-pipeline",
    "runtime-seed",
    "runtime-probe",
    "runtime-auto",
    "runtime-cycle",
    "runtime-data",
    "runtime-features",
    "data-audit",
    "real-world-transitions",
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


def process_steps(action: str) -> tuple[str, ...]:
    return ACTION_PROCESS_STEPS.get(normalize_action(action), ())


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
    world_features = os.getenv("NEUROKERNEL_TRAIN_FEATURES")
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
    menu_args.features = world_features if menu_args.action in {"check", "train", "data"} else runtime_features
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
    menu_args.max_candidates = 4
    menu_args.seed_cycles = 0
    menu_args.epochs = 100 if menu_args.action in {"runtime-pipeline", "runtime-train"} else 50
    menu_args.batch_size = 128
    menu_args.device = "cuda" if menu_args.action in {"runtime-pipeline", "runtime-train"} else "auto"
    menu_args.patience = 20
    menu_args.min_rows = 1
    menu_args.min_actions = 4 if menu_args.action in {"runtime-seed", "runtime-probe", "runtime-pipeline"} else 1
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
