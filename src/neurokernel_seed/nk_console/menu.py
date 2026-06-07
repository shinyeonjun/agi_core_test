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
    "auto": "runtime-auto",
    "learn": "runtime-auto",
    "train-runtime": "runtime-train",
    "seed": "runtime-seed",
    "runtime-seed": "runtime-seed",
    "seed-runtime": "runtime-seed",
    "auto-deploy": "runtime-cycle",
    "learn-deploy": "runtime-cycle",
    "cycle": "runtime-cycle",
    "runtime-cycle": "runtime-cycle",
    "dataset": "data",
    "features": "data",
    "runtime-data": "runtime-data",
    "runtime-features": "runtime-features",
    "runtime-train": "runtime-train",
    "runtime-deploy": "deploy-runtime",
    "runtime-use": "deploy-runtime",
    "edge": "current",
    "models": "list",
    "rank": "top",
    "ship": "deploy-best",
    "switch": "use",
    "q": "exit",
    "quit": "exit",
    "exit": "exit",
}


DASHBOARD_COMMANDS = (
    MenuCommand("1", "runtime-seed", "seed", "create runtime seed data only"),
    MenuCommand("2", "runtime-auto", "learn", "runtime data -> features -> train"),
    MenuCommand("3", "runtime-cycle", "cycle", "runtime data -> train -> deploy slot"),
    MenuCommand("4", "compare", "compare", "world current vs best"),
    MenuCommand("5", "deploy-best", "deploy", "activate best world model"),
    MenuCommand("0", "exit", "exit", "close"),
)


KNOWN_ACTIONS = {
    "data",
    "runtime-seed",
    "runtime-auto",
    "runtime-cycle",
    "runtime-data",
    "runtime-features",
    "runtime-train",
    "deploy-runtime",
    "check",
    "train",
    "deploy",
    "deploy-use",
    "deploy-best",
    "bench",
    "bench-current",
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
    menu_args.limit = 5 if menu_args.action in {"status", "top", "compare", "deploy-best"} else None
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
    if menu_args.action == "deploy-runtime":
        menu_args.model = runtime_model
    menu_args.test_ratio = 0.2
    menu_args.project_root = "."
    menu_args.profile = "readonly-basic"
    menu_args.target = "orangepi5"
    menu_args.cycles = 8
    menu_args.min_rows = 1
    menu_args.min_actions = 4 if menu_args.action == "runtime-seed" else 1
    menu_args.lr = 1e-3
    menu_args.weight_decay = 1e-4
    menu_args.hidden_dim = 64
    menu_args.hidden_layers = 2
    menu_args.allow_no_execution = False
    menu_args.include_failures = True
    menu_args.export_dataset = True
    menu_args.runtime_model_out = runtime_model
    menu_args.deploy_runtime = menu_args.action == "runtime-cycle"
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
