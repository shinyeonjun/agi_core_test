from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import TorchFeatureDataset, load_feature_manifest, load_feature_rows, require_torch
from .mlp import WorldModelConfig, build_model
from .train import evaluate_model, resolve_device


def eval_world_model(checkpoint_path: str | Path, features_path: str | Path, split: str = "test", device: str = "auto") -> dict:
    torch = require_torch()
    eval_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    manifest = checkpoint["manifest"]
    if load_feature_manifest(features_path)["schema_version"] != manifest["schema_version"]:
        raise ValueError("feature schema mismatch")
    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(eval_device)
    rows = load_feature_rows(features_path, split)
    dataset = TorchFeatureDataset(rows)
    metrics = evaluate_model(model, dataset, manifest, eval_device, detailed=True)
    return {"checkpoint": str(checkpoint_path), "features": str(features_path), "split": split, "device": str(eval_device), **metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-world-model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(eval_world_model(args.checkpoint, args.features, args.split, args.device), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
