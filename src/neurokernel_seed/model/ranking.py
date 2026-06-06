from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import TorchFeatureDataset, load_feature_manifest, load_feature_rows, require_torch
from .mlp import WorldModelConfig, build_model
from .train import resolve_device


def eval_action_ranking(checkpoint_path: str | Path, features_path: str | Path, split: str = "test", device: str = "auto") -> dict[str, Any]:
    torch = require_torch()
    eval_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    manifest = checkpoint["manifest"]
    feature_manifest = load_feature_manifest(features_path)
    if feature_manifest["schema_version"] != manifest["schema_version"]:
        raise ValueError("feature schema mismatch")
    if "progress_delta" not in manifest["target_layout"]:
        raise ValueError("action ranking requires feature schema with progress_delta targets")
    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(eval_device)
    rows = load_feature_rows(features_path, split)
    dataset = TorchFeatureDataset(rows)
    model.eval()
    with torch.no_grad():
        prediction = model(dataset.inputs.to(eval_device)).detach().cpu()
    groups = _candidate_groups(dataset)
    metrics = _ranking_metrics(groups, prediction, dataset, manifest)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["features"] = str(features_path)
    metrics["split"] = split
    metrics["device"] = str(eval_device)
    return metrics


def _candidate_groups(dataset: TorchFeatureDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.meta):
        if not sample.candidate_set_id:
            continue
        groups.setdefault(sample.candidate_set_id, []).append(index)
    if not groups:
        raise ValueError("no candidate_set_id values found; export counterfactual feature v3 first")
    return groups


def _ranking_metrics(groups: dict[str, list[int]], prediction, dataset: TorchFeatureDataset, manifest: dict[str, Any]) -> dict[str, Any]:
    top1 = 0
    top2 = 0
    first_step_top1 = 0
    first_step_total = 0
    pairwise_correct = 0
    pairwise_total = 0
    regret_total = 0.0
    env_stats: dict[str, dict[str, float]] = {}
    for indices in groups.values():
        actual_scores = [float(dataset.meta[index].actual_action_score or 0.0) for index in indices]
        predicted_scores = [_predicted_action_score(prediction[index], manifest) for index in indices]
        actual_best_pos = max(range(len(indices)), key=lambda pos: actual_scores[pos])
        predicted_order = sorted(range(len(indices)), key=lambda pos: predicted_scores[pos], reverse=True)
        predicted_best_pos = predicted_order[0]
        chosen_actual = actual_scores[predicted_best_pos]
        best_actual = actual_scores[actual_best_pos]
        hit_top1 = 1.0 if predicted_best_pos == actual_best_pos else 0.0
        hit_top2 = 1.0 if actual_best_pos in predicted_order[: min(2, len(predicted_order))] else 0.0
        regret = best_actual - chosen_actual
        top1 += int(hit_top1)
        top2 += int(hit_top2)
        regret_total += regret
        sample = dataset.meta[indices[0]]
        stat = env_stats.setdefault(sample.env_name, {"groups": 0.0, "top1": 0.0, "top2": 0.0, "regret": 0.0, "first_step_groups": 0.0, "first_step_top1": 0.0})
        stat["groups"] += 1.0
        stat["top1"] += hit_top1
        stat["top2"] += hit_top2
        stat["regret"] += regret
        if sample.step_index == 0:
            first_step_total += 1
            first_step_top1 += int(hit_top1)
            stat["first_step_groups"] += 1.0
            stat["first_step_top1"] += hit_top1
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                actual_delta = actual_scores[left] - actual_scores[right]
                predicted_delta = predicted_scores[left] - predicted_scores[right]
                if actual_delta == 0:
                    continue
                pairwise_total += 1
                if actual_delta * predicted_delta > 0:
                    pairwise_correct += 1
    group_count = len(groups)
    return {
        "candidate_groups": group_count,
        "grouped_top1_action_accuracy": top1 / group_count,
        "grouped_top2_action_recall": top2 / group_count,
        "pairwise_ranking_accuracy": pairwise_correct / pairwise_total if pairwise_total else 0.0,
        "counterfactual_regret": regret_total / group_count,
        "first_step_accuracy": first_step_top1 / first_step_total if first_step_total else 0.0,
        "first_step_groups": first_step_total,
        "by_env": _finalize_env_stats(env_stats),
    }


def _predicted_action_score(row_prediction, manifest: dict[str, Any]) -> float:
    torch = require_torch()
    layout = manifest["target_layout"]
    reward = float(row_prediction[int(layout["reward"])])
    local_success = float(torch.sigmoid(row_prediction[int(layout["local_success"])]))
    progress_delta = float(row_prediction[int(layout["progress_delta"])])
    information_gain = max(0.0, min(1.0, float(row_prediction[int(layout["information_gain"])])))
    return reward + 2.0 * progress_delta + information_gain + 0.5 * local_success


def _finalize_env_stats(env_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for env_name, stat in sorted(env_stats.items()):
        groups = max(1.0, stat["groups"])
        first_groups = stat["first_step_groups"]
        output[env_name] = {
            "candidate_groups": stat["groups"],
            "grouped_top1_action_accuracy": stat["top1"] / groups,
            "grouped_top2_action_recall": stat["top2"] / groups,
            "counterfactual_regret": stat["regret"] / groups,
            "first_step_accuracy": stat["first_step_top1"] / first_groups if first_groups else 0.0,
            "first_step_groups": first_groups,
        }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval-action-ranking")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    print(json.dumps(eval_action_ranking(args.checkpoint, args.features, args.split, args.device), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
