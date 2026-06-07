from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from neurokernel_seed.model.dataset import TorchFeatureDataset, load_feature_manifest, load_feature_rows, require_torch
from neurokernel_seed.model.mlp import WorldModelConfig, build_model
from neurokernel_seed.model.train import resolve_device
from neurokernel_seed.replay.runtime_features import (
    RUNTIME_FEATURE_SCHEMA_VERSION,
    TARGET_NAMES,
    check_runtime_feature_gates,
    validate_runtime_features,
)


RUNTIME_ACTION_MODEL_SCHEMA_VERSION = "neurokernel-runtime-action-model-v1"
RUNTIME_LOSS_WEIGHTS = {
    "success": 1.0,
    "reward": 0.5,
    "duration_seconds_log1p": 0.2,
    "failure_present": 0.8,
}


def train_runtime_action_model(
    features_path: str | Path,
    out_path: str | Path,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    seed: int = 11,
    device: str = "auto",
    patience: int = 20,
    min_rows: int = 10,
    min_actions: int = 1,
) -> dict[str, Any]:
    torch = require_torch()
    train_device = resolve_device(device)
    manifest = _load_runtime_feature_manifest(features_path)
    gate = check_runtime_feature_gates(features_path, min_rows=min_rows, min_actions=min_actions)
    if not gate["ready_for_runtime_model_training"]:
        raise RuntimeError(f"runtime feature gates failed: {gate['gates']}")
    train_rows = load_feature_rows(features_path, "train")
    test_rows = load_feature_rows(features_path, "test")

    torch.manual_seed(seed)
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_ds = TorchFeatureDataset(train_rows)
    test_ds = TorchFeatureDataset(test_rows)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    config = WorldModelConfig(
        input_dim=int(manifest["input_dim"]),
        target_dim=int(manifest["target_dim"]),
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )
    model = build_model(config).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    best_test: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets, masks in loader:
            inputs = inputs.to(train_device)
            targets = targets.to(train_device)
            masks = masks.to(train_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            losses = compute_runtime_action_losses(prediction, targets, masks, manifest)
            losses["total"].backward()
            optimizer.step()
            total_loss += float(losses["total"].detach().item())
            batches += 1
        metrics = evaluate_runtime_action_model(model, test_ds, manifest, train_device)
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_test = metrics
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append({"epoch": epoch, "device": str(train_device), "train_loss": total_loss / max(1, batches), **metrics})
        if patience > 0 and epoch - best_epoch >= patience:
            history.append({"epoch": epoch, "device": str(train_device), "early_stopped": 1, "best_epoch": best_epoch, "best_loss": best_loss})
            break

    model.load_state_dict(best_state)
    model.to(train_device)
    final_train = evaluate_runtime_action_model(model, train_ds, manifest, train_device, detailed=True)
    final_test = evaluate_runtime_action_model(model, test_ds, manifest, train_device, detailed=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": RUNTIME_ACTION_MODEL_SCHEMA_VERSION,
        "model_state_dict": best_state,
        "config": config.as_dict(),
        "manifest": manifest,
        "device": str(train_device),
        "best_epoch": best_epoch,
        "metrics": {
            "train": final_train,
            "test": final_test,
            "best_test": best_test,
            "history": history,
            "weight_decay": weight_decay,
            "patience": patience,
            "dataset_gate": gate,
        },
    }
    torch.save(checkpoint, out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(checkpoint["metrics"], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": RUNTIME_ACTION_MODEL_SCHEMA_VERSION,
                "model_slot": "runtime_action_model",
                "checkpoint": str(out),
                "metrics": str(metrics_path),
                "feature_schema_version": manifest.get("schema_version"),
                "rows": manifest.get("rows"),
                "input_dim": manifest.get("input_dim"),
                "target_dim": manifest.get("target_dim"),
                "target_names": manifest.get("target_names"),
                "data_origin": "runtime_experience_log",
                "activation_status": "candidate",
                "dataset_gate": gate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "checkpoint": str(out),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "device": str(train_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "best_epoch": best_epoch,
        "train": final_train,
        "test": final_test,
        "history": history,
        "dataset_gate": gate,
    }


def eval_runtime_action_checkpoint(
    checkpoint_path: str | Path,
    features_path: str | Path,
    split: str = "test",
    device: str = "auto",
) -> dict[str, Any]:
    torch = require_torch()
    eval_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    manifest = checkpoint["manifest"]
    feature_manifest = _load_runtime_feature_manifest(features_path)
    if feature_manifest["schema_version"] != manifest["schema_version"]:
        raise ValueError("runtime feature schema mismatch")
    if int(feature_manifest["input_dim"]) != int(manifest["input_dim"]):
        raise ValueError("runtime feature input dimension mismatch")
    config = WorldModelConfig.from_dict(checkpoint["config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(eval_device)
    rows = load_feature_rows(features_path, split)
    dataset = TorchFeatureDataset(rows)
    metrics = evaluate_runtime_action_model(model, dataset, manifest, eval_device, detailed=True)
    return {"checkpoint": str(checkpoint_path), "features": str(features_path), "split": split, "device": str(eval_device), **metrics}


def compute_runtime_action_losses(prediction, target, target_mask, manifest: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    layout = manifest["target_layout"]
    success_index = int(layout["success"])
    reward_index = int(layout["reward"])
    duration_index = int(layout["duration_seconds_log1p"])
    failure_index = int(layout["failure_present"])
    success_loss = _masked_bce_with_logits(prediction[:, success_index], target[:, success_index], target_mask[:, success_index], torch)
    reward_loss = _masked_smooth_l1(prediction[:, reward_index], target[:, reward_index], target_mask[:, reward_index], torch)
    duration_loss = _masked_smooth_l1(prediction[:, duration_index], target[:, duration_index], target_mask[:, duration_index], torch)
    failure_loss = _masked_bce_with_logits(prediction[:, failure_index], target[:, failure_index], target_mask[:, failure_index], torch)
    total = (
        RUNTIME_LOSS_WEIGHTS["success"] * success_loss
        + RUNTIME_LOSS_WEIGHTS["reward"] * reward_loss
        + RUNTIME_LOSS_WEIGHTS["duration_seconds_log1p"] * duration_loss
        + RUNTIME_LOSS_WEIGHTS["failure_present"] * failure_loss
    )
    return {
        "total": total,
        "success": success_loss,
        "reward": reward_loss,
        "duration_seconds_log1p": duration_loss,
        "failure_present": failure_loss,
    }


def _masked_bce_with_logits(prediction, target, mask, torch):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction="none")
    return (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _masked_smooth_l1(prediction, target, mask, torch):
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    return (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def evaluate_runtime_action_model(model, dataset: TorchFeatureDataset, manifest: dict[str, Any], device=None, detailed: bool = False) -> dict[str, Any]:
    torch = require_torch()
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = dataset.inputs.to(device)
        targets = dataset.targets.to(device)
        masks = dataset.masks.to(device)
        prediction = model(inputs)
        losses = compute_runtime_action_losses(prediction, targets, masks, manifest)
        metrics = compute_runtime_action_metrics(prediction, targets, masks, manifest)
        ranking = compute_runtime_action_ranking_metrics(prediction, targets, masks, dataset, manifest)
        result: dict[str, Any] = {"loss": float(losses["total"].item()), **metrics, "rows": float(len(dataset))}
        result.update(ranking)
        if detailed:
            result["by_action"] = _runtime_group_metrics(prediction, targets, dataset, manifest, "action_key")
            result["by_env"] = _runtime_group_metrics(prediction, targets, dataset, manifest, "env_name")
        return result


def compute_runtime_action_metrics(prediction, target, target_mask, manifest: dict[str, Any]) -> dict[str, float]:
    torch = require_torch()
    layout = manifest["target_layout"]
    success_index = int(layout["success"])
    reward_index = int(layout["reward"])
    duration_index = int(layout["duration_seconds_log1p"])
    failure_index = int(layout["failure_present"])
    success_prob = torch.sigmoid(prediction[:, success_index])
    failure_prob = torch.sigmoid(prediction[:, failure_index])
    success_pred = (success_prob >= 0.5).float()
    failure_pred = (failure_prob >= 0.5).float()
    reward_mask = target_mask[:, reward_index]
    duration_mask = target_mask[:, duration_index]
    success_mask = target_mask[:, success_index]
    failure_mask = target_mask[:, failure_index]
    reward_error = prediction[:, reward_index] - target[:, reward_index]
    duration_error = prediction[:, duration_index] - target[:, duration_index]
    return {
        "known_success_rows": float(success_mask.sum().item()),
        "success_accuracy": _masked_accuracy(success_pred, target[:, success_index], success_mask, torch),
        "failure_present_accuracy": _masked_accuracy(failure_pred, target[:, failure_index], failure_mask, torch),
        "reward_mae": _masked_mean(torch.abs(reward_error), reward_mask, torch),
        "reward_mse": _masked_mean(reward_error**2, reward_mask, torch),
        "duration_log1p_mae": _masked_mean(torch.abs(duration_error), duration_mask, torch),
        "duration_log1p_mse": _masked_mean(duration_error**2, duration_mask, torch),
        "mean_success_probability": float(success_prob.mean().item()),
        "mean_failure_probability": float(failure_prob.mean().item()),
    }


def compute_runtime_action_ranking_metrics(prediction, target, target_mask, dataset: TorchFeatureDataset, manifest: dict[str, Any]) -> dict[str, float]:
    torch = require_torch()
    layout = manifest["target_layout"]
    success_index = int(layout["success"])
    reward_index = int(layout["reward"])
    duration_index = int(layout["duration_seconds_log1p"])
    failure_index = int(layout["failure_present"])
    success_prob = torch.sigmoid(prediction[:, success_index])
    failure_prob = torch.sigmoid(prediction[:, failure_index])
    predicted_score = success_prob + 0.5 * prediction[:, reward_index] - 0.2 * torch.clamp(prediction[:, duration_index], min=0.0) - 0.8 * failure_prob
    actual_score = target[:, success_index] + 0.5 * target[:, reward_index] - 0.2 * target[:, duration_index] - 0.8 * target[:, failure_index]
    known_mask = (
        (target_mask[:, success_index] > 0.0)
        & (target_mask[:, reward_index] > 0.0)
        & (target_mask[:, duration_index] > 0.0)
        & (target_mask[:, failure_index] > 0.0)
    )
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.meta):
        group_id = sample.candidate_set_id or f"row_{index}"
        groups.setdefault(str(group_id), []).append(index)
    candidate_groups = sum(1 for indices in groups.values() if len(indices) >= 2)
    evaluated = 0
    top1_correct = 0
    regret_values: list[float] = []
    pairwise_total = 0
    pairwise_correct = 0
    for indices in groups.values():
        known_indices = [index for index in indices if bool(known_mask[index].item())]
        if len(known_indices) < 2:
            continue
        evaluated += 1
        pred_values = predicted_score[known_indices]
        actual_values = actual_score[known_indices]
        predicted_best_offset = int(torch.argmax(pred_values).item())
        actual_best_offset = int(torch.argmax(actual_values).item())
        if predicted_best_offset == actual_best_offset:
            top1_correct += 1
        regret_values.append(float((actual_values[actual_best_offset] - actual_values[predicted_best_offset]).item()))
        for left in range(len(known_indices)):
            for right in range(left + 1, len(known_indices)):
                actual_delta = float((actual_values[left] - actual_values[right]).item())
                if actual_delta == 0.0:
                    continue
                predicted_delta = float((pred_values[left] - pred_values[right]).item())
                pairwise_total += 1
                if (actual_delta > 0.0 and predicted_delta > 0.0) or (actual_delta < 0.0 and predicted_delta < 0.0):
                    pairwise_correct += 1
    top1_accuracy = float(top1_correct / evaluated) if evaluated else 0.0
    mean_regret = float(sum(regret_values) / len(regret_values)) if regret_values else 0.0
    pairwise_accuracy = float(pairwise_correct / pairwise_total) if pairwise_total else 0.0
    return {
        "ranking_candidate_groups": float(candidate_groups),
        "ranking_evaluable_groups": float(evaluated),
        "ranking_skipped_groups": float(max(0, candidate_groups - evaluated)),
        "ranking_top1_accuracy": top1_accuracy,
        "top1_action_accuracy": top1_accuracy,
        "ranking_mean_regret": mean_regret,
        "mean_best_action_regret": mean_regret,
        "ranking_pairwise_accuracy": pairwise_accuracy,
        "mean_pairwise_ranking_accuracy": pairwise_accuracy,
        "ranking_pairwise_pairs": float(pairwise_total),
    }


def _masked_accuracy(prediction, target, mask, torch) -> float:
    correct = (prediction == target).float()
    return _masked_mean(correct, mask, torch)


def _masked_mean(values, mask, torch) -> float:
    denominator = mask.sum()
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float(((values * mask).sum() / torch.clamp(denominator, min=1.0)).item())


def _runtime_group_metrics(prediction, targets, dataset: TorchFeatureDataset, manifest: dict[str, Any], field: str) -> dict[str, dict[str, float]]:
    torch = require_torch()
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.meta):
        groups.setdefault(str(getattr(sample, field)), []).append(index)
    result: dict[str, dict[str, float]] = {}
    for key, indices in sorted(groups.items()):
        idx = torch.tensor(indices, dtype=torch.long, device=prediction.device)
        masks = dataset.masks.to(prediction.device).index_select(0, idx)
        metrics = compute_runtime_action_metrics(prediction.index_select(0, idx), targets.index_select(0, idx), masks, manifest)
        result[key] = {**metrics, "rows": float(len(indices))}
    return result


def _load_runtime_feature_manifest(path: str | Path) -> dict[str, Any]:
    validation = validate_runtime_features(path)
    if not validation["accepted"]:
        raise ValueError(f"runtime feature validation failed: {validation['errors'][:3]}")
    manifest = load_feature_manifest(path)
    if manifest.get("schema_version") != RUNTIME_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"expected {RUNTIME_FEATURE_SCHEMA_VERSION}, got {manifest.get('schema_version')}")
    missing_targets = [name for name in TARGET_NAMES if name not in manifest.get("target_layout", {})]
    if missing_targets:
        raise ValueError(f"runtime feature manifest missing targets: {missing_targets}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train-runtime-action-model")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="artifacts/runtime_action_model.pt")
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
    args = parser.parse_args(argv)
    result = train_runtime_action_model(
        args.features,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        device=args.device,
        patience=args.patience,
        min_rows=args.min_rows,
        min_actions=args.min_actions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
