from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .dataset import TorchFeatureDataset, load_feature_manifest, load_feature_rows, require_torch
from .mlp import WorldModelConfig, build_model, compute_losses, compute_metrics


def resolve_device(device: str):
    torch = require_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return torch.device(device)


def train_world_model(
    features_path: str | Path,
    out_path: str | Path,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    hidden_layers: int = 2,
    seed: int = 7,
    device: str = "auto",
    patience: int = 60,
) -> dict:
    torch = require_torch()
    train_device = resolve_device(device)
    manifest = load_feature_manifest(features_path)
    train_rows = load_feature_rows(features_path, "train")
    test_rows = load_feature_rows(features_path, "test")
    torch.manual_seed(seed)
    if train_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    train_ds = TorchFeatureDataset(train_rows)
    test_ds = TorchFeatureDataset(test_rows)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    config = WorldModelConfig(input_dim=int(manifest["input_dim"]), target_dim=int(manifest["target_dim"]), hidden_dim=hidden_dim, hidden_layers=hidden_layers)
    model = build_model(config).to(train_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float | int | str]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    best_test: dict[str, float] = {}
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
            losses = compute_losses(prediction, targets, masks, manifest)
            losses["total"].backward()
            optimizer.step()
            total_loss += float(losses["total"].detach().item())
            batches += 1
        metrics = evaluate_model(model, test_ds, manifest, train_device)
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
    final_train = evaluate_model(model, train_ds, manifest, train_device, detailed=True)
    final_test = evaluate_model(model, test_ds, manifest, train_device, detailed=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "config": config.as_dict(),
        "manifest": manifest,
        "device": str(train_device),
        "best_epoch": best_epoch,
        "metrics": {"train": final_train, "test": final_test, "best_test": best_test, "history": history, "weight_decay": weight_decay, "patience": patience},
    }
    torch.save(checkpoint, out)
    metrics_path = out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(checkpoint["metrics"], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"checkpoint": str(out), "metrics": str(metrics_path), "device": str(train_device), "cuda_available": bool(torch.cuda.is_available()), "best_epoch": best_epoch, "train": final_train, "test": final_test, "history": history, "weight_decay": weight_decay, "patience": patience}


def evaluate_model(model, dataset: TorchFeatureDataset, manifest: dict, device=None, detailed: bool = False) -> dict:
    torch = require_torch()
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs = dataset.inputs.to(device)
        targets = dataset.targets.to(device)
        masks = dataset.masks.to(device)
        prediction = model(inputs)
        losses = compute_losses(prediction, targets, masks, manifest)
        metrics = compute_metrics(prediction, targets, masks, manifest)
        result = {"loss": float(losses["total"].item()), **metrics, "rows": float(len(dataset))}
        if detailed:
            result["by_env"] = _group_metrics(prediction, targets, masks, dataset, manifest, "env_name")
            result["by_action"] = _group_metrics(prediction, targets, masks, dataset, manifest, "action_key")
        return result


def _group_metrics(prediction, targets, masks, dataset: TorchFeatureDataset, manifest: dict, field: str) -> dict[str, dict[str, float]]:
    torch = require_torch()
    groups: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset.meta):
        groups.setdefault(str(getattr(sample, field)), []).append(index)
    result: dict[str, dict[str, float]] = {}
    for key, indices in sorted(groups.items()):
        idx = torch.tensor(indices, dtype=torch.long, device=prediction.device)
        metrics = compute_metrics(prediction.index_select(0, idx), targets.index_select(0, idx), masks.index_select(0, idx), manifest)
        result[key] = {**metrics, "rows": float(len(indices))}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train-world-model")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", default="artifacts/world_model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args(argv)
    result = train_world_model(
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
