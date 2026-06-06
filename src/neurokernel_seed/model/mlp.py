from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dataset import require_torch


@dataclass(frozen=True)
class WorldModelConfig:
    input_dim: int
    target_dim: int
    hidden_dim: int = 64
    hidden_layers: int = 2

    def as_dict(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldModelConfig":
        return cls(
            input_dim=int(data["input_dim"]),
            target_dim=int(data["target_dim"]),
            hidden_dim=int(data.get("hidden_dim", 64)),
            hidden_layers=int(data.get("hidden_layers", 2)),
        )


def build_model(config: WorldModelConfig):
    torch = require_torch()
    layers = []
    current = config.input_dim
    for _ in range(config.hidden_layers):
        layers.append(torch.nn.Linear(current, config.hidden_dim))
        layers.append(torch.nn.ReLU())
        current = config.hidden_dim
    layers.append(torch.nn.Linear(current, config.target_dim))
    return torch.nn.Sequential(*layers)


def compute_losses(prediction, target, target_mask, manifest: dict[str, Any], weights: dict[str, float] | None = None) -> dict[str, Any]:
    torch = require_torch()
    weights = weights or {"state": 1.0, "reward": 0.5, "done": 0.5, "local_success": 0.7, "progress_delta": 1.5, "information_gain": 0.8}
    max_state_dim = int(manifest["max_state_dim"])
    reward_index = int(manifest["target_layout"]["reward"])
    done_index = int(manifest["target_layout"]["done"])
    success_index = manifest["target_layout"].get("local_success", manifest["target_layout"].get("success"))
    progress_index = manifest["target_layout"].get("progress_delta")
    information_index = manifest["target_layout"].get("information_gain")

    state_pred = prediction[:, :max_state_dim]
    state_target = target[:, :max_state_dim]
    state_mask = target_mask[:, :max_state_dim]
    state_denominator = torch.clamp(state_mask.sum(), min=1.0)
    state_element_loss = torch.nn.functional.smooth_l1_loss(state_pred, state_target, reduction="none")
    state_loss = (state_element_loss * state_mask).sum() / state_denominator

    reward_loss = torch.nn.functional.smooth_l1_loss(prediction[:, reward_index], target[:, reward_index])
    done_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction[:, done_index], target[:, done_index])
    total = weights["state"] * state_loss + weights["reward"] * reward_loss + weights["done"] * done_loss
    losses = {"total": total, "state": state_loss, "reward": reward_loss, "done": done_loss}
    if success_index is not None:
        success_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction[:, int(success_index)], target[:, int(success_index)])
        losses["local_success"] = success_loss
        losses["total"] = losses["total"] + weights["local_success"] * success_loss
    if progress_index is not None:
        progress_loss = torch.nn.functional.smooth_l1_loss(prediction[:, int(progress_index)], target[:, int(progress_index)])
        losses["progress_delta"] = progress_loss
        losses["total"] = losses["total"] + weights["progress_delta"] * progress_loss
    if information_index is not None:
        information_loss = torch.nn.functional.smooth_l1_loss(prediction[:, int(information_index)], target[:, int(information_index)])
        losses["information_gain"] = information_loss
        losses["total"] = losses["total"] + weights["information_gain"] * information_loss
    return losses


def compute_metrics(prediction, target, target_mask, manifest: dict[str, Any]) -> dict[str, float]:
    torch = require_torch()
    max_state_dim = int(manifest["max_state_dim"])
    reward_index = int(manifest["target_layout"]["reward"])
    done_index = int(manifest["target_layout"]["done"])
    success_index = manifest["target_layout"].get("local_success", manifest["target_layout"].get("success"))
    progress_index = manifest["target_layout"].get("progress_delta")
    information_index = manifest["target_layout"].get("information_gain")
    with torch.no_grad():
        state_mask = target_mask[:, :max_state_dim]
        state_denominator = torch.clamp(state_mask.sum(), min=1.0)
        state_mse = ((((prediction[:, :max_state_dim] - target[:, :max_state_dim]) ** 2) * state_mask).sum() / state_denominator).item()
        state_mae = ((torch.abs(prediction[:, :max_state_dim] - target[:, :max_state_dim]) * state_mask).sum() / state_denominator).item()
        reward_mse = torch.nn.functional.mse_loss(prediction[:, reward_index], target[:, reward_index]).item()
        reward_mae = torch.nn.functional.l1_loss(prediction[:, reward_index], target[:, reward_index]).item()
        done_prob = torch.sigmoid(prediction[:, done_index])
        done_pred = (done_prob >= 0.5).float()
        done_acc = (done_pred == target[:, done_index]).float().mean().item()
        metrics = {
            "state_mse": float(state_mse),
            "state_mae": float(state_mae),
            "reward_mse": float(reward_mse),
            "reward_mae": float(reward_mae),
            "done_accuracy": float(done_acc),
        }
        if success_index is not None:
            success_prob = torch.sigmoid(prediction[:, int(success_index)])
            success_pred = (success_prob >= 0.5).float()
            success_acc = (success_pred == target[:, int(success_index)]).float().mean().item()
            metrics["local_success_accuracy"] = float(success_acc)
        if progress_index is not None:
            progress_mae = torch.nn.functional.l1_loss(prediction[:, int(progress_index)], target[:, int(progress_index)]).item()
            progress_mse = torch.nn.functional.mse_loss(prediction[:, int(progress_index)], target[:, int(progress_index)]).item()
            metrics["progress_delta_mae"] = float(progress_mae)
            metrics["progress_delta_mse"] = float(progress_mse)
        if information_index is not None:
            information_mae = torch.nn.functional.l1_loss(prediction[:, int(information_index)], target[:, int(information_index)]).item()
            information_mse = torch.nn.functional.mse_loss(prediction[:, int(information_index)], target[:, int(information_index)]).item()
            metrics["information_gain_mae"] = float(information_mae)
            metrics["information_gain_mse"] = float(information_mse)
    return metrics
