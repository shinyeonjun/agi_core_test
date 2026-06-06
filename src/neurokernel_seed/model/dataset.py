from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FeatureSample:
    input_vector: list[float]
    target_vector: list[float]
    target_mask: list[float]
    split: str
    env_name: str
    action_key: str
    step_index: int = 0
    candidate_set_id: str | None = None
    actual_action_score: float | None = None


def load_feature_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.suffix != ".json":
        manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_feature_samples(path: str | Path) -> Iterable[FeatureSample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield FeatureSample(
                input_vector=[float(value) for value in row["input_vector"]],
                target_vector=[float(value) for value in row["target_vector"]],
                target_mask=[float(value) for value in row["target_mask"]],
                split=str(row["split"]),
                env_name=str(row["env_name"]),
                action_key=str(row["action_key"]),
                step_index=int(row.get("step_index", 0)),
                candidate_set_id=row.get("candidate_set_id"),
                actual_action_score=float(row["actual_action_score"]) if row.get("actual_action_score") is not None else None,
            )


def load_feature_rows(path: str | Path, split: str | None = None) -> list[FeatureSample]:
    rows = list(iter_feature_samples(path))
    if split is not None and split != "all":
        rows = [row for row in rows if row.split == split]
    if not rows:
        raise ValueError(f"no feature rows found for split={split!r}")
    return rows


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for model training/evaluation. Install torch in the notebook training environment.") from exc
    return torch


class TorchFeatureDataset:
    def __init__(self, rows: list[FeatureSample]):
        torch = require_torch()
        self.inputs = torch.tensor([row.input_vector for row in rows], dtype=torch.float32)
        self.targets = torch.tensor([row.target_vector for row in rows], dtype=torch.float32)
        self.masks = torch.tensor([row.target_mask for row in rows], dtype=torch.float32)
        self.meta = rows

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], self.masks[index]
