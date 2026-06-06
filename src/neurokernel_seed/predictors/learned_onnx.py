from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from neurokernel_seed.core.schema import Action, Prediction, WorldState
from neurokernel_seed.replay.dataset import encode_input_vector
from .base import BasePredictor


class OnnxWorldModelPredictor(BasePredictor):
    name = "learned-onnx"

    def __init__(self, model_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to use OnnxWorldModelPredictor on this machine") from exc
        self.model_path = Path(model_path)
        self.meta_path = self.model_path.with_suffix(".manifest.json")
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(self.meta_path)
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.manifest = self.meta["manifest"]
        self.config = self.meta["config"]
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def predict(self, state: WorldState, action: Action) -> Prediction:
        import numpy as np
        input_vector = self._encode_input(state, action)
        output = self.session.run(None, {"input_vector": np.array([input_vector], dtype=np.float32)})[0][0]
        max_state_dim = int(self.manifest["max_state_dim"])
        reward_index = int(self.manifest["target_layout"]["reward"])
        done_index = int(self.manifest["target_layout"]["done"])
        success_index = self.manifest["target_layout"].get("local_success", self.manifest["target_layout"].get("success"))
        progress_index = self.manifest["target_layout"].get("progress_delta")
        information_index = self.manifest["target_layout"].get("information_gain")
        next_state = tuple(float(value) for value in output[: len(state.vector)])
        reward = float(output[reward_index])
        done_probability = _sigmoid(float(output[done_index]))
        if success_index is None:
            success_probability = done_probability if reward > 0 else 0.0
        else:
            success_probability = _sigmoid(float(output[int(success_index)]))
        confidence = max(0.0, min(1.0, max(abs(done_probability - 0.5), abs(success_probability - 0.5)) * 2.0))
        return Prediction(
            next_state_vector=next_state,
            reward=reward,
            terminal=done_probability >= 0.5,
            confidence=confidence,
            reason="onnx world model",
            success_probability=success_probability,
            progress_delta=float(output[int(progress_index)]) if progress_index is not None else None,
            information_gain=max(0.0, min(1.0, float(output[int(information_index)]))) if information_index is not None else None,
        )

    def _encode_input(self, state: WorldState, action: Action) -> list[float]:
        return encode_input_vector(state.env_name, state.facts, state.vector, action, self.manifest)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))
