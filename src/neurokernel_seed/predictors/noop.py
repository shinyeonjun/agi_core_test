from __future__ import annotations

from neurokernel_seed.core.schema import Action, Prediction, WorldState
from .base import BasePredictor


class NoOpPredictor(BasePredictor):
    name = "noop"

    def predict(self, state: WorldState, action: Action) -> Prediction:
        return Prediction(state.vector, reward=0.0, terminal=state.terminal, confidence=0.1, reason="no-op prediction")
