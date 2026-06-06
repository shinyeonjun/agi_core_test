from __future__ import annotations

from abc import ABC, abstractmethod

from neurokernel_seed.core.schema import Action, Prediction, WorldState


class BasePredictor(ABC):
    name: str

    @abstractmethod
    def predict(self, state: WorldState, action: Action) -> Prediction:
        ...
