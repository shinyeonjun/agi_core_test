from __future__ import annotations

from abc import ABC, abstractmethod

from neurokernel_seed.core.schema import Action, WorldState
from neurokernel_seed.envs.base import MicroWorld


class Agent(ABC):
    name: str

    @abstractmethod
    def act(self, env: MicroWorld, state: WorldState) -> Action:
        ...
