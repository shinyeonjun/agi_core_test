from __future__ import annotations

import random

from neurokernel_seed.core.schema import Action, WorldState
from neurokernel_seed.envs.base import MicroWorld
from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.random = random.Random(seed)

    def act(self, env: MicroWorld, state: WorldState) -> Action:
        return self.random.choice(env.candidate_actions(state))


class HeuristicAgent(Agent):
    name = "heuristic"

    def act(self, env: MicroWorld, state: WorldState) -> Action:
        return env.expert_action(state)
