from __future__ import annotations

from neurokernel_seed.core.gate import ActionGate, GateMode, Predictor
from neurokernel_seed.core.schema import Action, WorldState
from neurokernel_seed.envs.base import MicroWorld
from .base import Agent


class GatedAgent(Agent):
    name = "gated"

    def __init__(self, predictor: Predictor, gate_mode: GateMode = "hybrid"):
        self.predictor = predictor
        self.gate_mode = gate_mode

    def act(self, env: MicroWorld, state: WorldState) -> Action:
        gate = ActionGate(env.action_specs, self.predictor, mode=self.gate_mode)
        return gate.choose(state, env.candidate_actions(state)).action
