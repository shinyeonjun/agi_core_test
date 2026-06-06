from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from neurokernel_seed.agents.base import Agent
from neurokernel_seed.core.schema import EpisodeEvent, EpisodeSummary, Prediction
from neurokernel_seed.core.validation import validate_action, validate_prediction, validate_transition
from neurokernel_seed.envs.base import MicroWorld
from neurokernel_seed.storage.sqlite_logger import SQLiteEpisodeLogger


@dataclass(frozen=True)
class EvalResult:
    summaries: list[EpisodeSummary]

    @property
    def success_rate(self) -> float:
        return sum(1 for item in self.summaries if item.success) / len(self.summaries) if self.summaries else 0.0

    @property
    def average_steps(self) -> float:
        return sum(item.steps for item in self.summaries) / len(self.summaries) if self.summaries else 0.0

    @property
    def average_reward(self) -> float:
        return sum(item.total_reward for item in self.summaries) / len(self.summaries) if self.summaries else 0.0

    @property
    def average_prediction_error(self) -> float | None:
        values = [item.prediction_error for item in self.summaries if item.prediction_error is not None]
        return sum(values) / len(values) if values else None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "episodes": len(self.summaries),
            "success_rate": self.success_rate,
            "avg_steps": self.average_steps,
            "avg_reward": self.average_reward,
            "avg_prediction_error": self.average_prediction_error,
        }


class Evaluator:
    def __init__(self, logger: SQLiteEpisodeLogger | None = None):
        self.logger = logger

    def run(self, env: MicroWorld, agent: Agent, episodes: int = 1, seed: int = 0) -> EvalResult:
        summaries: list[EpisodeSummary] = []
        for offset in range(episodes):
            summaries.append(self._run_episode(env, agent, seed + offset))
        return EvalResult(summaries)

    def _run_episode(self, env: MicroWorld, agent: Agent, seed: int) -> EpisodeSummary:
        episode_id = str(uuid.uuid4())
        state = env.reset(seed)
        steps = 0
        total_reward = 0.0
        errors: list[float] = []
        if self.logger:
            self.logger.start_episode(episode_id, env.name, agent.name, seed, env.metadata)
        while not state.terminal and steps < env.max_steps:
            action = agent.act(env, state)
            validate_action(action, env.action_specs)
            prediction = self._predict_if_available(agent, state, action)
            if prediction is not None:
                validate_prediction(prediction, len(state.vector))
            result = env.step(action)
            validate_transition(result, env.action_specs)
            if prediction is not None:
                errors.append(_vector_rmse(prediction.next_state_vector, result.next_state.vector))
            total_reward += result.reward
            event = EpisodeEvent(episode_id, env.name, steps, result, prediction)
            if self.logger:
                self.logger.log_event(event)
            state = result.next_state
            steps += 1
        summary = EpisodeSummary(
            episode_id=episode_id,
            env_name=env.name,
            agent_name=agent.name,
            seed=seed,
            steps=steps,
            total_reward=total_reward,
            success=env.success(state),
            done=state.terminal,
            prediction_error=sum(errors) / len(errors) if errors else None,
        )
        if self.logger:
            self.logger.finish_episode(summary)
        return summary

    @staticmethod
    def _predict_if_available(agent: Agent, state, action) -> Prediction | None:
        predictor = getattr(agent, "predictor", None)
        if predictor is None:
            return None
        return predictor.predict(state, action)


def _vector_rmse(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) / len(left))
