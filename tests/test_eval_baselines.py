from neurokernel_seed.agents.baselines import HeuristicAgent
from neurokernel_seed.agents.gated import GatedAgent
from neurokernel_seed.envs.registry import list_envs, make_env
from neurokernel_seed.eval.evaluator import Evaluator
from neurokernel_seed.predictors.symbolic import SymbolicPredictor


def test_heuristic_baseline_solves_test_worlds():
    for env_name in list_envs("test"):
        result = Evaluator().run(make_env(env_name), HeuristicAgent(), episodes=2)
        assert result.success_rate == 1.0


def test_gated_symbolic_baseline_solves_test_worlds():
    for env_name in list_envs("test"):
        result = Evaluator().run(make_env(env_name), GatedAgent(SymbolicPredictor()), episodes=2)
        assert result.success_rate == 1.0
        assert result.average_prediction_error == 0.0


def test_symbolic_predictor_uses_env_config_for_train_and_test_worlds():
    for split in ("train", "test"):
        for env_name in list_envs(split):
            result = Evaluator().run(make_env(env_name), GatedAgent(SymbolicPredictor()), episodes=1)
            assert result.success_rate == 1.0
            assert result.average_prediction_error == 0.0
