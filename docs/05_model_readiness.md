# Model Readiness

This document defines the non-model contract required before starting the v0 world-model experiment.

## Completed Non-Model Contracts

- Environments: lock, tool, maze with train/test splits
- Baselines: random, heuristic, gated
- Storage: SQLite episodes/events
- Replay schema: neurokernel-replay-v2
- Counterfactual schema: neurokernel-counterfactual-transition-v1
- Feature schema: neurokernel-flat-transition-v3

## Flat Transition Feature

The v0 world model should consume fixed-shape numeric samples, not raw replay JSON.

```text
input_vector = env_family_onehot + task_config + state_padded + action_onehot
target_vector = next_state_padded + reward + done + local_success + progress_delta + information_gain
target_mask = valid_next_state_dims + reward_mask + done_mask + local_success_mask + progress_delta_mask + information_gain_mask
```

Current all-split dimensions:

```text
env_family_onehot: 3
task_config: 22
state_padded: 4
action_onehot: 12
input_dim: 41
target_dim: 7
v3 target_dim: 9
```

The target_mask prevents padded state dimensions from contributing to next-state loss. The task_config block exposes task conditions such as lock code, maze target color, tool sequence order, and max_steps so the model does not have to memorize train/test env names.

## Generate Model-Ready Data

```bash
cd ~/projects/neurokernel-agi-seed
source venv/bin/activate
python -m neurokernel_seed.cli collect-counterfactual-dataset \
  --split all \
  --episodes 10 \
  --agents random heuristic gated \
  --counterfactual-out data/model_ready/counterfactual.jsonl \
  --features-out data/model_ready/features.jsonl
```

Validate outputs:

```bash
python -m neurokernel_seed.cli validate-counterfactual data/model_ready/counterfactual.jsonl
python -m neurokernel_seed.cli validate-features data/model_ready/features.jsonl
```

## v0 Model Scope

Start with a flat MLP only.

```text
input: input_vector
outputs:
- next_state_padded
- reward
- done
- local_success
- progress_delta
- information_gain
loss:
- masked SmoothL1 for next_state
- SmoothL1 for reward
- BCE for done
- BCE for local_success
- SmoothL1 for progress_delta
- SmoothL1 for information_gain
```

Do not start with:

- object-centric models
- memory-augmented models
- prioritized replay
- graph neural networks
- Dreamer/latent world models
- RKNN optimization

## Readiness Checklist

- pytest passes
- replay v2 validation passes for trajectory data
- counterfactual v1 validation passes for action ranking data
- feature v3 validation passes
- train/test splits are recorded in the feature manifest
- random failure data and heuristic/gated success data are collected together
- action vocabulary is fixed in the manifest
- input and target dimensions are fixed in the manifest
- evaluation reports next_state/reward/done/local_success/progress_delta/information_gain metrics by env and by action
- action ranking evaluation reports grouped_top1_action_accuracy, grouped_top2_action_recall, pairwise_ranking_accuracy, counterfactual_regret, and first_step_accuracy
