# Benchmark Protocol

This document defines the repeatable benchmark used before changing the model, schema, dataset, or gate.

## Benchmark Goals

- Prove the current baseline still works.
- Detect whether new probes fail in known, explainable ways.
- Keep model improvement decisions based on measured failures, not vibes.

## Standard Commands

Run unit and contract tests first:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m pytest tests
```

Run the standard model benchmark:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-benchmark `
  --model D:\agi_seed\artifacts\world_model_v4.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\v4_1 `
  --episodes 5 `
  --trace-episodes 3 `
  --max-failures-per-env 3
```

Run strict mode only when deciding whether a vNext model is ready to replace the baseline:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-benchmark `
  --model D:\agi_seed\artifacts\world_model_vNext.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\vNext_strict `
  --episodes 10 `
  --trace-episodes 5 `
  --max-failures-per-env 5 `
  --strict
```

## Output Files

The benchmark writes:

- `hard_heldout_eval.json`
- `hard_failure_traces.json`
- `benchmark_summary.json`

Use `benchmark_summary.json` as the first file to inspect.

## Pass Gates

Baseline gates must pass every time:

| Group | Minimum learned success rate |
| --- | ---: |
| `easy_test` | 1.0 |
| `hard_value_heldout` | 1.0 |
| `hard_length_heldout` | 1.0 |
| `partial_hidden_smoke` | 1.0 |

Probe gates are diagnostic in normal mode and required in strict mode:

| Group | vNext target |
| --- | ---: |
| `hard_length5_probe` | 0.85 |
| `partial_hidden_hard` | 0.65 |
| `full_hidden_probe` | 0.85 |

Failure trace gate:

- Allowed diagnostic buckets:
  - `lock_length_schema_limit`
  - `information_collected_but_execution_failed`
- Any new bucket is a required investigation before schema/model changes.

## Current v4.1 Reading

Current v4 passes the baseline gates.

The v4.1 probes isolate remaining failures to LockWorld:

- `lock_length_schema_limit`: LockWorld length 5 is beyond the current fixed-slot comfort zone.
- `information_collected_but_execution_failed`: the agent reveals information, then fails to use the revealed slot reliably.

This means the next implementation target is schema/dataset/evaluator, not a larger model.

## v4.2 SlotFlat Experiment

The v4 flat feature schema remains the stable baseline. The v4.2 experiment adds a parallel SlotFlat schema:

- schema: `neurokernel-slot-transition-v1`
- max slots: `8`
- belief object: explicit known/unknown/current slot state
- input layout: env family, belief globals, slot-flat tensor, padded state, action one-hot

Use SlotFlat only for new experiments until strict benchmark passes. Do not overwrite v4 artifacts.

## SlotFlat Balanced Dataset Rule

`neurokernel-slot-transition-v1` remains the schema version for raw and balanced SlotFlat files.

Balanced files must set:

- `dataset_profile`: `neurokernel-slot-balanced-v1`
- sampling unit: `candidate_set_id`
- heldout/test rows: preserved unless explicitly dropped

This keeps model input compatibility stable while making train data less LockWorld-dominated.

## Evidence Rules

- A benchmark result is valid only if the command, model path, output directory, episode counts, and summary JSON are preserved.
- Do not compare runs with different episode counts as if they are identical.
- Do not promote a model if baseline gates regress.
- Do not treat probe success as AGI evidence; treat it as local generalization evidence inside MicroWorlds.

## v4.4 Execution Compatibility Gate

v4.4 changes the gate, not the model. The learned model still predicts candidate outcomes, but the gate now enforces execution compatibility:

- If the current required action is known, incompatible actions are hard-blocked in `hybrid` mode.
- If the current slot is unknown, reveal/inspect-style actions are preferred and execution is blocked until information is known.
- Terminal-only actions are blocked until the state is finishable.
- The model score is used only after compatibility filtering.

This fixed the observed failure where the model preferred `press(digit=1)` even though the visible LockWorld code required `press(digit=2)`.

Comparison evidence with `world_model_slot_v2.onnx`:

| Gate mode | Result |
| --- | --- |
| `model_only` | Failed baseline and probes; reproduced wrong execution and hidden-info failures. |
| `prior_only` | Passed normal benchmark and probes. |
| `hybrid` with enforced compatibility | Passed strict benchmark and probes. |

Strict v4.4 command:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-benchmark `
  --model D:\agi_seed\artifacts\world_model_slot_v2.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\slot_v2_v44_hybrid_enforced_strict `
  --episodes 10 `
  --trace-episodes 5 `
  --max-failures-per-env 5 `
  --gate-mode hybrid `
  --strict
```

Strict result:

- `baseline_passed`: true
- `probe_passed`: true
- `diagnostic_passed`: true
- `failure_count`: 0
- `easy_test`: 1.0
- `hard_value_heldout`: 1.0
- `hard_length_heldout`: 1.0
- `partial_hidden_smoke`: 1.0
- `partial_hidden_hard`: 1.0
- `full_hidden_probe`: 1.0
- `hard_length5_probe`: 1.0

Interpretation:

- The remaining v4.3 failure was mainly an execution-selection problem, not a larger neural architecture problem.
- `hybrid` is now the promotion candidate because it keeps the learned predictor in the loop while preventing known impossible actions from overriding the state contract.
- The next benchmark should increase diversity and episode counts before claiming stronger generalization.

## v4.5 Gate Ablation

Run gate ablation before changing the model. This decomposes success into:

- `model_only`: learned world model score only.
- `prior_only`: compatibility prior only.
- `hybrid`: model score after compatibility enforcement.

Command:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-gate-ablation `
  --model D:\agi_seed\artifacts\world_model_slot_v2.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\gate_ablation_v4_4 `
  --episodes 50 `
  --trace-episodes 10 `
  --max-failures-per-env 10 `
  --strict
```

Smoke result from `gate_ablation_v4_4_smoke`:

- `model_only`: failed
- `prior_only`: passed
- `hybrid`: passed
- `macro_success_rate.model_only`: 0.4639
- `macro_success_rate.prior_only`: 1.0
- `macro_success_rate.hybrid`: 1.0
- `hybrid_gain_over_model`: 0.5361
- `hybrid_gain_over_prior`: 0.0
- verdict: `prior_dominated`

Interpretation:

- v4.4 success is real, but current benchmark does not prove the learned world model adds decision value.
- Compatibility prior fixes model-only failures.
- The next benchmark must include model-needed cases where structurally compatible actions have different costs, outcomes, or delayed consequences.

## v4.6 Model-Needed Probe

v4.6 adds cases where compatibility alone is not enough:

- `lock_trap`: the correct `press(digit=...)` is structurally compatible but bad until `inspect` disarms the trap.
- `maze_hazard`: the target `move(door=...)` is compatible but bad until `inspect` clears the hazard.
- `tool_precondition`: the required tool action is compatible but bad until `search` prepares the precondition.

New gate mode:

- `hybrid_veto`: starts from the compatibility-gated hybrid score, then lets the model veto compatible-but-bad actions and choose a setup/reveal action.

Important reading:

- Existing `world_model_slot_v2.onnx` is not expected to solve v4.6, because it was not trained on these dynamics.
- A failing `hybrid_veto` smoke run is useful evidence that the probe is actually model-needed.
- Promotion requires retraining on the merged SlotFlat + model-needed dataset.

Generate model-needed counterfactual rows:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli collect-counterfactual-dataset `
  --split all `
  --episodes 0 `
  --agents heuristic `
  --no-rollouts `
  --curriculum-profile slot_model_needed_v1 `
  --curriculum-target-groups-per-family 4000 `
  --counterfactual-out D:\agi_seed\data\model_ready\counterfactual_model_needed_v1.jsonl `
  --features-out D:\agi_seed\data\model_ready\features_model_needed_flat_unused.jsonl
```

Re-export the existing SlotFlat dataset with the current fixed action vocabulary:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_slot_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_fixed_vocab.jsonl
```

Export model-needed SlotFlat features:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_model_needed_v1.jsonl `
  --out D:\agi_seed\data\model_ready\features_model_needed_slot_v1.jsonl
```

Merge the datasets:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli merge-slot-features `
  D:\agi_seed\data\model_ready\features_slot_v2_fixed_vocab.jsonl `
  D:\agi_seed\data\model_ready\features_model_needed_slot_v1.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v1.jsonl
```

Train and export:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v1.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v1.pt `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-world-model-onnx `
  --checkpoint D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v1.pt `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v1.onnx
```

Run v4.6 ablation:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-gate-ablation `
  --model D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v1.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\gate_ablation_v4_6_model_needed_v1 `
  --episodes 50 `
  --trace-episodes 10 `
  --max-failures-per-env 10 `
  --strict
```

Target signal:

- `model_needed_probe.hybrid_veto` should beat `prior_only`.
- `hybrid_veto_gain_over_prior` should be positive on `model_needed_probe`.
- Baseline groups must not regress.

## v4.7 Model-Needed Failure Debug

After v4.6 shows `hybrid_veto > prior_only`, use focused debugging before changing architecture.

Command:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli debug-model-needed-failures `
  --model D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v1.onnx `
  --out D:\agi_seed\artifacts\benchmark\gate_ablation_v4_6_model_needed_v1\model_needed_failure_debug_v4_7.json
```

Default focus envs:

- `lock.probe.trap.1`
- `maze.probe.hazard.green`
- `maze.probe.hazard.red`

The debug report records every candidate action with:

- predicted reward, progress delta, local success, information gain
- actual counterfactual reward, progress delta, success
- model score
- compatibility prior and guard components
- model veto
- final score
- chosen versus expected action
- failure buckets

Current v4.7 diagnosis:

| Env | Main diagnosis |
| --- | --- |
| `lock.probe.trap.1` | Model predicts the correct post-disarm press as bad, so `model_veto` blocks it. |
| `maze.probe.hazard.green` | Model predicts the correct post-inspect target move as bad, so `model_veto` blocks it. |
| `maze.probe.hazard.red` | Model overvalues a wrong door and undervalues the correct target door. |

Bucket counts:

- `transition_undertrained`: 3
- `veto_false_positive`: 3
- `veto_harm`: 3
- `veto_too_weak`: 3
- `veto_not_triggered`: 1

Interpretation:

- The probe design is working.
- The failure is not missing candidates or metadata.
- The core issue is transition undertraining around post-setup states:
  - after `inspect` disarms a LockTrap,
  - after `inspect` clears a MazeHazard.
- Do not add planner, memory, GNN, LLM, pairwise ranking, or bigger models yet.

Next target:

- Improve model-needed curriculum with more post-setup execution rows.
- Keep `ToolPrecondition` as a regression group.
- Re-train and aim for `model_needed_probe.hybrid_veto >= 0.75`.

## v4.8 Post-Setup Transition Repair

v4.7 showed that the first model-needed signal exists, but the model still undervalues the correct action after setup actions such as `inspect`. v4.8 repairs the dataset and diagnostics before changing model architecture.

Generate the repaired model-needed curriculum:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli collect-counterfactual-dataset `
  --split all `
  --episodes 0 `
  --agents heuristic `
  --no-rollouts `
  --curriculum-profile slot_model_needed_v2 `
  --curriculum-target-groups-per-family 6000 `
  --counterfactual-out D:\agi_seed\data\model_ready\counterfactual_model_needed_v2.jsonl `
  --features-out D:\agi_seed\data\model_ready\features_model_needed_v2_flat_unused.jsonl
```

Export, merge, and gate-check SlotFlat features:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_slot_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_fixed_vocab.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_model_needed_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_model_needed_slot_v2.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli merge-slot-features `
  D:\agi_seed\data\model_ready\features_slot_v2_fixed_vocab.jsonl `
  D:\agi_seed\data\model_ready\features_model_needed_slot_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v2.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli check-slot-dataset-gates `
  D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v2.jsonl
```

Train, export, and run the ablation:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v2.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v2.pt `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-world-model-onnx `
  --checkpoint D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v2.pt `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v2.onnx

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-gate-ablation `
  --model D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v2.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\gate_ablation_v4_8_model_needed_v2 `
  --episodes 50 `
  --trace-episodes 10 `
  --max-failures-per-env 10 `
  --strict
```

Success criteria:

- `check-slot-dataset-gates` passes on the merged dataset.
- The audit contains nonzero `post_setup_execution_rows`, `post_setup_positive_execution_rows`, and `post_setup_correct_action_positive_rows`.
- `model_needed_probe.hybrid_veto >= 0.75`.
- `hybrid_veto` should beat `prior_only` on `model_needed_probe`.
- `post_setup_correct_action_pred_good_rate` should improve versus v4.6.
- `post_setup_correct_action_veto_rate` should decrease versus v4.6.

## v4.9 Maze Post-Setup Target-Color Grounding

v4.8 improved the model-needed probe, but remaining failures are concentrated in `MazeHazard` after `inspect` clears the hazard. The model must learn that `move(target_color)` is good and wrong doors are bad after setup.

Before training, generate the v4.9 curriculum and run the Maze grounding audit:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli collect-counterfactual-dataset `
  --split all `
  --episodes 0 `
  --agents heuristic `
  --no-rollouts `
  --curriculum-profile slot_model_needed_v3 `
  --curriculum-target-groups-per-family 6000 `
  --counterfactual-out D:\agi_seed\data\model_ready\counterfactual_model_needed_v3.jsonl `
  --features-out D:\agi_seed\data\model_ready\features_model_needed_v3_flat_unused.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_model_needed_v3.jsonl `
  --out D:\agi_seed\data\model_ready\features_model_needed_slot_v3.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli audit-maze-grounding `
  D:\agi_seed\data\model_ready\features_model_needed_slot_v3.jsonl
```

The audit must pass with:

- zero action one-hot roundtrip failures
- `target_color_encoded_rows == maze_rows`
- `single_target_slot_rows == maze_rows`
- positive post-setup target moves for red, green, and blue

Re-export the base SlotFlat dataset with the v4.9 SlotFlat schema, merge, gate-check, and train:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_slot_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_schema_v2.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli merge-slot-features `
  D:\agi_seed\data\model_ready\features_slot_v2_schema_v2.jsonl `
  D:\agi_seed\data\model_ready\features_model_needed_slot_v3.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli check-slot-dataset-gates `
  D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_v2_model_needed_v3.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3.pt `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-world-model-onnx `
  --checkpoint D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3.pt `
  --out D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3.onnx

D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-gate-ablation `
  --model D:\agi_seed\artifacts\world_model_slot_v2_model_needed_v3.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\gate_ablation_v4_9_model_needed_v3 `
  --episodes 50 `
  --trace-episodes 10 `
  --max-failures-per-env 10 `
  --strict
```

Success criteria:

- `model_needed_probe.hybrid_veto >= 0.75`
- `model_needed_probe.hybrid_veto > prior_only`
- `maze_post_setup_target_move_veto_rate < 0.1`
- `maze_wrong_door_pred_good_rate < 0.1`
- `maze_target_color_grounding_accuracy` improves versus v4.8
- no lock/tool regression
