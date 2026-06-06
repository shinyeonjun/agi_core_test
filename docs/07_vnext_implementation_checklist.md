# vNext Implementation Checklist

This checklist is the next work order after the v4.1 benchmark.

## Phase 0 - Freeze And Verify

- [ ] Keep `v4_first_hard_success` as the rollback baseline.
- [ ] Run `pytest tests`.
- [ ] Run `run-benchmark` normal mode against `world_model_v4.onnx`.
- [ ] Save `benchmark_summary.json`.
- [ ] Confirm baseline gates remain 1.0.
- [ ] Confirm any probe failures are classified into known buckets.

## Phase 1 - Failure Reporter

- [x] Separate `lock_length_schema_limit`.
- [x] Separate `information_collected_but_execution_failed`.
- [x] Avoid labeling successful inspect/reveal as `acted_before_information`.
- [x] Make classifier tolerate compact traces with missing score fields.
- [ ] Add a compact human report command if manual JSON reading becomes slow.

## Phase 2 - State And Schema Design

- [x] Decide schema direction:
  - SlotFlat v1 with shared slot encoding, while keeping flat v4 as the rollback baseline.
- [x] Add `BeliefState` as the explicit known/unknown/current-slot state object.
- [x] Add `neurokernel-slot-transition-v1` as a parallel experimental schema.
- [x] Represent known slots without leaking hidden answers.
- [x] Represent current slot status:
  - `current_index`
  - `sequence_length`
  - `current_slot_known`
  - `known_mask`
  - `unknown_mask`
- [x] Represent observation memory:
  - `last_observation_type`
  - `last_observation_slot`
  - `last_observation_value`
  - `information_gain`
- [x] Define how unknown values are encoded.
- [ ] Define how `can_finish` and `can_summarize` are computed without hidden leakage.
- [x] Version the schema as vNext and do not overwrite v4 manifests.

## Phase 3 - Dataset Curriculum

- [x] Add `slot_v2` curriculum generator profile.
- [x] Generate visible/partial/hidden LockWorld curriculum examples.
- [x] Generate Maze visible/partial/hidden curriculum examples.
- [x] Generate Tool visible/partial/hidden curriculum examples.
- [x] Generate information-gain and post-reveal candidate groups.
- [x] Add pre-training SlotFlat audit gate.
- [ ] Generate visible LockWorld length 5 examples at full scale.
- [ ] Generate partial-hidden LockWorld length 4 examples at full scale.
- [ ] Generate full-hidden LockWorld length 4 examples at full scale.
- [ ] Include failure replay examples from v4.1 traces.
- [x] Add SlotFlat dataset audit for family, visibility, source, information-gain, post-reveal, progress, and length skew.
- [x] Add candidate-group balanced SlotFlat exporter.
- [ ] Train on `neurokernel-slot-balanced-v1`.
- [ ] Balance success/failure rows so the model does not learn only easy visible tasks.
- [ ] Validate feature manifest dimensions and action vocabulary.
- [ ] Keep train/test/heldout/probe split labels explicit.

## Phase 4 - Gate And Evaluation

- [x] Keep existing Action Gate behavior stable for v4 baseline groups.
- [x] Add or confirm visibility guard:
  - prefer reveal action when current slot is unknown
  - penalize execution action when current slot is unknown
  - penalize terminal-only action before final step
- [x] Add hard execution compatibility enforcement for `hybrid` gate mode.
- [x] Add `model_only`, `prior_only`, and `hybrid` gate modes.
- [x] Add candidate-level compatibility traces:
  - required action key
  - model-only score
  - prior-only score
  - hybrid score
- [x] Add compatibility metrics:
  - `compatibility_override_rate`
  - `compatibility_fix_rate`
  - `compatibility_harm_rate`
  - `required_action_match_rate`
  - `unknown_reveal_choice_rate`
- [ ] Add separate count metrics if JSON reading becomes slow:
  - `repeated_action_count`
  - `schema_limit_failure_count`
  - `information_collected_but_execution_failed_count`
- [x] Ensure `run-benchmark --strict` fails if vNext targets are missed.
- [x] Pass v4.4 strict benchmark with `world_model_slot_v2.onnx` and enforced `hybrid` gate.

## Phase 5 - Train And Promote

- [x] Train vNext on laptop/CUDA.
- [x] Export ONNX.
- [x] Run ONNX verification locally.
- [x] Run `run-benchmark --strict`.
- [ ] Deploy to Orange Pi.
- [ ] Run Orange Pi runtime smoke eval.
- [ ] Freeze a new baseline only if:
  - baseline groups do not regress,
  - strict probe gates pass,
  - trace buckets have no unknown failures,
  - ONNX runtime works on Orange Pi.

## Do Not Do Yet

- [ ] GNN
- [ ] Dreamer
- [ ] MPC/RL
- [ ] RKNN/NPU optimization
- [ ] LLM integration
- [ ] pairwise ranking loss, unless the next larger benchmark shows the model ranking itself is the bottleneck
- [ ] action score head, unless compatibility-gated hybrid stops improving

These are blocked until schema/dataset/evaluator failures are cleared.

## v4.4 Current Status

`slot_v2_v44_hybrid_enforced_strict` is the current strongest local baseline.

Artifacts:

- model: `D:\agi_seed\artifacts\world_model_slot_v2.onnx`
- summary: `D:\agi_seed\artifacts\benchmark\slot_v2_v44_hybrid_enforced_strict\benchmark_summary.json`
- hard eval: `D:\agi_seed\artifacts\benchmark\slot_v2_v44_hybrid_enforced_strict\hard_heldout_eval.json`
- failure traces: `D:\agi_seed\artifacts\benchmark\slot_v2_v44_hybrid_enforced_strict\hard_failure_traces.json`

Result:

- baseline gates: passed
- probe gates: passed
- diagnostic gate: passed
- failure count: 0

Next required work:

- [x] Re-run v4.4 with larger episode counts before freezing as a stronger baseline.
- [ ] Add a robustness benchmark with more heldout seeds and more varied Lock/Maze/Tool configurations.
- [x] Compare `hybrid` against `prior_only` on the current v4.4 benchmark.
- [ ] Compare `hybrid` against `prior_only` on harder cases where model prediction should matter.
- [x] Add an explicit report that flags when `prior_only` equals or beats `hybrid`, because that means the model is not adding decision value yet.

## v4.5 Gate Ablation Status

Implemented:

- [x] `run-gate-ablation` CLI command.
- [x] `gate_ablation_v4_4.json` report.
- [x] `model_only`, `prior_only`, and `hybrid` comparison in one run.
- [x] Group-level gains:
  - `hybrid_gain_over_model`
  - `hybrid_gain_over_prior`
  - `prior_gain_over_model`
- [x] Report verdict:
  - `prior_dominated`
  - `hybrid_adds_value`
  - `compatibility_fix`
  - `unresolved_or_regressed`

Current smoke result:

- `model_only` failed.
- `prior_only` passed.
- `hybrid` passed.
- `hybrid_gain_over_prior` was `0.0`.
- verdict: `prior_dominated`.

Meaning:

- v4.4 is a good execution core baseline.
- Current benchmark still does not prove that the learned model adds unique decision value.
- The next implementation target is `model_needed_probe`, not a bigger model.

## v4.6 Model-Needed Probe Requirements

Do this before changing model architecture.

- [x] Add probe cases where `prior_only` cannot trivially solve the task.
- [x] Keep actions inside the current ONNX action vocabulary unless retraining is planned.
- [x] Prefer existing action names:
  - `inspect`
  - `press(digit=1..4)`
  - `move(door="red|blue|green")`
  - `search`
  - `read`
  - `summarize`
- [x] Create cases where structurally compatible actions differ by:
  - cost
  - delayed consequence
  - hidden trap/precondition
  - setup/reveal requirement
- [x] Add `hybrid_veto` gate mode.
- [x] Add `model_veto` candidate traces and diagnostic metrics.
- [x] Add `slot_model_needed_v1` curriculum profile.
- [x] Add `merge-slot-features` so model-needed rows can be appended without hand-editing JSONL.
- [x] Fix SlotFlat action vocab so exports remain merge-compatible across curriculum files.
- [x] A useful model-needed group should show:
  - `prior_only < hybrid`, or
  - `prior_only` failure with clear evidence that the current model/predictor is not enough yet.

Current smoke result with the old `world_model_slot_v2.onnx`:

- `model_needed_probe.prior_only`: failed on trap/precondition cases.
- `model_needed_probe.hybrid_veto`: also failed because the old model was never trained on model-needed dynamics.
- Meaning: the probe is wired; the next action is data generation and retraining, not architecture escalation.

Next required work:

- [x] Generate full `slot_model_needed_v1` counterfactual dataset.
- [x] Re-export `features_slot_v2` with the current fixed action vocabulary.
- [x] Merge `features_slot_v2_fixed_vocab` + `features_model_needed_slot_v1`.
- [x] Train `world_model_slot_v2_model_needed_v1.pt`.
- [x] Export `world_model_slot_v2_model_needed_v1.onnx`.
- [x] Run `run-gate-ablation` and confirm `hybrid_veto` beats `prior_only` on `model_needed_probe`.

## v4.7 Model-Needed Failure Debug Status

Implemented:

- [x] Candidate-level score breakdown:
  - `model_score`
  - `compatibility_prior`
  - `visibility_guard`
  - `finish_guard`
  - `reset_guard`
  - `model_veto`
  - `final_score`
- [x] `debug-model-needed-failures` CLI command.
- [x] Focused debug output for:
  - `lock.probe.trap.1`
  - `maze.probe.hazard.green`
  - `maze.probe.hazard.red`
- [x] Model-needed buckets:
  - `state_feature_missing`
  - `transition_undertrained`
  - `veto_not_triggered`
  - `veto_too_weak`
  - `veto_harm`
  - `veto_false_positive`
  - `candidate_missing`
  - `metadata_error`
  - `probe_design_error`
- [x] Freeze `v4.6_first_model_needed_signal` baseline.

Current diagnosis:

- `lock.probe.trap.1`: post-disarm correct press is predicted as bad, then vetoed.
- `maze.probe.hazard.green`: post-inspect correct target move is predicted as bad, then vetoed.
- `maze.probe.hazard.red`: wrong door is overvalued while correct target door is undervalued and vetoed.

Next required work:

- [ ] Add more post-setup execution rows for LockTrap:
  - trap armed -> inspect/disarm -> correct press good.
  - trap disarmed -> repeated inspect low value.
- [ ] Add more post-setup execution rows for MazeHazard:
  - hazard active -> inspect/clear -> target move good.
  - hazard cleared -> repeated inspect low value.
  - wrong doors remain bad after clear.
- [ ] Re-train without changing architecture.
- [ ] Target `model_needed_probe.hybrid_veto >= 0.75`.

## v4.2 SlotFlat Commands

Generate counterfactual rows with longer LockWorld coverage:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli collect-counterfactual-dataset `
  --split all `
  --episodes 20 `
  --agents random heuristic gated `
  --lock-code-lengths 3 4 5 6 `
  --counterfactual-out D:\agi_seed\data\model_ready\counterfactual_slot_v1.jsonl `
  --features-out D:\agi_seed\data\model_ready\features_flat_v4_from_slot_run.jsonl
```

Export SlotFlat features:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_slot_v1.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v1.jsonl
```

Validate SlotFlat features:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli validate-slot-features `
  D:\agi_seed\data\model_ready\features_slot_v1.jsonl
```

Train a SlotFlat world model:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_v1.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_v1.pt `
  --epochs 300 `
  --device cuda
```

Export ONNX:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-world-model-onnx `
  --checkpoint D:\agi_seed\artifacts\world_model_slot_v1.pt `
  --out D:\agi_seed\artifacts\world_model_slot_v1.onnx
```

Run strict benchmark:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli run-benchmark `
  --model D:\agi_seed\artifacts\world_model_slot_v1.onnx `
  --out-dir D:\agi_seed\artifacts\benchmark\slot_v1_strict `
  --episodes 10 `
  --trace-episodes 5 `
  --max-failures-per-env 5 `
  --strict
```

## v4.2 Balanced SlotFlat Commands

Audit the raw SlotFlat dataset:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli audit-slot-dataset `
  D:\agi_seed\data\model_ready\features_slot_v1.jsonl
```

Export a train-balanced SlotFlat dataset. This keeps the schema version as `neurokernel-slot-transition-v1` and marks the manifest with `dataset_profile=neurokernel-slot-balanced-v1`.

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-balanced-slot-dataset `
  D:\agi_seed\data\model_ready\features_slot_v1.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_balanced_v1.jsonl `
  --target-rows 120000 `
  --seed 7
```

Validate the balanced dataset:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli validate-slot-features `
  D:\agi_seed\data\model_ready\features_slot_balanced_v1.jsonl
```

Train the balanced SlotFlat model:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_balanced_v1.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_balanced_v1.pt `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10
```

## v4.3 Slot Curriculum v2 Commands

Generate the new curriculum counterfactual dataset. Start with `4000`; increase to `8000` after the first full benchmark if the data gates pass but benchmark is still noisy.

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli collect-counterfactual-dataset `
  --split all `
  --episodes 0 `
  --agents heuristic `
  --no-rollouts `
  --curriculum-profile slot_v2 `
  --curriculum-target-groups-per-family 4000 `
  --counterfactual-out D:\agi_seed\data\model_ready\counterfactual_slot_v2.jsonl `
  --features-out D:\agi_seed\data\model_ready\features_slot_v2_flat_unused.jsonl
```

Export SlotFlat features:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli export-slot-features `
  D:\agi_seed\data\model_ready\counterfactual_slot_v2.jsonl `
  --out D:\agi_seed\data\model_ready\features_slot_v2.jsonl
```

Run the dataset gate before training:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli check-slot-dataset-gates `
  D:\agi_seed\data\model_ready\features_slot_v2.jsonl
```

Train only if the gate passes:

```powershell
D:\agi_seed\neurokernel-agi-seed\venv\Scripts\python.exe -m neurokernel_seed.cli train-world-model `
  --features D:\agi_seed\data\model_ready\features_slot_v2.jsonl `
  --out D:\agi_seed\artifacts\world_model_slot_v2.pt `
  --epochs 50 `
  --batch-size 1024 `
  --device cuda `
  --patience 10
```

## v4.8 Post-Setup Repair Checklist

- [x] Add setup metadata to model-needed worlds:
  - `setup_state`
  - `setup_known`
  - `setup_effective`
  - `post_setup_state`
- [x] Add `slot_model_needed_v2` curriculum profile.
- [x] Generate post-setup execution snapshots after setup has already succeeded.
- [x] Persist row-level post-setup labels into SlotFlat samples.
- [x] Audit post-setup coverage:
  - `post_setup_execution_rows`
  - `post_setup_positive_execution_rows`
  - `post_setup_correct_action_rows`
  - `post_setup_correct_action_positive_rows`
- [x] Add dataset gates for post-setup candidate groups.
- [x] Add benchmark diagnostics:
  - `post_setup_correct_action_pred_good_rate`
  - `post_setup_correct_action_veto_rate`
  - `veto_false_positive_rate`
  - `veto_true_positive_rate`
- [x] Add regression test for the repaired curriculum.
- [ ] User runs full v4.8 dataset generation.
- [ ] User runs merged dataset gate check.
- [ ] User trains `world_model_slot_v2_model_needed_v2.pt`.
- [ ] User exports `world_model_slot_v2_model_needed_v2.onnx`.
- [ ] User runs v4.8 gate ablation.
- [ ] User runs v4.8 model-needed failure debug.

Do not start v4.9 until the v4.8 ablation answers these questions:

- Did `model_needed_probe.hybrid_veto` improve over v4.6?
- Did `post_setup_correct_action_pred_good_rate` go up?
- Did `post_setup_correct_action_veto_rate` go down?
- Did `prior_only` stay below or equal to the model-aware modes on the model-needed probe?
- Did baseline groups avoid regression?

## v4.9 Maze Grounding Checklist

- [x] Freeze v4.8 baseline as `v4.8_maze_grounding_pre_repair`.
- [x] Bump SlotFlat schema to `neurokernel-slot-transition-v2`.
- [x] Add explicit Maze global features:
  - target color one-hot
  - hazard present/active/cleared
  - setup state
  - post-setup state
  - can move target
- [x] Add Maze door-slot features:
  - `maze.is_target_slot`
  - `maze.hazard_known`
  - `maze.hazard_active`
  - `maze.hazard_cleared`
  - `maze.blocked`
  - `maze.inspected`
- [x] Fix Maze slot encoding to keep all door slots, not only `sequence_length=1`.
- [x] Add `audit-maze-grounding` CLI.
- [x] Add action one-hot roundtrip and target-slot audit.
- [x] Add `slot_model_needed_v3` curriculum profile.
- [x] Add `safe_no_hazard` support to `MazeHazardWorld`.
- [x] Add state-aware post-setup model veto.
- [x] Add Maze-specific benchmark metrics:
  - `maze_post_setup_target_move_veto_rate`
  - `maze_wrong_door_pred_good_rate`
  - `maze_target_color_grounding_accuracy`
  - `maze_hazard_clear_to_execute_success_rate`
- [x] Add regression tests for v4.9 audit and state-aware veto.
- [ ] User generates full v4.9 dataset.
- [ ] User runs `audit-maze-grounding`.
- [ ] User merges with re-exported base SlotFlat schema v2 features.
- [ ] User trains `world_model_slot_v2_model_needed_v3.pt`.
- [ ] User exports `world_model_slot_v2_model_needed_v3.onnx`.
- [ ] User runs v4.9 gate ablation.
- [ ] User compares v4.9 against v4.8 baseline.
