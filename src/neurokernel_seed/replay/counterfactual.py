from __future__ import annotations

import hashlib
import itertools
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from neurokernel_seed.agents.baselines import HeuristicAgent, RandomAgent
from neurokernel_seed.agents.gated import GatedAgent
from neurokernel_seed.core.schema import Action, WorldState
from neurokernel_seed.envs.base import MicroWorld
from neurokernel_seed.envs.lock_world import LockWorld
from neurokernel_seed.envs.memory_maze import MemoryMaze
from neurokernel_seed.envs.model_needed import LockTrapWorld, MazeHazardWorld, ToolPreconditionWorld
from neurokernel_seed.envs.registry import list_envs, make_env
from neurokernel_seed.envs.tool_world import ToolWorld
from neurokernel_seed.predictors.symbolic import SymbolicPredictor
from neurokernel_seed.replay.dataset import canonical_action_key

COUNTERFACTUAL_SCHEMA_VERSION = "neurokernel-counterfactual-transition-v1"
DEFAULT_LOCK_CODE_LENGTHS = (3, 4)
LOCK_DIGITS = (1, 2, 3, 4)
MAZE_TARGET_COLORS = ("red", "blue", "green")
MAZE_MAX_STEPS = (5, 7, 8, 10)
TOOL_ACTIONS = ("search", "read", "summarize")
CURRICULUM_PROFILES = ("legacy", "slot_v2", "slot_model_needed_v1", "slot_model_needed_v2", "slot_model_needed_v3")
CURRICULUM_VISIBILITY_MODES = ("visible", "partial_hidden", "hidden")
TERMINAL_VALID_TOOL_SEQUENCES = (
    ("search", "read", "summarize"),
    ("read", "search", "summarize"),
    ("search", "search", "summarize"),
    ("read", "read", "summarize"),
    ("search", "read", "search", "summarize"),
    ("read", "search", "read", "summarize"),
)


@dataclass(frozen=True)
class ReachableSnapshot:
    env_name: str
    split: str
    source_agent: str
    seed: int
    step_index: int
    restore_state: dict[str, Any]
    state: WorldState

    @property
    def restore_id(self) -> str:
        return _hash_json({"env_name": self.env_name, "restore_state": self.restore_state})

    @property
    def agent_state_id(self) -> str:
        return _hash_json({"env_name": self.env_name, "state_id": self.state.state_id, "state_vector": self.state.vector, "state_facts": self.state.facts})


def export_candidate_counterfactuals(
    out_path: str | Path,
    *,
    split: str = "all",
    episodes: int = 20,
    agents: list[str] | None = None,
    include_rollouts: bool = True,
    include_variants: bool = True,
    lock_code_lengths: Sequence[int] | None = None,
    curriculum_profile: str = "legacy",
    curriculum_target_groups_per_family: int = 4_000,
) -> dict[str, Any]:
    if curriculum_profile not in CURRICULUM_PROFILES:
        raise ValueError(f"unknown curriculum profile: {curriculum_profile}")
    agents = agents or ["random", "heuristic", "gated"]
    snapshots: list[ReachableSnapshot] = []
    if include_rollouts:
        snapshots.extend(collect_reachable_snapshots(split=split, episodes=episodes, agents=agents))
    if include_variants:
        snapshots.extend(
            collect_variant_snapshots(
                split=split,
                lock_code_lengths=lock_code_lengths or DEFAULT_LOCK_CODE_LENGTHS,
                curriculum_profile=curriculum_profile,
                curriculum_target_groups_per_family=curriculum_target_groups_per_family,
            )
        )
    snapshots = _dedupe_snapshots(snapshots)

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        rows.extend(_evaluate_snapshot_candidates(snapshot))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    envs = sorted({row["env_name"] for row in rows})
    source_counts: dict[str, int] = {}
    for snapshot in snapshots:
        source_counts[snapshot.source_agent] = source_counts.get(snapshot.source_agent, 0) + 1
    meta = {
        "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
        "rows": len(rows),
        "unique_states": len(snapshots),
        "envs": envs,
        "splits": sorted({row["split"] for row in rows if row.get("split")}),
        "source_state_counts": dict(sorted(source_counts.items())),
        "include_rollouts": include_rollouts,
        "include_variants": include_variants,
        "lock_code_lengths": list(lock_code_lengths or DEFAULT_LOCK_CODE_LENGTHS),
        "curriculum_profile": curriculum_profile,
        "curriculum_target_groups_per_family": curriculum_target_groups_per_family,
        "candidate_count_by_env": _candidate_count_by_env(rows),
        "out": str(out),
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def collect_reachable_snapshots(*, split: str = "all", episodes: int = 20, agents: list[str] | None = None) -> list[ReachableSnapshot]:
    agents = agents or ["random", "heuristic", "gated"]
    snapshots_by_restore_id: dict[str, ReachableSnapshot] = {}
    for env_name in list_envs(split):
        for agent_name in agents:
            for seed in range(episodes):
                env = make_env(env_name)
                agent = _make_agent(agent_name, seed)
                state = env.reset(seed)
                step_index = 0
                while not state.terminal and step_index < env.max_steps:
                    snapshot = ReachableSnapshot(env.name, env.split, agent_name, seed, step_index, env.snapshot(), state)
                    snapshots_by_restore_id.setdefault(snapshot.restore_id, snapshot)
                    action = agent.act(env, state)
                    result = env.step(action)
                    state = result.next_state
                    step_index += 1
    return sorted(snapshots_by_restore_id.values(), key=lambda item: (item.env_name, item.restore_id))


def collect_variant_snapshots(
    *,
    split: str = "all",
    lock_code_lengths: Sequence[int] = DEFAULT_LOCK_CODE_LENGTHS,
    curriculum_profile: str = "legacy",
    curriculum_target_groups_per_family: int = 4_000,
) -> list[ReachableSnapshot]:
    if curriculum_profile == "slot_v2":
        return collect_slot_curriculum_v2_snapshots(split=split, target_groups_per_family=curriculum_target_groups_per_family)
    if curriculum_profile in {"slot_model_needed_v1", "slot_model_needed_v2"}:
        return collect_slot_model_needed_v1_snapshots(split=split, target_groups_per_family=curriculum_target_groups_per_family)
    if curriculum_profile == "slot_model_needed_v3":
        return collect_slot_model_needed_v3_snapshots(split=split, target_groups_per_family=curriculum_target_groups_per_family)
    snapshots: list[ReachableSnapshot] = []
    snapshots.extend(_collect_lock_canonical_snapshots(split, lock_code_lengths))
    snapshots.extend(_collect_maze_bfs_snapshots(split))
    snapshots.extend(_collect_tool_variant_snapshots(split))
    snapshots.extend(_collect_partial_hidden_snapshots(split))
    return _dedupe_snapshots(snapshots)


def collect_slot_curriculum_v2_snapshots(*, split: str = "all", target_groups_per_family: int = 4_000) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    snapshots.extend(_collect_lock_curriculum_v2_snapshots(split, target_groups_per_family))
    snapshots.extend(_collect_maze_curriculum_v2_snapshots(split, target_groups_per_family))
    snapshots.extend(_collect_tool_curriculum_v2_snapshots(split, target_groups_per_family))
    return _dedupe_snapshots(snapshots)


def collect_slot_model_needed_v1_snapshots(*, split: str = "all", target_groups_per_family: int = 1_000) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    snapshots.extend(_collect_lock_trap_snapshots(split, target_groups_per_family))
    snapshots.extend(_collect_maze_hazard_snapshots(split, target_groups_per_family))
    snapshots.extend(_collect_tool_precondition_snapshots(split, target_groups_per_family))
    return _dedupe_snapshots(snapshots)


def collect_slot_model_needed_v3_snapshots(*, split: str = "all", target_groups_per_family: int = 1_000) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    snapshots.extend(_collect_lock_trap_snapshots(split, target_groups_per_family))
    snapshots.extend(_collect_maze_hazard_v3_snapshots(split, target_groups_per_family * 2))
    snapshots.extend(_collect_tool_precondition_snapshots(split, target_groups_per_family))
    return _dedupe_snapshots(snapshots)


def validate_counterfactual(path: str | Path) -> dict[str, Any]:
    rows = 0
    states: set[str] = set()
    envs: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _validate_counterfactual_row(row, line_no)
            rows += 1
            states.add(str(row["restore_id"]))
            envs.add(str(row["env_name"]))
    return {"schema_version": COUNTERFACTUAL_SCHEMA_VERSION, "rows": rows, "unique_states": len(states), "envs": sorted(envs)}


def _collect_lock_canonical_snapshots(split: str, lock_code_lengths: Sequence[int]) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    for length in lock_code_lengths:
        for code_index, code in enumerate(itertools.product(LOCK_DIGITS, repeat=int(length))):
            item_split = _variant_split(code_index)
            if not _include_split(item_split, split):
                continue
            max_steps = max(6, len(code) + 3)
            env_name = f"lock.{item_split}.canon.{len(code)}.{code_index:04d}"
            env = LockWorld(env_name, item_split, tuple(int(value) for value in code), max_steps)
            for progress in range(len(code)):
                for steps in range(progress, max_steps):
                    env._state = env._make_state(progress, steps, False)
                    state = env._state
                    snapshots.append(ReachableSnapshot(env.name, env.split, "lock-canonical", _variant_seed(env.name), steps, env.snapshot(), state))
    return snapshots


def _collect_maze_bfs_snapshots(split: str) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    for item_split in ("train", "test"):
        if not _include_split(item_split, split):
            continue
        for target_color in MAZE_TARGET_COLORS:
            for max_steps in MAZE_MAX_STEPS:
                env_name = f"maze.{item_split}.bfs.{target_color}.{max_steps}"
                env = MemoryMaze(env_name, item_split, target_color, max_steps)
                snapshots.extend(_bfs_snapshots(env, "maze-bfs", max_depth=max_steps - 1))
    return snapshots


def _collect_tool_variant_snapshots(split: str) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    for item_split in ("train", "test"):
        if not _include_split(item_split, split):
            continue
        for seq_index, sequence in enumerate(TERMINAL_VALID_TOOL_SEQUENCES):
            max_steps = len(sequence) + 3
            env_name = f"tool.{item_split}.seq.{seq_index:02d}"
            env = ToolWorld(env_name, item_split, max_steps, tuple(sequence))
            for stage in range(len(sequence)):
                for steps in range(stage, max_steps):
                    env._state = env._make_state(stage, steps, False)
                    state = env._state
                    snapshots.append(ReachableSnapshot(env.name, env.split, "tool-variant", _variant_seed(env.name), steps, env.snapshot(), state))
    return snapshots


def _collect_lock_curriculum_v2_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    code_iter = _bounded_lock_codes((3, 4, 5), max(32, target_groups // 3))
    for code_index, code in enumerate(code_iter):
        item_split = _variant_split(code_index)
        if not _include_split(item_split, split):
            continue
        max_steps = max(8, len(code) + 5)
        for visibility in CURRICULUM_VISIBILITY_MODES:
            if len([snap for snap in snapshots if snap.env_name.startswith("lock.")]) >= target_groups:
                return snapshots
            env_name = f"lock.{item_split}.slot_v2.{visibility}.{len(code)}.{code_index:05d}"
            env = LockWorld(env_name, item_split, tuple(int(value) for value in code), max_steps, visibility_mode=visibility)
            for progress in range(len(code)):
                steps = min(max_steps - 1, progress)
                if visibility == "visible":
                    env.reset(0)
                    env._state = env._make_state(progress, steps, False)
                    snapshots.append(ReachableSnapshot(env.name, env.split, "slot-v2-lock-visible", _variant_seed(env.name), steps, env.snapshot(), env._state))
                    continue
                env.reset(0)
                env._known_mask = _known_mask_for_visibility(len(code), visibility)
                env._last_observation = {"type": "none", "slot": -1, "value": 0}
                env._state = env._make_state(progress, steps, False)
                snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-lock-{visibility}-pre", _variant_seed(env.name), steps, env.snapshot(), env._state))
                env.reset(0)
                env._known_mask = _known_mask_for_visibility(len(code), visibility)
                env._known_mask[progress] = True
                env._last_observation = {"type": "inspect", "slot": progress, "value": code[progress]}
                env._state = env._make_state(progress, min(max_steps - 1, steps + 1), False)
                snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-lock-{visibility}-post", _variant_seed(env.name), steps + 1, env.snapshot(), env._state))
    return snapshots[:target_groups]


def _collect_maze_curriculum_v2_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    door_permutations = list(itertools.permutations(MemoryMaze.doors))
    variant_index = 0
    while len(snapshots) < target_groups:
        target_color = MAZE_TARGET_COLORS[variant_index % len(MAZE_TARGET_COLORS)]
        doors = door_permutations[variant_index % len(door_permutations)]
        max_steps = MAZE_MAX_STEPS[variant_index % len(MAZE_MAX_STEPS)]
        item_split = _variant_split(variant_index)
        if _include_split(item_split, split):
            for visibility in CURRICULUM_VISIBILITY_MODES:
                env_name = f"maze.{item_split}.slot_v2.{visibility}.{target_color}.{variant_index:05d}"
                env = MemoryMaze(env_name, item_split, target_color, max_steps, doors, visibility_mode=visibility)
                if visibility == "visible":
                    state = env.reset(0)
                    snapshots.append(ReachableSnapshot(env.name, env.split, "slot-v2-maze-visible", _variant_seed(env.name), 0, env.snapshot(), state))
                    env._state = env._make_state(True, False, min(2, max_steps - 1), False)
                    snapshots.append(ReachableSnapshot(env.name, env.split, "slot-v2-maze-zero-info", _variant_seed(env.name), 2, env.snapshot(), env._state))
                else:
                    state = env.reset(0)
                    snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-maze-{visibility}-pre", _variant_seed(env.name), 0, env.snapshot(), state))
                    env._state = env._make_state(True, False, 1, False)
                    snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-maze-{visibility}-post", _variant_seed(env.name), 1, env.snapshot(), env._state))
                if len(snapshots) >= target_groups:
                    break
        variant_index += 1
    return snapshots[:target_groups]


def _collect_tool_curriculum_v2_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    sequences = _tool_curriculum_sequences(max(16, target_groups // 4))
    for seq_index, sequence in enumerate(sequences):
        item_split = _variant_split(seq_index)
        if not _include_split(item_split, split):
            continue
        max_steps = len(sequence) + 5
        for visibility in CURRICULUM_VISIBILITY_MODES:
            if len(snapshots) >= target_groups:
                return snapshots
            env_name = f"tool.{item_split}.slot_v2.{visibility}.{len(sequence)}.{seq_index:05d}"
            env = ToolWorld(env_name, item_split, max_steps, sequence, visibility_mode=visibility)
            for stage in range(len(sequence)):
                steps = min(max_steps - 1, stage)
                if visibility == "visible":
                    env.reset(0)
                    env._state = env._make_state(stage, steps, False)
                    snapshots.append(ReachableSnapshot(env.name, env.split, "slot-v2-tool-visible", _variant_seed(env.name), steps, env.snapshot(), env._state))
                    continue
                env.reset(0)
                env._known_mask = _known_mask_for_visibility(len(sequence), visibility)
                env._last_observation = {"type": "none", "slot": -1, "value": ""}
                env._state = env._make_state(stage, steps, False)
                snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-tool-{visibility}-pre", _variant_seed(env.name), steps, env.snapshot(), env._state))
                env.reset(0)
                env._known_mask = _known_mask_for_visibility(len(sequence), visibility)
                env._known_mask[stage] = True
                reveal_action = sequence[stage] if sequence[stage] in {"search", "read"} else "read"
                env._last_observation = {"type": reveal_action, "slot": stage, "value": sequence[stage]}
                env._state = env._make_state(stage, min(max_steps - 1, steps + 1), False)
                snapshots.append(ReachableSnapshot(env.name, env.split, f"slot-v2-tool-{visibility}-post", _variant_seed(env.name), steps + 1, env.snapshot(), env._state))
                if len(snapshots) >= target_groups:
                    return snapshots
    return snapshots[:target_groups]


def _collect_lock_trap_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    codes = _bounded_lock_codes((3, 4), max(16, target_groups // 2))
    for code_index, code in enumerate(codes):
        item_split = _variant_split(code_index)
        if not _include_split(item_split, split):
            continue
        for trap_armed in (True, False):
            env_name = f"lock.{item_split}.model_needed.trap.{int(trap_armed)}.{code_index:05d}"
            env = LockTrapWorld(env_name, item_split, tuple(code), max(8, len(code) + 5), trap_armed)
            for progress in range(len(code)):
                step_values = [progress]
                if not trap_armed:
                    step_values = sorted({progress, min(env.max_steps - 1, progress + 1), min(env.max_steps - 1, progress + 2), min(env.max_steps - 1, progress + 4)})
                for steps in step_values:
                    if len(snapshots) >= target_groups:
                        return snapshots
                    env.trap_armed = trap_armed
                    env._state = env._make_state(progress, steps, False)
                    source = "slot-model-needed-lock-trap-pre"
                    if not trap_armed:
                        source = "slot-model-needed-lock-trap-post-setup"
                    snapshots.append(ReachableSnapshot(env.name, env.split, source, _variant_seed(env.name), steps, env.snapshot(), env._state))
    return snapshots[:target_groups]


def _collect_maze_hazard_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    door_permutations = list(itertools.permutations(MemoryMaze.doors))
    index = 0
    while len(snapshots) < target_groups:
        item_split = _variant_split(index)
        if _include_split(item_split, split):
            target_color = MAZE_TARGET_COLORS[index % len(MAZE_TARGET_COLORS)]
            doors = door_permutations[index % len(door_permutations)]
            for hazard_active in (True, False):
                env_name = f"maze.{item_split}.model_needed.hazard.{int(hazard_active)}.{target_color}.{index:05d}"
                env = MazeHazardWorld(env_name, item_split, target_color, 7, doors, hazard_active)
                step_values = [0] if hazard_active else [1, 2, 3, 4, 5]
                for steps in step_values:
                    env.hazard_active = hazard_active
                    env._state = env._make_state(True, False, min(steps, env.max_steps - 1), False)
                    source = "slot-model-needed-maze-hazard-pre" if hazard_active else "slot-model-needed-maze-hazard-post-setup"
                    snapshots.append(ReachableSnapshot(env.name, env.split, source, _variant_seed(env.name), steps, env.snapshot(), env._state))
                    if len(snapshots) >= target_groups:
                        break
                if len(snapshots) >= target_groups:
                    break
        index += 1
    return snapshots[:target_groups]


def _collect_maze_hazard_v3_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    door_permutations = list(itertools.permutations(MemoryMaze.doors))
    scenarios = (
        "hazard_unknown",
        "hazard_active_known",
        "post_setup_cleared",
        "post_setup_cleared",
        "post_setup_cleared",
        "safe_no_hazard",
    )
    index = 0
    while len(snapshots) < target_groups:
        item_split = _variant_split(index)
        if _include_split(item_split, split):
            target_color = MAZE_TARGET_COLORS[index % len(MAZE_TARGET_COLORS)]
            doors = door_permutations[index % len(door_permutations)]
            scenario = scenarios[index % len(scenarios)]
            if scenario == "hazard_unknown":
                env = MazeHazardWorld(f"maze.{item_split}.model_needed.v3.hazard_unknown.{target_color}.{index:05d}", item_split, target_color, 7, doors, True, True)
                env._state = env._make_state(False, False, 0, False)
                source = "slot-model-needed-maze-v3-hazard-unknown"
                step_values = [0]
            elif scenario == "hazard_active_known":
                env = MazeHazardWorld(f"maze.{item_split}.model_needed.v3.hazard_active.{target_color}.{index:05d}", item_split, target_color, 7, doors, True, True)
                source = "slot-model-needed-maze-v3-hazard-active-known"
                step_values = [0, 1]
            elif scenario == "safe_no_hazard":
                env = MazeHazardWorld(f"maze.{item_split}.model_needed.v3.safe_no_hazard.{target_color}.{index:05d}", item_split, target_color, 7, doors, False, False)
                source = "slot-model-needed-maze-v3-safe-no-hazard"
                step_values = [0, 1, 3]
            else:
                env = MazeHazardWorld(f"maze.{item_split}.model_needed.v3.post_setup.{target_color}.{index:05d}", item_split, target_color, 7, doors, False, True)
                source = "slot-model-needed-maze-v3-post-setup-cleared"
                step_values = [1, 2, 3, 4, 5, 6]
            for steps in step_values:
                if len(snapshots) >= target_groups:
                    break
                if scenario == "hazard_unknown":
                    env.hazard_active = True
                    env._state = env._make_state(False, False, min(steps, env.max_steps - 1), False)
                elif scenario == "hazard_active_known":
                    env.hazard_active = True
                    env._state = env._make_state(True, False, min(steps, env.max_steps - 1), False)
                else:
                    env.hazard_active = False
                    env._state = env._make_state(True, False, min(steps, env.max_steps - 1), False)
                snapshots.append(ReachableSnapshot(env.name, env.split, source, _variant_seed(env.name), steps, env.snapshot(), env._state))
        index += 1
    return snapshots[:target_groups]


def _collect_tool_precondition_snapshots(split: str, target_groups: int) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    sequences = (("read", "summarize"), ("read", "read", "summarize"))
    index = 0
    while len(snapshots) < target_groups:
        item_split = _variant_split(index)
        if _include_split(item_split, split):
            sequence = sequences[index % len(sequences)]
            for ready in (False, True):
                env_name = f"tool.{item_split}.model_needed.precondition.{int(ready)}.{index:05d}"
                env = ToolPreconditionWorld(env_name, item_split, len(sequence) + 4, sequence, ready)
                for stage in range(len(sequence)):
                    step_values = [stage] if not ready else sorted({stage, min(env.max_steps - 1, stage + 1), min(env.max_steps - 1, stage + 2)})
                    for steps in step_values:
                        if len(snapshots) >= target_groups:
                            return snapshots
                        env.precondition_ready = ready
                        env._state = env._make_state(stage, steps, False)
                        source = "slot-model-needed-tool-precondition-pre" if not ready else "slot-model-needed-tool-precondition-post-setup"
                        snapshots.append(ReachableSnapshot(env.name, env.split, source, _variant_seed(env.name), steps, env.snapshot(), env._state))
        index += 1
    return snapshots[:target_groups]


def _collect_partial_hidden_snapshots(split: str) -> list[ReachableSnapshot]:
    snapshots: list[ReachableSnapshot] = []
    for item_split in ("train", "test"):
        if not _include_split(item_split, split):
            continue
        lock_codes = ((2, 4, 1), (1, 3, 4), (4, 2, 2, 1))
        for code_index, code in enumerate(lock_codes):
            env = LockWorld(f"lock.{item_split}.partial_hidden.{code_index}", item_split, code, max(7, len(code) + 4), visibility_mode="partial_hidden")
            for progress in range(len(code)):
                env.reset(0)
                env._known_mask = env._initial_known_mask()
                env._state = env._make_state(progress, progress, False)
                snapshots.append(ReachableSnapshot(env.name, env.split, "partial-hidden", _variant_seed(env.name), progress, env.snapshot(), env._state))
        tool_sequences = (
            ("search", "read", "search", "summarize"),
            ("read", "search", "read", "summarize"),
            ("search", "search", "read", "summarize"),
            ("read", "read", "search", "summarize"),
        )
        for seq_index, sequence in enumerate(tool_sequences):
            env = ToolWorld(f"tool.{item_split}.partial_hidden.{seq_index}", item_split, len(sequence) + 5, sequence, visibility_mode="partial_hidden")
            for stage in range(len(sequence)):
                env.reset(0)
                env._known_mask = env._initial_known_mask()
                env._state = env._make_state(stage, stage, False)
                snapshots.append(ReachableSnapshot(env.name, env.split, "partial-hidden", _variant_seed(env.name), stage, env.snapshot(), env._state))
    return snapshots


def _bfs_snapshots(env: MicroWorld, source_agent: str, max_depth: int) -> list[ReachableSnapshot]:
    start_state = env.reset(0)
    start_snapshot = env.snapshot()
    seen: set[str] = set()
    results: list[ReachableSnapshot] = []
    queue: deque[tuple[dict[str, Any], int]] = deque([(start_snapshot, 0)])
    while queue:
        restore_state, depth = queue.popleft()
        current_env = _make_env_from_restore_state(env.name, env.split, restore_state)
        state = current_env.restore(restore_state)
        snapshot = ReachableSnapshot(current_env.name, current_env.split, source_agent, _variant_seed(current_env.name), depth, current_env.snapshot(), state)
        if snapshot.restore_id in seen:
            continue
        seen.add(snapshot.restore_id)
        if not state.terminal:
            results.append(snapshot)
        if state.terminal or depth >= max_depth:
            continue
        for action in current_env.candidate_actions(state):
            probe_env = _make_env_from_restore_state(env.name, env.split, restore_state)
            probe_env.restore(restore_state)
            result = probe_env.step(action)
            if not result.next_state.terminal or depth + 1 <= max_depth:
                queue.append((probe_env.snapshot(), depth + 1))
    if not results and not start_state.terminal:
        results.append(ReachableSnapshot(env.name, env.split, source_agent, _variant_seed(env.name), 0, start_snapshot, start_state))
    return results


def _evaluate_snapshot_candidates(snapshot: ReachableSnapshot) -> list[dict[str, Any]]:
    env = _make_env_from_snapshot(snapshot)
    state = env.restore(snapshot.restore_state)
    candidates = env.candidate_actions(state)
    candidate_keys = [canonical_action_key(action) for action in candidates]
    candidate_set_id = _hash_json({"env_name": snapshot.env_name, "restore_id": snapshot.restore_id, "candidate_keys": candidate_keys})
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(candidates):
        probe_env = _make_env_from_snapshot(snapshot)
        before = probe_env.restore(snapshot.restore_state)
        result = probe_env.step(action)
        progress_before = _progress(snapshot.env_name, before)
        progress_after = _progress(snapshot.env_name, result.next_state)
        progress_delta = progress_after - progress_before
        local_success = 1.0 if bool(result.info.get("success")) else 0.0
        information_gain = float(result.info.get("information_gain", _information_gain(snapshot.env_name, before, result.next_state, action)))
        actual_action_score = float(result.reward) + 2.0 * progress_delta + information_gain + 0.5 * local_success
        post_setup_execution = _is_post_setup_execution(before, action)
        setup_state = str(before.facts.get("setup_state", "none"))
        rows.append(
            {
                "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
                "world_id": f"{snapshot.env_name}.seed_{snapshot.seed:04d}",
                "env_family": snapshot.env_name.split(".", 1)[0],
                "env_name": snapshot.env_name,
                "split": snapshot.split,
                "source_agent": snapshot.source_agent,
                "source_seed": snapshot.seed,
                "source_step_index": snapshot.step_index,
                "restore_id": snapshot.restore_id,
                "agent_state_id": snapshot.agent_state_id,
                "candidate_set_id": candidate_set_id,
                "candidate_count": len(candidates),
                "candidate_action_index": index,
                "candidate_action": _action_to_dict(action),
                "candidate_action_key": canonical_action_key(action),
                "candidate_action_metadata": _action_metadata(env.action_specs, action),
                "setup_state": setup_state,
                "setup_known": bool(before.facts.get("setup_known", False)),
                "setup_effective": bool(before.facts.get("setup_effective", False)),
                "post_setup_state": bool(before.facts.get("post_setup_state", False)),
                "post_setup_execution": post_setup_execution,
                "model_needed_kind": before.facts.get("model_needed_kind"),
                "state_vector": list(before.vector),
                "state_facts": before.facts,
                "next_state_vector": list(result.next_state.vector),
                "next_state_facts": result.next_state.facts,
                "reward": float(result.reward),
                "done": bool(result.done),
                "local_success": local_success,
                "progress_before": progress_before,
                "progress_after": progress_after,
                "progress_delta": progress_delta,
                "information_gain": information_gain,
                "state_change_magnitude": _state_change_magnitude(before, result.next_state),
                "actual_action_score": actual_action_score,
                "valid_action": 1.0,
                "failure_reason": None,
            }
        )
    return rows


def _make_env_from_snapshot(snapshot: ReachableSnapshot) -> MicroWorld:
    return _make_env_from_restore_state(snapshot.env_name, snapshot.split, snapshot.restore_state)


def _make_env_from_restore_state(env_name: str, split: str, restore_state: dict[str, Any]) -> MicroWorld:
    try:
        return make_env(env_name)
    except KeyError:
        pass
    kind = restore_state.get("model_needed_kind")
    if kind == "lock_trap":
        state = dict(restore_state.get("state") or {})
        code = tuple(int(value) for value in restore_state.get("code", state.get("code", (2, 4, 1))))
        return LockTrapWorld(env_name, split, code, int(restore_state.get("max_steps", state.get("max_steps", 8))), bool(restore_state.get("trap_armed", state.get("trap_armed", True))))
    if kind == "maze_hazard":
        state = dict(restore_state.get("state") or {})
        target_color = str(restore_state.get("target_color", state.get("target_color", "green")))
        doors = tuple(str(value) for value in restore_state.get("doors", state.get("doors", MemoryMaze.doors)))
        return MazeHazardWorld(
            env_name,
            split,
            target_color,
            int(restore_state.get("max_steps", state.get("max_steps", 7))),
            doors,
            bool(restore_state.get("hazard_active", state.get("hazard_active", True))),
            bool(restore_state.get("hazard_present", state.get("hazard_present", True))),
        )
    if kind == "tool_precondition":
        state = dict(restore_state.get("state") or {})
        sequence = tuple(str(value) for value in restore_state.get("sequence", state.get("sequence", ("read", "summarize"))))
        return ToolPreconditionWorld(env_name, split, int(restore_state.get("max_steps", state.get("max_steps", len(sequence) + 4))), sequence, bool(restore_state.get("precondition_ready", state.get("precondition_ready", False))))
    family = env_name.split(".", 1)[0]
    state = dict(restore_state.get("state") or {})
    if family == "lock":
        code = tuple(int(value) for value in restore_state.get("code", state.get("code", (1, 2, 3))))
        return LockWorld(env_name, split, code, int(restore_state.get("max_steps", state.get("max_steps", 6))), str(restore_state.get("visibility_mode", state.get("visibility_mode", "visible"))))
    if family == "maze":
        target_color = str(restore_state.get("target_color", state.get("target_color", "blue")))
        doors = tuple(str(value) for value in restore_state.get("doors", state.get("doors", MemoryMaze.doors)))
        visibility_mode = str(restore_state.get("visibility_mode", state.get("visibility_mode", "partial_hidden")))
        return MemoryMaze(env_name, split, target_color, int(restore_state.get("max_steps", state.get("max_steps", 7))), doors, visibility_mode)
    if family == "tool":
        sequence = tuple(str(value) for value in restore_state.get("sequence", state.get("sequence", TOOL_ACTIONS)))
        return ToolWorld(env_name, split, int(restore_state.get("max_steps", state.get("max_steps", 5))), sequence, str(restore_state.get("visibility_mode", state.get("visibility_mode", "visible"))))
    raise KeyError(f"unknown environment family: {family}")


def _make_agent(name: str, seed: int):
    if name == "random":
        return RandomAgent(seed)
    if name == "heuristic":
        return HeuristicAgent()
    if name == "gated":
        return GatedAgent(SymbolicPredictor())
    raise ValueError(name)


def _progress(env_name: str, state: WorldState) -> float:
    family = env_name.split(".", 1)[0]
    if family == "lock":
        return float(state.facts["progress"]) / max(1.0, float(len(state.facts["code"])))
    if family == "maze":
        if bool(state.facts["escaped"]):
            return 1.0
        return 0.3 if bool(state.facts["saw_hint"]) else 0.0
    if family == "tool":
        return float(state.facts["stage"]) / max(1.0, float(len(state.facts["sequence"])))
    return float(state.vector[0])


def _information_gain(env_name: str, before: WorldState, after: WorldState, action: Action) -> float:
    family = env_name.split(".", 1)[0]
    if family == "maze" and action.name == "inspect" and not before.facts.get("saw_hint") and after.facts.get("saw_hint"):
        return 1.0
    if family == "tool":
        before_stage = int(before.facts["stage"])
        after_stage = int(after.facts["stage"])
        if action.name in {"search", "read"} and after_stage > before_stage:
            return 1.0
    return 0.0


def _state_change_magnitude(before: WorldState, after: WorldState) -> float:
    return sum(abs(float(left) - float(right)) for left, right in zip(before.vector, after.vector)) / len(before.vector)


def _action_to_dict(action: Action) -> dict[str, Any]:
    return {"name": action.name, "params": dict(action.params), "source": action.source, "confidence": action.confidence}


def _action_metadata(specs, action: Action) -> dict[str, bool]:
    for spec in specs:
        if spec.name == action.name:
            return {
                "reveals_information": spec.reveals_information,
                "requires_known_slot": spec.requires_known_slot,
                "terminal_only": spec.terminal_only,
                "consumes_progress_step": spec.consumes_progress_step,
            }
    return {
        "reveals_information": False,
        "requires_known_slot": False,
        "terminal_only": False,
        "consumes_progress_step": True,
    }


def _is_post_setup_execution(state: WorldState, action: Action) -> bool:
    if not bool(state.facts.get("post_setup_state", False)):
        return False
    family = state.env_name.split(".", 1)[0]
    if family == "lock":
        return action.name == "press"
    if family == "maze":
        return action.name == "move"
    if family == "tool":
        return action.name in {"read", "summarize"}
    return False


def _candidate_count_by_env(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["env_name"])] = int(row["candidate_count"])
    return counts


def _dedupe_snapshots(snapshots: Iterable[ReachableSnapshot]) -> list[ReachableSnapshot]:
    by_restore_id: dict[str, ReachableSnapshot] = {}
    for snapshot in snapshots:
        by_restore_id.setdefault(snapshot.restore_id, snapshot)
    return sorted(by_restore_id.values(), key=lambda item: (item.env_name, item.restore_id))


def _bounded_lock_codes(lengths: Sequence[int], count: int) -> list[tuple[int, ...]]:
    codes: list[tuple[int, ...]] = []
    for length in lengths:
        for code in itertools.product(LOCK_DIGITS, repeat=int(length)):
            codes.append(tuple(int(value) for value in code))
            if len(codes) >= count:
                return codes
    return codes


def _known_mask_for_visibility(length: int, visibility: str) -> list[bool]:
    if visibility == "visible":
        return [True] * length
    if visibility == "hidden":
        return [False] * length
    midpoint = max(1, length // 2) if length else 0
    return [index < midpoint for index in range(length)]


def _tool_curriculum_sequences(count: int) -> list[tuple[str, ...]]:
    base_sequences: list[tuple[str, ...]] = []
    for length in (3, 4, 5, 6):
        for prefix in itertools.product(("search", "read"), repeat=length - 1):
            sequence = tuple(prefix) + ("summarize",)
            base_sequences.append(sequence)
    if not base_sequences:
        return []
    return [base_sequences[index % len(base_sequences)] for index in range(count)]


def _variant_split(index: int) -> str:
    return "test" if index % 5 == 0 else "train"


def _include_split(item_split: str, requested_split: str) -> bool:
    return requested_split == "all" or item_split == requested_split


def _variant_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _validate_counterfactual_row(row: dict[str, Any], line_no: int) -> None:
    required = [
        "schema_version",
        "env_name",
        "split",
        "restore_id",
        "agent_state_id",
        "candidate_set_id",
        "candidate_action",
        "candidate_action_key",
        "state_vector",
        "next_state_vector",
        "reward",
        "done",
        "local_success",
        "progress_delta",
        "information_gain",
        "actual_action_score",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"line {line_no}: missing keys {missing}")
    if row["schema_version"] != COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError(f"line {line_no}: schema mismatch")
    if len(row["state_vector"]) != len(row["next_state_vector"]):
        raise ValueError(f"line {line_no}: state vector length mismatch")


def _hash_json(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
