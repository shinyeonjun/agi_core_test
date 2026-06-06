from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurokernel_seed.core.belief import SLOT_VALUE_VOCAB, BeliefState, belief_from_facts
from neurokernel_seed.core.gate import current_required_action_key
from neurokernel_seed.core.schema import Action, WorldState
from neurokernel_seed.replay.dataset import canonical_action_key, load_records

SLOT_FEATURE_SCHEMA_VERSION = "neurokernel-slot-transition-v2"
BALANCED_SLOT_DATASET_PROFILE = "neurokernel-slot-balanced-v1"
MAX_SLOTS = 8
VISIBILITY_MODES = ("visible", "partial_hidden", "hidden")
OBSERVATION_TYPES = ("none", "inspect", "search", "read", "observe_panel", "read_clue")
SLOT_ACTION_VOCAB = [
    "inspect",
    'move(door="blue")',
    'move(door="green")',
    'move(door="red")',
    "press(digit=1)",
    "press(digit=2)",
    "press(digit=3)",
    "press(digit=4)",
    "read",
    "reset",
    "search",
    "summarize",
]
GLOBAL_FEATURE_NAMES = [
    *[f"visibility_mode.{mode}" for mode in VISIBILITY_MODES],
    "current_index",
    "sequence_length",
    "remaining_steps",
    "progress_fraction",
    "current_slot_known",
    "pending_information_need",
    "executable_now",
    "can_finish",
    "maze.target_color_known",
    "maze.target_color.red",
    "maze.target_color.blue",
    "maze.target_color.green",
    "maze.hazard_known",
    "maze.hazard_present",
    "maze.hazard_active",
    "maze.hazard_cleared",
    "setup_state.pre_setup",
    "setup_state.post_setup_done",
    "setup_state.safe_no_hazard",
    "post_setup_state",
    "can_move_target",
    *[f"last_observation_type.{kind}" for kind in OBSERVATION_TYPES],
    "last_observation_slot",
    *[f"last_observation_value.{value}" for value in SLOT_VALUE_VOCAB],
]
SLOT_FEATURE_NAMES = [
    "known_mask",
    "unknown_mask",
    "is_current_slot",
    "is_past_slot",
    "is_future_slot",
    "revealed_flag",
    "confidence",
    "maze.is_target_slot",
    "maze.hazard_known",
    "maze.hazard_active",
    "maze.hazard_cleared",
    "maze.blocked",
    "maze.inspected",
    *[f"value.{value}" for value in SLOT_VALUE_VOCAB],
]


@dataclass(frozen=True)
class SlotEncoderManifest:
    schema_version: str
    replay_schema_version: str
    envs: list[str]
    env_families: list[str]
    splits: list[str]
    action_vocab: list[str]
    max_slots: int
    slot_value_vocab: list[str]
    global_feature_names: list[str]
    slot_feature_names: list[str]
    max_state_dim: int
    state_dim_by_env: dict[str, int]
    input_dim: int
    target_dim: int
    input_layout: dict[str, Any]
    target_layout: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "replay_schema_version": self.replay_schema_version,
            "envs": self.envs,
            "env_families": self.env_families,
            "splits": self.splits,
            "action_vocab": self.action_vocab,
            "max_slots": self.max_slots,
            "slot_value_vocab": self.slot_value_vocab,
            "global_feature_names": self.global_feature_names,
            "slot_feature_names": self.slot_feature_names,
            "max_state_dim": self.max_state_dim,
            "state_dim_by_env": self.state_dim_by_env,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "input_layout": self.input_layout,
            "target_layout": self.target_layout,
        }


@dataclass
class CandidateGroupSummary:
    group_key: str
    candidate_set_id: str
    split: str
    env_name: str
    env_family: str
    visibility_mode: str
    source_label: str
    length_bucket: str
    rows: int = 0
    has_success: bool = False
    has_information_gain: bool = False
    has_post_reveal_execution: bool = False
    has_post_setup_execution: bool = False
    has_post_setup_positive_execution: bool = False
    has_post_setup_correct_action: bool = False
    has_positive_progress: bool = False
    has_negative_progress: bool = False
    has_zero_progress: bool = False


def inspect_slot_features(path: str | Path) -> dict[str, Any]:
    records = load_records(path)
    manifest = build_slot_manifest(records)
    return {"rows": len(records), **manifest.as_dict()}


def export_slot_features(replay_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    records = load_records(replay_path)
    manifest = build_slot_manifest(records)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            sample = encode_slot_record(record, manifest, row_index)
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps({"rows": len(records), **manifest.as_dict()}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": len(records), "out": str(out), "manifest": str(manifest_path), **manifest.as_dict()}


def validate_slot_features(path: str | Path) -> dict[str, Any]:
    feature_path = Path(path)
    manifest_path = feature_path.with_suffix(feature_path.suffix + ".manifest.json")
    if not feature_path.exists():
        raise ValueError(f"missing slot feature file: {feature_path}")
    if not manifest_path.exists():
        raise ValueError(f"missing slot feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = 0
    with feature_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            _validate_slot_sample(sample, manifest, line_no)
            rows += 1
    if rows != manifest.get("rows"):
        raise ValueError(f"slot feature row count mismatch: data={rows}, manifest={manifest.get('rows')}")
    return {"rows": rows, "schema_version": manifest["schema_version"], "input_dim": manifest["input_dim"], "target_dim": manifest["target_dim"]}


def merge_slot_feature_files(paths: list[str | Path], out_path: str | Path) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("merge requires at least two slot feature files")
    manifests = [_read_slot_manifest(Path(path)) for path in paths]
    base = dict(manifests[0])
    for manifest in manifests[1:]:
        _validate_merge_compatible(base, manifest)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    envs: set[str] = set()
    splits: set[str] = set()
    state_dim_by_env: dict[str, int] = {}
    with out.open("w", encoding="utf-8") as dest:
        for path, source_manifest in zip(paths, manifests, strict=True):
            source_state_dims = source_manifest.get("state_dim_by_env", {})
            with Path(path).open("r", encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    sample["row_index"] = rows
                    env_name = str(sample["env_name"])
                    split = str(sample.get("split", "unknown"))
                    envs.add(env_name)
                    splits.add(split)
                    state_dim = source_state_dims.get(env_name)
                    if state_dim is None:
                        state_dim = _state_dim_from_sample(sample, base)
                    state_dim_by_env.setdefault(env_name, int(state_dim))
                    dest.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                    rows += 1
    merged_manifest = dict(base)
    merged_manifest["rows"] = rows
    merged_manifest["envs"] = sorted(envs)
    merged_manifest["splits"] = sorted(splits)
    merged_manifest["state_dim_by_env"] = dict(sorted(state_dim_by_env.items()))
    merged_manifest["source_datasets"] = [str(Path(path)) for path in paths]
    merged_manifest["dataset_profile"] = "neurokernel-slot-merged-v1"
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(merged_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    validation = validate_slot_features(out)
    return {"out": str(out), "manifest": str(manifest_path), "rows": rows, "schema_version": validation["schema_version"], "input_dim": validation["input_dim"], "target_dim": validation["target_dim"]}


def audit_slot_dataset(path: str | Path) -> dict[str, Any]:
    audit, _, _ = _scan_slot_dataset(path)
    return audit


def audit_maze_grounding(path: str | Path, *, limit: int = 20) -> dict[str, Any]:
    feature_path = Path(path)
    manifest = _read_slot_manifest(feature_path)
    if manifest.get("schema_version") != SLOT_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"maze grounding audit requires {SLOT_FEATURE_SCHEMA_VERSION}")
    layout = manifest["input_layout"]
    action_vocab = list(manifest["action_vocab"])
    global_names = list(manifest["global_feature_names"])
    slot_names = list(manifest["slot_feature_names"])
    target_indices = _target_indices(manifest)
    action_start, action_end = int(layout["action_onehot"][0]), int(layout["action_onehot"][1])
    global_start, _global_end = int(layout["belief_global"][0]), int(layout["belief_global"][1])
    slot_start, _slot_end = int(layout["slot_flat"][0]), int(layout["slot_flat"][1])
    per_slot = len(slot_names)
    global_index = {name: global_start + index for index, name in enumerate(global_names)}
    slot_index = {name: index for index, name in enumerate(slot_names)}
    totals: Counter[str] = Counter()
    by_color: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    with feature_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if _env_family(sample) != "maze":
                continue
            source = sample.get("source", {})
            target_color = str(source.get("visible_target_color") or source.get("target_color") or "")
            action_key = str(sample.get("action_key", ""))
            input_vector = list(sample.get("input_vector", []))
            target_vector = list(sample.get("target_vector", []))
            decoded_action = _decode_action_onehot(input_vector[action_start:action_end], action_vocab)
            action_matches = decoded_action == action_key
            target_onehot = _global_value(input_vector, global_index.get(f"maze.target_color.{target_color}"))
            target_known = _global_value(input_vector, global_index.get("maze.target_color_known"))
            hazard_active = _global_value(input_vector, global_index.get("maze.hazard_active"))
            hazard_cleared = _global_value(input_vector, global_index.get("maze.hazard_cleared"))
            post_setup = _global_value(input_vector, global_index.get("post_setup_state"))
            target_slot_count = _maze_target_slot_count(input_vector, slot_start, per_slot, int(manifest.get("max_slots", MAX_SLOTS)), slot_index)
            success = _target_value(target_vector, target_indices["local_success"]) > 0.5
            progress_delta = _target_value(target_vector, target_indices["progress_delta"])
            is_move = action_key.startswith("move(")
            required = str(source.get("current_required_action_key") or "")
            is_target_action = bool(required and action_key == required)
            totals["maze_rows"] += 1
            if "model_needed" in str(source.get("model_needed_kind", "")) or source.get("model_needed_kind") == "maze_hazard":
                totals["maze_model_needed_rows"] += 1
            if action_matches:
                totals["action_roundtrip_ok"] += 1
            if target_color in {"red", "green", "blue"} and target_known > 0.5 and target_onehot > 0.5:
                totals["target_color_encoded_rows"] += 1
            if target_slot_count == 1:
                totals["single_target_slot_rows"] += 1
            if bool(source.get("post_setup_state", False)) and is_move:
                totals["post_setup_move_rows"] += 1
                by_color[target_color]["post_setup_move_rows"] += 1
                if is_target_action:
                    totals["post_setup_target_move_rows"] += 1
                    by_color[target_color]["post_setup_target_move_rows"] += 1
                    if success or progress_delta > 1e-9:
                        totals["post_setup_target_move_positive_rows"] += 1
                        by_color[target_color]["post_setup_target_move_positive_rows"] += 1
                else:
                    totals["post_setup_wrong_move_rows"] += 1
                    by_color[target_color]["post_setup_wrong_move_rows"] += 1
            row_failures: list[str] = []
            if not action_matches:
                row_failures.append("action_onehot_roundtrip_mismatch")
            if target_color in {"red", "green", "blue"} and (target_known <= 0.5 or target_onehot <= 0.5):
                row_failures.append("target_color_not_encoded")
            if target_color in {"red", "green", "blue"} and target_slot_count != 1:
                row_failures.append("target_slot_count_not_one")
            if row_failures and len(failures) < limit:
                failures.append({"line": line_no, "env_name": sample.get("env_name"), "action_key": action_key, "decoded_action": decoded_action, "target_color": target_color, "target_slot_count": target_slot_count, "failures": row_failures})
            if len(examples) < limit and bool(source.get("post_setup_state", False)) and is_move:
                examples.append(
                    {
                        "line": line_no,
                        "env_name": sample.get("env_name"),
                        "action_key": action_key,
                        "decoded_action": decoded_action,
                        "target_color": target_color,
                        "required_action_key": required,
                        "target_known_feature": target_known,
                        "target_onehot_feature": target_onehot,
                        "hazard_active_feature": hazard_active,
                        "hazard_cleared_feature": hazard_cleared,
                        "post_setup_feature": post_setup,
                        "target_slot_count": target_slot_count,
                        "success": success,
                        "progress_delta": progress_delta,
                    }
                )
    color_report = {color: dict(counts) for color, counts in sorted(by_color.items())}
    passed = (
        totals["maze_rows"] > 0
        and totals["action_roundtrip_ok"] == totals["maze_rows"]
        and totals["target_color_encoded_rows"] == totals["maze_rows"]
        and totals["single_target_slot_rows"] == totals["maze_rows"]
        and totals["post_setup_target_move_positive_rows"] > 0
        and all(color_report.get(color, {}).get("post_setup_target_move_positive_rows", 0) > 0 for color in ("red", "green", "blue"))
    )
    return {
        "passed": bool(passed),
        "path": str(path),
        "schema_version": manifest["schema_version"],
        "rows": manifest.get("rows"),
        "counts": dict(totals),
        "by_target_color": color_report,
        "failure_examples": failures,
        "post_setup_examples": examples,
    }


def check_slot_dataset_gates(
    path: str | Path,
    *,
    lock_max_ratio: float = 0.60,
    maze_min_ratio: float = 0.15,
    tool_min_ratio: float = 0.15,
    hiddenish_min_ratio: float = 0.25,
    information_gain_group_min_ratio: float = 0.08,
    post_reveal_group_min_ratio: float = 0.08,
    post_setup_execution_group_min_ratio: float = 0.03,
    post_setup_positive_group_min_ratio: float = 0.02,
    post_setup_correct_action_group_min_ratio: float = 0.02,
) -> dict[str, Any]:
    audit = audit_slot_dataset(path)
    train_family = audit.get("candidate_groups_by_split_family", {}).get("train", {})
    train_visibility = audit.get("candidate_groups_by_split_visibility", {}).get("train", {})
    train_groups = sum(int(value) for value in train_family.values())
    total_groups = max(1, int(audit.get("candidate_groups", 0)))
    special = audit.get("special_candidate_groups", {})
    gates = {
        "lock_max_ratio": _gate_ratio(train_family.get("lock", 0), train_groups, lock_max_ratio, mode="max"),
        "maze_min_ratio": _gate_ratio(train_family.get("maze", 0), train_groups, maze_min_ratio, mode="min"),
        "tool_min_ratio": _gate_ratio(train_family.get("tool", 0), train_groups, tool_min_ratio, mode="min"),
        "partial_hidden_or_hidden_min_ratio": _gate_ratio(train_visibility.get("partial_hidden", 0) + train_visibility.get("hidden", 0), train_groups, hiddenish_min_ratio, mode="min"),
        "information_gain_group_min_ratio": _gate_ratio(special.get("information_gain_groups", 0), total_groups, information_gain_group_min_ratio, mode="min"),
        "post_reveal_group_min_ratio": _gate_ratio(special.get("post_reveal_execution_groups", 0), total_groups, post_reveal_group_min_ratio, mode="min"),
        "post_setup_execution_group_min_ratio": _gate_ratio(special.get("post_setup_execution_groups", 0), total_groups, post_setup_execution_group_min_ratio, mode="min"),
        "post_setup_positive_group_min_ratio": _gate_ratio(special.get("post_setup_positive_execution_groups", 0), total_groups, post_setup_positive_group_min_ratio, mode="min"),
        "post_setup_correct_action_group_min_ratio": _gate_ratio(special.get("post_setup_correct_action_groups", 0), total_groups, post_setup_correct_action_group_min_ratio, mode="min"),
    }
    return {
        "passed": all(item["passed"] for item in gates.values()),
        "path": str(path),
        "schema_version": audit["schema_version"],
        "dataset_profile": audit.get("dataset_profile"),
        "candidate_groups": audit["candidate_groups"],
        "train_candidate_groups": train_groups,
        "gates": gates,
        "audit_shortage_report": audit.get("shortage_report", []),
    }


def export_balanced_slot_dataset(
    path: str | Path,
    out_path: str | Path,
    *,
    target_rows: int = 120_000,
    seed: int = 0,
    include_test: bool = True,
    family_ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Export a train-balanced slot dataset while preserving candidate groups."""
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    raw_path = Path(path)
    out = Path(out_path)
    family_ratios = family_ratios or {"lock": 0.45, "maze": 0.275, "tool": 0.275}
    audit_before, groups, raw_manifest = _scan_slot_dataset(raw_path)
    selected_groups, selection_report = _select_balanced_train_groups(groups, target_rows, seed, family_ratios)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    train_rows_written = 0
    heldout_rows_written = 0
    with raw_path.open("r", encoding="utf-8") as source, out.open("w", encoding="utf-8") as dest:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            split = str(sample.get("split", "unknown"))
            group_key = _candidate_group_key(sample, line_no)
            write_row = split == "train" and group_key in selected_groups
            if include_test and split != "train":
                write_row = True
            if not write_row:
                continue
            dest.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            if split == "train":
                train_rows_written += 1
            else:
                heldout_rows_written += 1
    manifest = dict(raw_manifest)
    manifest["rows"] = rows_written
    manifest["schema_version"] = raw_manifest["schema_version"]
    manifest["dataset_profile"] = BALANCED_SLOT_DATASET_PROFILE
    manifest["source_dataset"] = str(raw_path)
    manifest["balanced_train_rows"] = train_rows_written
    manifest["heldout_rows_preserved"] = heldout_rows_written
    manifest["heldout_untouched"] = bool(include_test)
    manifest["sampling_policy"] = {
        "unit": "candidate_set_id",
        "target_train_rows": target_rows,
        "seed": seed,
        "family_ratios": family_ratios,
        "priority": [
            "information_gain_positive",
            "post_reveal_execution",
            "partial_or_hidden_visibility",
            "successful_candidate_group",
            "nonzero_progress",
        ],
    }
    manifest["selection_report"] = selection_report
    manifest["audit_before"] = audit_before
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    audit_after = audit_slot_dataset(out)
    manifest["audit_after"] = audit_after
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "out": str(out),
        "manifest": str(manifest_path),
        "schema_version": manifest["schema_version"],
        "dataset_profile": BALANCED_SLOT_DATASET_PROFILE,
        "rows": rows_written,
        "balanced_train_rows": train_rows_written,
        "heldout_rows_preserved": heldout_rows_written,
        "selection_report": selection_report,
        "audit_after": audit_after,
    }


def build_slot_manifest(records: list[dict[str, Any]]) -> SlotEncoderManifest:
    if not records:
        raise ValueError("cannot build slot feature manifest from empty replay")
    source_schema = str(records[0]["schema_version"])
    if source_schema != "neurokernel-counterfactual-transition-v1":
        raise ValueError("slot features currently require counterfactual transition rows")
    envs = sorted({record["env_name"] for record in records})
    splits = sorted({record["split"] for record in records if record.get("split")})
    env_families = sorted({record["env_family"] for record in records})
    state_dim_by_env: dict[str, int] = {}
    for record in records:
        env_name = record["env_name"]
        dim = len(record["state_vector"])
        previous = state_dim_by_env.setdefault(env_name, dim)
        if previous != dim:
            raise ValueError(f"state vector dimension changed for {env_name}: {previous} -> {dim}")
    max_state_dim = max(state_dim_by_env.values())
    observed_action_vocab = {record["candidate_action_key"] for record in records}
    action_vocab = sorted(set(SLOT_ACTION_VOCAB) | observed_action_vocab)
    env_family_end = len(env_families)
    global_end = env_family_end + len(GLOBAL_FEATURE_NAMES)
    slot_end = global_end + MAX_SLOTS * len(SLOT_FEATURE_NAMES)
    state_end = slot_end + max_state_dim
    input_layout = {
        "env_family_onehot": [0, env_family_end],
        "belief_global": [env_family_end, global_end],
        "slot_flat": [global_end, slot_end],
        "state_padded": [slot_end, state_end],
        "action_onehot": [state_end, state_end + len(action_vocab)],
    }
    target_layout = {
        "next_state_padded": [0, max_state_dim],
        "reward": max_state_dim,
        "done": max_state_dim + 1,
        "local_success": max_state_dim + 2,
        "progress_delta": max_state_dim + 3,
        "information_gain": max_state_dim + 4,
    }
    return SlotEncoderManifest(
        schema_version=SLOT_FEATURE_SCHEMA_VERSION,
        replay_schema_version=source_schema,
        envs=envs,
        env_families=env_families,
        splits=splits,
        action_vocab=action_vocab,
        max_slots=MAX_SLOTS,
        slot_value_vocab=list(SLOT_VALUE_VOCAB),
        global_feature_names=list(GLOBAL_FEATURE_NAMES),
        slot_feature_names=list(SLOT_FEATURE_NAMES),
        max_state_dim=max_state_dim,
        state_dim_by_env=state_dim_by_env,
        input_dim=len(env_families) + len(GLOBAL_FEATURE_NAMES) + MAX_SLOTS * len(SLOT_FEATURE_NAMES) + max_state_dim + len(action_vocab),
        target_dim=max_state_dim + 5,
        input_layout=input_layout,
        target_layout=target_layout,
    )


def encode_slot_record(record: dict[str, Any], manifest: SlotEncoderManifest, row_index: int = 0) -> dict[str, Any]:
    action = record["candidate_action"]
    action_key = canonical_action_key(action)
    state_vector = _pad(record["state_vector"], manifest.max_state_dim)
    next_state_vector = _pad(record["next_state_vector"], manifest.max_state_dim)
    state_mask = [1.0] * len(record["state_vector"]) + [0.0] * (manifest.max_state_dim - len(record["state_vector"]))
    input_vector = encode_slot_input_vector(record["env_name"], record.get("state_facts", {}), record["state_vector"], action, manifest.as_dict())
    return {
        "schema_version": manifest.schema_version,
        "row_index": row_index,
        "episode_id": record.get("world_id", ""),
        "step_index": record.get("source_step_index", 0),
        "env_name": record["env_name"],
        "split": record["split"],
        "action_key": action_key,
        "candidate_set_id": record.get("candidate_set_id"),
        "actual_action_score": record.get("actual_action_score"),
        "belief": _belief_summary(belief_from_facts(record["env_name"], record.get("state_facts", {}))),
        "input_vector": input_vector,
        "target_vector": next_state_vector + [float(record["reward"]), 1.0 if record["done"] else 0.0, float(record["local_success"]), float(record["progress_delta"]), float(record["information_gain"])],
        "target_mask": state_mask + [1.0, 1.0, 1.0, 1.0, 1.0],
        "source": {
            "restore_id": record["restore_id"],
            "agent_state_id": record["agent_state_id"],
            "candidate_set_id": record["candidate_set_id"],
            "actual_action_score": float(record["actual_action_score"]),
            "source_agent": record.get("source_agent"),
            "target_color": record.get("state_facts", {}).get("target_color"),
            "visible_target_color": record.get("state_facts", {}).get("visible_target_color"),
            "doors": list(record.get("state_facts", {}).get("doors") or []),
            "hazard_present": bool(record.get("state_facts", {}).get("hazard_present", "hazard_active" in record.get("state_facts", {}))),
            "hazard_active": bool(record.get("state_facts", {}).get("hazard_active", False)),
            "hazard_cleared": bool(record.get("state_facts", {}).get("hazard_cleared", False)),
            "current_required_action_key": _source_current_required_action_key(record),
            "setup_state": record.get("setup_state"),
            "setup_known": bool(record.get("setup_known", False)),
            "setup_effective": bool(record.get("setup_effective", False)),
            "post_setup_state": bool(record.get("post_setup_state", False)),
            "post_setup_execution": bool(record.get("post_setup_execution", False)),
            "model_needed_kind": record.get("model_needed_kind"),
        },
    }


def encode_slot_input_vector(env_name: str, state_facts: dict[str, Any], state_vector: list[float] | tuple[float, ...], action: dict[str, Any] | Action, manifest: dict[str, Any]) -> list[float]:
    action_key = canonical_action_key(action)
    action_vocab = list(manifest["action_vocab"])
    if action_key not in action_vocab:
        raise ValueError(f"action not in slot model manifest: {action_key}")
    env_families = list(manifest["env_families"])
    family = env_name.split(".", 1)[0]
    if family not in env_families:
        raise ValueError(f"env family not in slot model manifest: {family}")
    belief = belief_from_facts(env_name, state_facts)
    return (
        _onehot(env_families.index(family), len(env_families))
        + encode_belief_global(belief, state_facts)
        + encode_slot_flat(belief, state_facts, int(manifest.get("max_slots", MAX_SLOTS)))
        + _pad(list(state_vector), int(manifest["max_state_dim"]))
        + _onehot(action_vocab.index(action_key), len(action_vocab))
    )


def encode_belief_global(belief: BeliefState, facts: dict[str, Any] | None = None) -> list[float]:
    facts = facts or {}
    values = {name: 0.0 for name in GLOBAL_FEATURE_NAMES}
    if f"visibility_mode.{belief.visibility_mode}" in values:
        values[f"visibility_mode.{belief.visibility_mode}"] = 1.0
    values["current_index"] = _clip01(belief.current_index / max(1, belief.sequence_length))
    values["sequence_length"] = _clip01(belief.sequence_length / MAX_SLOTS)
    values["remaining_steps"] = _clip01(belief.remaining_steps / 12.0)
    values["progress_fraction"] = _clip01(belief.progress_fraction)
    values["current_slot_known"] = 1.0 if belief.current_slot_known else 0.0
    values["pending_information_need"] = 1.0 if belief.pending_information_need else 0.0
    values["executable_now"] = 1.0 if belief.executable_now else 0.0
    values["can_finish"] = 1.0 if belief.can_finish else 0.0
    if belief.env_family == "maze":
        target_color = _maze_target_color(facts)
        if target_color:
            values["maze.target_color_known"] = 1.0
            key = f"maze.target_color.{target_color}"
            if key in values:
                values[key] = 1.0
        hazard_known = "hazard_active" in facts or "hazard_cleared" in facts or "hazard_present" in facts
        hazard_present = bool(facts.get("hazard_present", hazard_known))
        hazard_active = bool(facts.get("hazard_active", False))
        hazard_cleared = bool(facts.get("hazard_cleared", False))
        values["maze.hazard_known"] = 1.0 if hazard_known else 0.0
        values["maze.hazard_present"] = 1.0 if hazard_present else 0.0
        values["maze.hazard_active"] = 1.0 if hazard_active else 0.0
        values["maze.hazard_cleared"] = 1.0 if hazard_cleared else 0.0
        values["can_move_target"] = 1.0 if target_color and not hazard_active and bool(facts.get("can_finish", False)) else 0.0
    setup_state = str(facts.get("setup_state", ""))
    if f"setup_state.{setup_state}" in values:
        values[f"setup_state.{setup_state}"] = 1.0
    values["post_setup_state"] = 1.0 if bool(facts.get("post_setup_state", False)) else 0.0
    if f"last_observation_type.{belief.last_observation.kind}" in values:
        values[f"last_observation_type.{belief.last_observation.kind}"] = 1.0
    values["last_observation_slot"] = _clip01((belief.last_observation.slot + 1) / max(1, belief.sequence_length + 1))
    if f"last_observation_value.{belief.last_observation.value_key}" in values:
        values[f"last_observation_value.{belief.last_observation.value_key}"] = 1.0
    return [float(values[name]) for name in GLOBAL_FEATURE_NAMES]


def encode_slot_flat(belief: BeliefState, facts: dict[str, Any] | None = None, max_slots: int = MAX_SLOTS) -> list[float]:
    facts = facts or {}
    values: list[float] = []
    target_color = _maze_target_color(facts) if belief.env_family == "maze" else ""
    hazard_known = bool("hazard_active" in facts or "hazard_cleared" in facts or "hazard_present" in facts)
    hazard_active = bool(facts.get("hazard_active", False))
    hazard_cleared = bool(facts.get("hazard_cleared", False))
    inspected = str(facts.get("last_observation_type", "")) == "inspect" or bool(facts.get("saw_hint", False))
    for slot_index in range(max_slots):
        slot_values = {name: 0.0 for name in SLOT_FEATURE_NAMES}
        if slot_index < len(belief.slots):
            slot = belief.slots[slot_index]
            slot_values["known_mask"] = 1.0 if slot.known else 0.0
            slot_values["unknown_mask"] = 1.0 if slot.unknown else 0.0
            slot_values["is_current_slot"] = 1.0 if slot_index == belief.current_index else 0.0
            slot_values["is_past_slot"] = 1.0 if slot_index < belief.current_index else 0.0
            slot_values["is_future_slot"] = 1.0 if slot_index > belief.current_index else 0.0
            slot_values["revealed_flag"] = 1.0 if slot.revealed else 0.0
            slot_values["confidence"] = _clip01(slot.confidence)
            if belief.env_family == "maze":
                color = slot.value_key.removeprefix("color.") if slot.value_key.startswith("color.") else ""
                is_target = bool(target_color and color == target_color)
                slot_values["maze.is_target_slot"] = 1.0 if is_target else 0.0
                slot_values["maze.hazard_known"] = 1.0 if hazard_known else 0.0
                slot_values["maze.hazard_active"] = 1.0 if hazard_active else 0.0
                slot_values["maze.hazard_cleared"] = 1.0 if hazard_cleared else 0.0
                slot_values["maze.blocked"] = 1.0 if hazard_active or not is_target else 0.0
                slot_values["maze.inspected"] = 1.0 if inspected else 0.0
            slot_values[f"value.{slot.value_key}"] = 1.0
        values.extend(float(slot_values[name]) for name in SLOT_FEATURE_NAMES)
    return values


def _belief_summary(belief: BeliefState) -> dict[str, Any]:
    return {
        "env_family": belief.env_family,
        "visibility_mode": belief.visibility_mode,
        "known_slots": list(belief.known_slots),
        "unknown_slots": list(belief.unknown_slots),
        "revealed_slots": list(belief.revealed_slots),
        "current_index": belief.current_index,
        "sequence_length": belief.sequence_length,
        "current_slot_known": belief.current_slot_known,
        "current_required_value": belief.current_required_value,
        "last_observation": {
            "kind": belief.last_observation.kind,
            "slot": belief.last_observation.slot,
            "value_key": belief.last_observation.value_key,
        },
        "pending_information_need": belief.pending_information_need,
        "executable_now": belief.executable_now,
    }


def _source_current_required_action_key(record: dict[str, Any]) -> str | None:
    facts = dict(record.get("state_facts", {}))
    if not facts:
        return None
    state = WorldState(
        str(record.get("env_name", "")),
        str(record.get("state_id", "")),
        tuple(float(value) for value in record.get("state_vector", [])),
        facts,
        bool(record.get("terminal", False)),
    )
    return current_required_action_key(state)


def _maze_target_color(facts: dict[str, Any]) -> str:
    target = str(facts.get("visible_target_color") or facts.get("target_color") or "")
    return target if target in {"red", "green", "blue"} else ""


def _decode_action_onehot(values: list[Any], action_vocab: list[str]) -> str | None:
    if not values or not action_vocab:
        return None
    floats = [float(value) for value in values]
    index = max(range(len(floats)), key=floats.__getitem__)
    if index >= len(action_vocab) or floats[index] <= 0.0:
        return None
    return action_vocab[index]


def _global_value(input_vector: list[Any], index: int | None) -> float:
    if index is None or index < 0 or index >= len(input_vector):
        return 0.0
    return float(input_vector[index])


def _maze_target_slot_count(
    input_vector: list[Any],
    slot_start: int,
    per_slot: int,
    max_slots: int,
    slot_index: dict[str, int],
) -> int:
    offset = slot_index.get("maze.is_target_slot")
    if offset is None:
        return 0
    count = 0
    for slot in range(max_slots):
        index = slot_start + slot * per_slot + offset
        if index < len(input_vector) and float(input_vector[index]) > 0.5:
            count += 1
    return count


def _validate_slot_sample(sample: dict[str, Any], manifest: dict[str, Any], line_no: int) -> None:
    required = ["schema_version", "input_vector", "target_vector", "target_mask", "env_name", "split", "action_key", "belief"]
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"line {line_no}: missing keys {missing}")
    if sample["schema_version"] != manifest["schema_version"]:
        raise ValueError(f"line {line_no}: schema mismatch")
    if len(sample["input_vector"]) != manifest["input_dim"]:
        raise ValueError(f"line {line_no}: input dimension mismatch")
    if len(sample["target_vector"]) != manifest["target_dim"]:
        raise ValueError(f"line {line_no}: target dimension mismatch")
    if len(sample["target_mask"]) != manifest["target_dim"]:
        raise ValueError(f"line {line_no}: target mask dimension mismatch")
    if sample["env_name"] not in manifest["envs"]:
        raise ValueError(f"line {line_no}: unknown env")
    if sample["action_key"] not in manifest["action_vocab"]:
        raise ValueError(f"line {line_no}: unknown action")


def _scan_slot_dataset(path: str | Path) -> tuple[dict[str, Any], dict[str, CandidateGroupSummary], dict[str, Any]]:
    feature_path = Path(path)
    manifest = _read_slot_manifest(feature_path)
    if manifest.get("schema_version") != SLOT_FEATURE_SCHEMA_VERSION:
        raise ValueError(f"slot audit requires {SLOT_FEATURE_SCHEMA_VERSION}")
    target_indices = _target_indices(manifest)
    groups: dict[str, CandidateGroupSummary] = {}
    row_by_split_family: dict[str, Counter[str]] = defaultdict(Counter)
    row_by_split_visibility: dict[str, Counter[str]] = defaultdict(Counter)
    row_by_split_source: dict[str, Counter[str]] = defaultdict(Counter)
    row_by_split_length: dict[str, Counter[str]] = defaultdict(Counter)
    row_by_env: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    post_setup_counts: Counter[str] = Counter()
    rows = 0
    with feature_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            split = str(sample.get("split", "unknown"))
            env_name = str(sample.get("env_name", "unknown"))
            family = _env_family(sample)
            visibility = _visibility_mode(sample)
            source_label = _source_label(sample)
            length_bucket = _length_bucket(sample)
            rows += 1
            row_by_split_family[split][family] += 1
            row_by_split_visibility[split][visibility] += 1
            row_by_split_source[split][source_label] += 1
            row_by_split_length[split][length_bucket] += 1
            row_by_env[env_name] += 1
            target_vector = sample.get("target_vector", [])
            success = _target_value(target_vector, target_indices["local_success"]) > 0.5
            information_gain = _target_value(target_vector, target_indices["information_gain"]) > 1e-9
            progress_delta = _target_value(target_vector, target_indices["progress_delta"])
            if success:
                target_counts["success_rows"] += 1
            if information_gain:
                target_counts["information_gain_rows"] += 1
            if progress_delta > 1e-9:
                target_counts["positive_progress_rows"] += 1
            elif progress_delta < -1e-9:
                target_counts["negative_progress_rows"] += 1
            else:
                target_counts["zero_progress_rows"] += 1
            if _is_post_reveal_execution(sample):
                target_counts["post_reveal_execution_rows"] += 1
            post_setup_execution = _is_post_setup_execution(sample)
            post_setup_correct = _is_post_setup_correct_action(sample)
            if post_setup_execution:
                post_setup_counts["post_setup_execution_rows"] += 1
                if progress_delta > 1e-9 or success:
                    post_setup_counts["post_setup_positive_execution_rows"] += 1
                if post_setup_correct:
                    post_setup_counts["post_setup_correct_action_rows"] += 1
                    if progress_delta > 1e-9 or success:
                        post_setup_counts["post_setup_correct_action_positive_rows"] += 1
            group_key = _candidate_group_key(sample, line_no)
            group = groups.get(group_key)
            if group is None:
                group = CandidateGroupSummary(
                    group_key=group_key,
                    candidate_set_id=str(sample.get("candidate_set_id") or sample.get("source", {}).get("candidate_set_id") or f"line:{line_no}"),
                    split=split,
                    env_name=env_name,
                    env_family=family,
                    visibility_mode=visibility,
                    source_label=source_label,
                    length_bucket=length_bucket,
                )
                groups[group_key] = group
            group.rows += 1
            group.has_success = group.has_success or success
            group.has_information_gain = group.has_information_gain or information_gain
            group.has_post_reveal_execution = group.has_post_reveal_execution or _is_post_reveal_execution(sample)
            group.has_post_setup_execution = group.has_post_setup_execution or post_setup_execution
            group.has_post_setup_positive_execution = group.has_post_setup_positive_execution or (post_setup_execution and (progress_delta > 1e-9 or success))
            group.has_post_setup_correct_action = group.has_post_setup_correct_action or post_setup_correct
            group.has_positive_progress = group.has_positive_progress or progress_delta > 1e-9
            group.has_negative_progress = group.has_negative_progress or progress_delta < -1e-9
            group.has_zero_progress = group.has_zero_progress or abs(progress_delta) <= 1e-9
    group_by_split_family: dict[str, Counter[str]] = defaultdict(Counter)
    group_by_split_visibility: dict[str, Counter[str]] = defaultdict(Counter)
    group_by_split_source: dict[str, Counter[str]] = defaultdict(Counter)
    group_by_split_length: dict[str, Counter[str]] = defaultdict(Counter)
    special_groups: Counter[str] = Counter()
    for group in groups.values():
        group_by_split_family[group.split][group.env_family] += 1
        group_by_split_visibility[group.split][group.visibility_mode] += 1
        group_by_split_source[group.split][group.source_label] += 1
        group_by_split_length[group.split][group.length_bucket] += 1
        if group.has_success:
            special_groups["success_groups"] += 1
        if group.has_information_gain:
            special_groups["information_gain_groups"] += 1
        if group.has_post_reveal_execution:
            special_groups["post_reveal_execution_groups"] += 1
        if group.has_post_setup_execution:
            special_groups["post_setup_execution_groups"] += 1
        if group.has_post_setup_positive_execution:
            special_groups["post_setup_positive_execution_groups"] += 1
        if group.has_post_setup_correct_action:
            special_groups["post_setup_correct_action_groups"] += 1
        if group.has_negative_progress:
            special_groups["negative_progress_groups"] += 1
    audit = {
        "path": str(feature_path),
        "rows": rows,
        "manifest_rows": manifest.get("rows"),
        "schema_version": manifest.get("schema_version"),
        "dataset_profile": manifest.get("dataset_profile", "raw"),
        "input_dim": manifest.get("input_dim"),
        "target_dim": manifest.get("target_dim"),
        "candidate_groups": len(groups),
        "rows_by_split_family": _counter_map(row_by_split_family),
        "rows_by_split_visibility": _counter_map(row_by_split_visibility),
        "rows_by_split_source": _counter_map(row_by_split_source),
        "rows_by_split_length": _counter_map(row_by_split_length),
        "candidate_groups_by_split_family": _counter_map(group_by_split_family),
        "candidate_groups_by_split_visibility": _counter_map(group_by_split_visibility),
        "candidate_groups_by_split_source": _counter_map(group_by_split_source),
        "candidate_groups_by_split_length": _counter_map(group_by_split_length),
        "target_counts": dict(target_counts),
        "post_setup_counts": dict(post_setup_counts),
        "special_candidate_groups": dict(special_groups),
        "top_envs_by_rows": row_by_env.most_common(20),
        "shortage_report": _audit_shortage_report(row_by_split_family, row_by_split_visibility, target_counts, rows),
    }
    return audit, groups, manifest


def _select_balanced_train_groups(
    groups: dict[str, CandidateGroupSummary],
    target_rows: int,
    seed: int,
    family_ratios: dict[str, float],
) -> tuple[set[str], dict[str, Any]]:
    rng = random.Random(seed)
    selected: set[str] = set()
    selected_rows = 0
    train_groups = [group for group in groups.values() if group.split == "train"]
    by_family: dict[str, list[CandidateGroupSummary]] = defaultdict(list)
    for group in train_groups:
        by_family[group.env_family].append(group)
    for family_groups in by_family.values():
        rng.shuffle(family_groups)
        family_groups.sort(key=_group_priority, reverse=True)
    family_report: dict[str, dict[str, Any]] = {}
    for family, ratio in family_ratios.items():
        quota = int(target_rows * ratio)
        rows_for_family = 0
        available_rows = sum(group.rows for group in by_family.get(family, []))
        for group in by_family.get(family, []):
            if rows_for_family >= quota:
                break
            selected.add(group.group_key)
            rows_for_family += group.rows
            selected_rows += group.rows
        family_report[family] = {
            "target_rows": quota,
            "available_rows": available_rows,
            "selected_rows": rows_for_family,
            "shortage_rows": max(0, quota - rows_for_family),
            "available_groups": len(by_family.get(family, [])),
        }
    if selected_rows < target_rows:
        leftovers = [group for group in train_groups if group.group_key not in selected]
        rng.shuffle(leftovers)
        leftovers.sort(key=_group_priority, reverse=True)
        for group in leftovers:
            if selected_rows >= target_rows:
                break
            selected.add(group.group_key)
            selected_rows += group.rows
    selected_groups = [groups[key] for key in selected]
    selection_report = {
        "target_train_rows": target_rows,
        "selected_train_rows": selected_rows,
        "selected_train_candidate_groups": len(selected),
        "family_report": family_report,
        "selected_rows_by_family": dict(_sum_group_rows(selected_groups, "env_family")),
        "selected_groups_by_family": dict(Counter(group.env_family for group in selected_groups)),
        "selected_rows_by_visibility": dict(_sum_group_rows(selected_groups, "visibility_mode")),
        "selected_groups_by_visibility": dict(Counter(group.visibility_mode for group in selected_groups)),
        "special_selected_groups": {
            "information_gain_groups": sum(1 for group in selected_groups if group.has_information_gain),
            "post_reveal_execution_groups": sum(1 for group in selected_groups if group.has_post_reveal_execution),
            "post_setup_execution_groups": sum(1 for group in selected_groups if group.has_post_setup_execution),
            "post_setup_positive_execution_groups": sum(1 for group in selected_groups if group.has_post_setup_positive_execution),
            "success_groups": sum(1 for group in selected_groups if group.has_success),
            "negative_progress_groups": sum(1 for group in selected_groups if group.has_negative_progress),
        },
        "shortage_report": _selection_shortage_report(train_groups, selected_groups, family_report),
    }
    return selected, selection_report


def _read_slot_manifest(feature_path: Path) -> dict[str, Any]:
    if not feature_path.exists():
        raise ValueError(f"missing slot feature file: {feature_path}")
    manifest_path = feature_path.with_suffix(feature_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"missing slot feature manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_merge_compatible(base: dict[str, Any], other: dict[str, Any]) -> None:
    keys = [
        "schema_version",
        "replay_schema_version",
        "env_families",
        "action_vocab",
        "max_slots",
        "slot_value_vocab",
        "global_feature_names",
        "slot_feature_names",
        "max_state_dim",
        "input_dim",
        "target_dim",
        "input_layout",
        "target_layout",
    ]
    for key in keys:
        if base.get(key) != other.get(key):
            raise ValueError(f"slot feature manifests are not merge-compatible: {key}")


def _state_dim_from_sample(sample: dict[str, Any], manifest: dict[str, Any]) -> int:
    layout = manifest["input_layout"]["state_padded"]
    start, end = int(layout[0]), int(layout[1])
    state_values = list(sample.get("input_vector", []))[start:end]
    for index in range(len(state_values) - 1, -1, -1):
        if abs(float(state_values[index])) > 1e-9:
            return index + 1
    return int(manifest.get("max_state_dim", len(state_values)))


def _target_indices(manifest: dict[str, Any]) -> dict[str, int]:
    layout = manifest.get("target_layout", {})
    required = ["local_success", "progress_delta", "information_gain"]
    missing = [name for name in required if name not in layout]
    if missing:
        raise ValueError(f"slot manifest missing target layout keys: {missing}")
    return {name: int(layout[name]) for name in required}


def _target_value(values: list[Any], index: int) -> float:
    if index < 0 or index >= len(values):
        return 0.0
    return float(values[index])


def _candidate_group_key(sample: dict[str, Any], line_no: int) -> str:
    split = str(sample.get("split", "unknown"))
    env_name = str(sample.get("env_name", "unknown"))
    candidate_set_id = sample.get("candidate_set_id") or sample.get("source", {}).get("candidate_set_id") or f"line:{line_no}"
    return f"{split}|{env_name}|{candidate_set_id}"


def _env_family(sample: dict[str, Any]) -> str:
    belief = sample.get("belief", {})
    if belief.get("env_family"):
        return str(belief["env_family"])
    return str(sample.get("env_name", "unknown")).split(".", 1)[0]


def _visibility_mode(sample: dict[str, Any]) -> str:
    visibility = sample.get("belief", {}).get("visibility_mode")
    if visibility:
        return str(visibility)
    env_name = str(sample.get("env_name", ""))
    if "partial" in env_name:
        return "partial_hidden"
    if "hidden" in env_name:
        return "hidden"
    return "visible"


def _source_label(sample: dict[str, Any]) -> str:
    source = sample.get("source", {})
    for key in ("source_label", "source_agent", "generator", "policy"):
        if source.get(key):
            return str(source[key])
    env_name = str(sample.get("env_name", ""))
    visibility = _visibility_mode(sample)
    family = _env_family(sample)
    if visibility != "visible":
        return "partial-hidden"
    if family == "lock" and ("canon" in env_name or "length" in env_name):
        return "lock-canonical"
    if family == "maze" and "bfs" in env_name:
        return "maze-bfs"
    if family == "tool" and ("seq" in env_name or "variant" in env_name):
        return "tool-variant"
    return "unknown"


def _length_bucket(sample: dict[str, Any]) -> str:
    length = sample.get("belief", {}).get("sequence_length")
    if length is None:
        return "unknown"
    return f"len_{int(length)}"


def _is_post_reveal_execution(sample: dict[str, Any]) -> bool:
    belief = sample.get("belief", {})
    observation = belief.get("last_observation", {})
    observed = observation.get("kind", "none") != "none"
    if not observed or not bool(belief.get("current_slot_known")):
        return False
    return not _is_information_action(str(sample.get("action_key", "")))


def _is_post_setup_execution(sample: dict[str, Any]) -> bool:
    source = sample.get("source", {})
    return bool(source.get("post_setup_execution", False))


def _is_post_setup_correct_action(sample: dict[str, Any]) -> bool:
    if not _is_post_setup_execution(sample):
        return False
    if str(sample.get("source", {}).get("setup_state", "")) != "post_setup_done":
        return False
    return _matches_current_required_action(str(sample.get("action_key", "")), sample.get("belief", {}))


def _matches_current_required_action(action_key: str, belief: dict[str, Any]) -> bool:
    required = str(belief.get("current_required_value", "unknown"))
    family = str(belief.get("env_family", ""))
    if required in {"", "unknown", "None"}:
        return False
    if family == "lock" and required.startswith("digit."):
        return action_key == f"press(digit={required.split('.', 1)[1]})"
    if family == "maze" and required.startswith("color."):
        return action_key == f'move(door="{required.split(".", 1)[1]}")'
    if family == "tool" and required.startswith("tool."):
        return action_key == required.split(".", 1)[1]
    return False


def _is_information_action(action_key: str) -> bool:
    return any(token in action_key for token in ("inspect", "search", "read", "observe", "clue"))


def _group_priority(group: CandidateGroupSummary) -> tuple[float, int]:
    score = 0.0
    if group.has_post_setup_correct_action:
        score += 140.0
    if group.has_post_setup_positive_execution:
        score += 130.0
    if group.has_post_setup_execution:
        score += 120.0
    if group.has_information_gain:
        score += 100.0
    if group.has_post_reveal_execution:
        score += 80.0
    if group.visibility_mode in {"partial_hidden", "hidden"}:
        score += 30.0
    if group.has_success:
        score += 10.0
    if group.has_positive_progress or group.has_negative_progress:
        score += 5.0
    return (score, -group.rows)


def _sum_group_rows(groups: list[CandidateGroupSummary], attr: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for group in groups:
        counter[str(getattr(group, attr))] += group.rows
    return counter


def _counter_map(values: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(counter) for key, counter in values.items()}


def _audit_shortage_report(
    rows_by_split_family: dict[str, Counter[str]],
    rows_by_split_visibility: dict[str, Counter[str]],
    target_counts: Counter[str],
    rows: int,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    train_family = rows_by_split_family.get("train", Counter())
    train_rows = sum(train_family.values())
    if train_rows:
        for family, minimum_ratio in {"maze": 0.2, "tool": 0.2}.items():
            actual = train_family.get(family, 0) / train_rows
            if actual < minimum_ratio:
                report.append({"kind": "family_underrepresented", "family": family, "actual_ratio": actual, "target_min_ratio": minimum_ratio})
    train_visibility = rows_by_split_visibility.get("train", Counter())
    hiddenish = train_visibility.get("partial_hidden", 0) + train_visibility.get("hidden", 0)
    if train_rows and hiddenish / train_rows < 0.2:
        report.append({"kind": "visibility_underrepresented", "visibility": "partial_or_hidden", "actual_ratio": hiddenish / train_rows, "target_min_ratio": 0.2})
    if rows and target_counts.get("information_gain_rows", 0) / rows < 0.05:
        report.append({"kind": "information_gain_underrepresented", "actual_ratio": target_counts.get("information_gain_rows", 0) / rows, "target_min_ratio": 0.05})
    if rows and target_counts.get("post_reveal_execution_rows", 0) / rows < 0.05:
        report.append({"kind": "post_reveal_execution_underrepresented", "actual_ratio": target_counts.get("post_reveal_execution_rows", 0) / rows, "target_min_ratio": 0.05})
    return report


def _selection_shortage_report(
    train_groups: list[CandidateGroupSummary],
    selected_groups: list[CandidateGroupSummary],
    family_report: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for family, item in family_report.items():
        if item["shortage_rows"] > 0:
            report.append({"kind": "family_quota_shortage", "family": family, **item})
    raw_info_groups = sum(1 for group in train_groups if group.has_information_gain)
    selected_info_groups = sum(1 for group in selected_groups if group.has_information_gain)
    if raw_info_groups < 100:
        report.append({"kind": "generator_coverage_shortage", "signal": "information_gain", "available_train_groups": raw_info_groups, "selected_train_groups": selected_info_groups})
    raw_post_reveal = sum(1 for group in train_groups if group.has_post_reveal_execution)
    selected_post_reveal = sum(1 for group in selected_groups if group.has_post_reveal_execution)
    if raw_post_reveal < 100:
        report.append({"kind": "generator_coverage_shortage", "signal": "post_reveal_execution", "available_train_groups": raw_post_reveal, "selected_train_groups": selected_post_reveal})
    return report


def _gate_ratio(numerator: int | float, denominator: int | float, threshold: float, *, mode: str) -> dict[str, Any]:
    ratio = float(numerator) / float(denominator) if denominator else 0.0
    if mode == "max":
        passed = ratio <= threshold
    elif mode == "min":
        passed = ratio >= threshold
    else:
        raise ValueError(mode)
    return {"actual": ratio, "threshold": threshold, "mode": mode, "passed": passed, "count": int(numerator), "total": int(denominator)}


def _onehot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def _pad(values: list[float], size: int) -> list[float]:
    if len(values) > size:
        raise ValueError(f"cannot pad vector of length {len(values)} to {size}")
    return [float(value) for value in values] + [0.0] * (size - len(values))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
