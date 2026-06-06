from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurokernel_seed.core.schema import Action
from neurokernel_seed.envs.registry import make_env
from neurokernel_seed.replay.validator import validate_replay

FEATURE_SCHEMA_VERSION = "neurokernel-flat-transition-v2"
COUNTERFACTUAL_FEATURE_SCHEMA_VERSION = "neurokernel-flat-transition-v4"
BASE_TASK_FEATURE_NAMES = [
    "lock.code.0",
    "lock.code.1",
    "lock.code.2",
    "lock.code.3",
    "lock.code_length",
    "maze.target.red",
    "maze.target.blue",
    "maze.target.green",
    "tool.sequence_length",
    "max_steps",
]
VISIBILITY_MODES = ("visible", "partial_hidden", "hidden")
OBSERVATION_TYPES = ("none", "inspect", "search", "read", "observe_panel", "read_clue")
MAX_SEQUENCE_SLOTS = 5
V4_STATE_FEATURE_NAMES = [
    *[f"visibility_mode.{mode}" for mode in VISIBILITY_MODES],
    *[f"known_mask.{index}" for index in range(MAX_SEQUENCE_SLOTS)],
    *[f"unknown_mask.{index}" for index in range(MAX_SEQUENCE_SLOTS)],
    "revealed_slots_count",
    "unknown_slots_count",
    "current_slot_known",
    *[f"last_observation_type.{name}" for name in OBSERVATION_TYPES],
    "last_observation_slot",
    "last_observation_value",
    "current_index",
    "sequence_length",
    "remaining_steps",
    "remaining_sequence_slots",
    "progress_fraction",
    "is_last_step_index",
    "can_finish",
    "can_summarize",
    "current_required_action.unknown",
    "action.meta.reveals_information",
    "action.meta.requires_known_slot",
    "action.meta.terminal_only",
    "action.meta.consumes_progress_step",
]


@dataclass(frozen=True)
class FlatEncoderManifest:
    schema_version: str
    replay_schema_version: str
    envs: list[str]
    env_families: list[str]
    splits: list[str]
    action_vocab: list[str]
    task_feature_names: list[str]
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
            "task_feature_names": self.task_feature_names,
            "max_state_dim": self.max_state_dim,
            "state_dim_by_env": self.state_dim_by_env,
            "input_dim": self.input_dim,
            "target_dim": self.target_dim,
            "input_layout": self.input_layout,
            "target_layout": self.target_layout,
        }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"no records found: {path}")
    schema_version = str(records[0].get("schema_version", ""))
    if schema_version == "neurokernel-replay-v2":
        validate_replay(path)
    elif schema_version == "neurokernel-counterfactual-transition-v1":
        from neurokernel_seed.replay.counterfactual import validate_counterfactual

        validate_counterfactual(path)
    else:
        raise ValueError(f"unsupported source schema: {schema_version}")
    return records


def inspect_replay_features(path: str | Path) -> dict[str, Any]:
    records = load_records(path)
    manifest = build_manifest(records)
    return {"rows": len(records), **manifest.as_dict()}


def export_flat_features(replay_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    records = load_records(replay_path)
    manifest = build_manifest(records)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(records):
            sample = encode_record(record, manifest, row_index)
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps({"rows": len(records), **manifest.as_dict()}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {"rows": len(records), "out": str(out), "manifest": str(manifest_path), **manifest.as_dict()}


def validate_flat_features(path: str | Path) -> dict[str, Any]:
    feature_path = Path(path)
    manifest_path = feature_path.with_suffix(feature_path.suffix + ".manifest.json")
    if not feature_path.exists():
        raise ValueError(f"missing feature file: {feature_path}")
    if not manifest_path.exists():
        raise ValueError(f"missing feature manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = 0
    with feature_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            _validate_sample(sample, manifest, line_no)
            rows += 1
    if rows != manifest.get("rows"):
        raise ValueError(f"feature row count mismatch: data={rows}, manifest={manifest.get('rows')}")
    return {"rows": rows, "schema_version": manifest["schema_version"], "input_dim": manifest["input_dim"], "target_dim": manifest["target_dim"]}


def build_manifest(records: list[dict[str, Any]]) -> FlatEncoderManifest:
    if not records:
        raise ValueError("cannot build feature manifest from empty replay")
    source_schema = str(records[0]["schema_version"])
    envs = sorted({record["env_name"] for record in records})
    splits = sorted({record["split"] for record in records if record.get("split")})
    state_dim_by_env: dict[str, int] = {}
    for record in records:
        env_name = record["env_name"]
        dim = len(record["state_vector"])
        previous = state_dim_by_env.setdefault(env_name, dim)
        if previous != dim:
            raise ValueError(f"state vector dimension changed for {env_name}: {previous} -> {dim}")
    max_state_dim = max(state_dim_by_env.values())
    action_vocab = _build_action_vocab(records, envs)
    env_families = sorted({_env_family(env_name) for env_name in envs})
    task_feature_names = BASE_TASK_FEATURE_NAMES + [f"tool.order.{action_key}" for action_key in action_vocab]
    if source_schema == "neurokernel-counterfactual-transition-v1":
        task_feature_names = task_feature_names + [f"current_required_action.{action_key}" for action_key in action_vocab] + V4_STATE_FEATURE_NAMES
    env_family_end = len(env_families)
    task_end = env_family_end + len(task_feature_names)
    state_end = task_end + max_state_dim
    input_layout = {
        "env_family_onehot": [0, env_family_end],
        "task_config": [env_family_end, task_end],
        "state_padded": [task_end, state_end],
        "action_onehot": [state_end, state_end + len(action_vocab)],
    }
    if source_schema == "neurokernel-counterfactual-transition-v1":
        schema_version = COUNTERFACTUAL_FEATURE_SCHEMA_VERSION
        target_layout = {
            "next_state_padded": [0, max_state_dim],
            "reward": max_state_dim,
            "done": max_state_dim + 1,
            "local_success": max_state_dim + 2,
            "progress_delta": max_state_dim + 3,
            "information_gain": max_state_dim + 4,
        }
        target_dim = max_state_dim + 5
    else:
        schema_version = FEATURE_SCHEMA_VERSION
        target_layout = {
            "next_state_padded": [0, max_state_dim],
            "reward": max_state_dim,
            "done": max_state_dim + 1,
            "success": max_state_dim + 2,
        }
        target_dim = max_state_dim + 3
    return FlatEncoderManifest(
        schema_version=schema_version,
        replay_schema_version=source_schema,
        envs=envs,
        env_families=env_families,
        splits=splits,
        action_vocab=action_vocab,
        task_feature_names=task_feature_names,
        max_state_dim=max_state_dim,
        state_dim_by_env=state_dim_by_env,
        input_dim=len(env_families) + len(task_feature_names) + max_state_dim + len(action_vocab),
        target_dim=target_dim,
        input_layout=input_layout,
        target_layout=target_layout,
    )


def encode_record(record: dict[str, Any], manifest: FlatEncoderManifest, row_index: int = 0) -> dict[str, Any]:
    source_schema = str(record["schema_version"])
    action = record["candidate_action"] if source_schema == "neurokernel-counterfactual-transition-v1" else record["action"]
    state_facts = record.get("state_facts", {})
    next_state = record["next_state_vector"] if source_schema == "neurokernel-counterfactual-transition-v1" else record["actual_next_state_vector"]
    action_key = canonical_action_key(action)
    if action_key not in manifest.action_vocab:
        raise ValueError(f"action not in vocabulary: {action_key}")
    state_vector = _pad(record["state_vector"], manifest.max_state_dim)
    next_state_vector = _pad(next_state, manifest.max_state_dim)
    state_mask = [1.0] * len(record["state_vector"]) + [0.0] * (manifest.max_state_dim - len(record["state_vector"]))
    input_vector = encode_input_vector(record["env_name"], state_facts, record["state_vector"], action, manifest.as_dict())
    if manifest.schema_version == COUNTERFACTUAL_FEATURE_SCHEMA_VERSION:
        target_vector = next_state_vector + [
            float(record["reward"]),
            1.0 if record["done"] else 0.0,
            float(record["local_success"]),
            float(record["progress_delta"]),
            float(record["information_gain"]),
        ]
        target_mask = state_mask + [1.0, 1.0, 1.0, 1.0, 1.0]
        source = {
            "restore_id": record["restore_id"],
            "agent_state_id": record["agent_state_id"],
            "candidate_set_id": record["candidate_set_id"],
            "actual_action_score": float(record["actual_action_score"]),
        }
    else:
        target_vector = next_state_vector + [float(record["reward"]), 1.0 if record["done"] else 0.0, _record_success(record)]
        target_mask = state_mask + [1.0, 1.0, 1.0]
        source = {"prev_state_id": record["prev_state_id"], "next_state_id": record["next_state_id"]}
    return {
        "schema_version": manifest.schema_version,
        "row_index": row_index,
        "episode_id": record.get("episode_id", record.get("world_id", "")),
        "step_index": record.get("step_index", record.get("source_step_index", 0)),
        "env_name": record["env_name"],
        "split": record["split"],
        "action_key": action_key,
        "candidate_set_id": record.get("candidate_set_id"),
        "actual_action_score": record.get("actual_action_score"),
        "input_vector": input_vector,
        "target_vector": target_vector,
        "target_mask": target_mask,
        "source": source,
    }


def canonical_action_key(action: dict[str, Any] | Action) -> str:
    if isinstance(action, Action):
        name = action.name
        params = action.params
    else:
        name = str(action["name"])
        params = dict(action.get("params", {}))
    if not params:
        return name
    encoded_params = ",".join(f"{key}={json.dumps(params[key], sort_keys=True)}" for key in sorted(params))
    return f"{name}({encoded_params})"


def encode_input_vector(env_name: str, state_facts: dict[str, Any], state_vector: list[float] | tuple[float, ...], action: dict[str, Any] | Action, manifest: dict[str, Any]) -> list[float]:
    if manifest.get("schema_version") in {"neurokernel-slot-transition-v1", "neurokernel-slot-transition-v2"}:
        from neurokernel_seed.replay.slot_dataset import encode_slot_input_vector

        return encode_slot_input_vector(env_name, state_facts, state_vector, action, manifest)

    action_key = canonical_action_key(action)
    action_vocab = list(manifest["action_vocab"])
    if action_key not in action_vocab:
        raise ValueError(f"action not in model manifest: {action_key}")
    max_state_dim = int(manifest["max_state_dim"])
    padded_state = _pad(list(state_vector), max_state_dim)
    action_vector = _onehot(action_vocab.index(action_key), len(action_vocab))

    if manifest["schema_version"] == "neurokernel-flat-transition-v1":
        envs = list(manifest["envs"])
        if env_name not in envs:
            raise ValueError(f"env not in model manifest: {env_name}")
        return _onehot(envs.index(env_name), len(envs)) + padded_state + action_vector

    env_families = list(manifest["env_families"])
    family = _env_family(env_name)
    if family not in env_families:
        raise ValueError(f"env family not in model manifest: {family}")
    env_family_vector = _onehot(env_families.index(family), len(env_families))
    task_vector = _task_config_vector(family, state_facts, action, manifest)
    return env_family_vector + task_vector + padded_state + action_vector


def _build_action_vocab(records: list[dict[str, Any]], envs: list[str]) -> list[str]:
    action_keys = {canonical_action_key(record["candidate_action"] if record["schema_version"] == "neurokernel-counterfactual-transition-v1" else record["action"]) for record in records}
    for env_name in envs:
        try:
            env = make_env(env_name)
        except KeyError:
            continue
        state = env.reset(0)
        action_keys.update(canonical_action_key(action) for action in env.candidate_actions(state))
    return sorted(action_keys)


def _env_family(env_name: str) -> str:
    return env_name.split(".", 1)[0]


def _record_success(record: dict[str, Any]) -> float:
    info = record.get("info") or {}
    if "success" in info:
        return 1.0 if bool(info["success"]) else 0.0
    if record.get("done") and float(record.get("reward", 0.0)) > 0:
        return 1.0
    return 0.0


def _task_config_vector(family: str, facts: dict[str, Any], action: dict[str, Any] | Action, manifest: dict[str, Any]) -> list[float]:
    names = list(manifest["task_feature_names"])
    values = {name: 0.0 for name in names}
    schema_version = str(manifest.get("schema_version", ""))
    max_steps = float(facts.get("max_steps") or 0.0)
    if "max_steps" in values:
        values["max_steps"] = max(0.0, min(1.0, max_steps / 10.0))

    if family == "lock":
        code = list((facts.get("visible_code") if schema_version == COUNTERFACTUAL_FEATURE_SCHEMA_VERSION else None) or facts.get("code") or [])
        for index in range(4):
            if index < len(code):
                values[f"lock.code.{index}"] = max(0.0, min(1.0, float(code[index]) / 4.0))
        values["lock.code_length"] = max(0.0, min(1.0, float(len(code)) / 4.0))
    elif family == "maze":
        target = str((facts.get("visible_target_color") if schema_version == COUNTERFACTUAL_FEATURE_SCHEMA_VERSION else None) or facts.get("target_color") or "")
        key = f"maze.target.{target}"
        if key in values:
            values[key] = 1.0
    elif family == "tool":
        sequence = [str(item) for item in ((facts.get("visible_sequence") if schema_version == COUNTERFACTUAL_FEATURE_SCHEMA_VERSION else None) or facts.get("sequence") or [])]
        values["tool.sequence_length"] = max(0.0, min(1.0, float(len(sequence)) / 8.0))
        sequence_len = max(1, len(sequence))
        action_vocab = list(manifest["action_vocab"])
        for action_key in action_vocab:
            action_name = action_key.split("(", 1)[0]
            if action_name in sequence:
                values[f"tool.order.{action_key}"] = float(sequence.index(action_name) + 1) / float(sequence_len)
    if schema_version == COUNTERFACTUAL_FEATURE_SCHEMA_VERSION:
        _add_v4_features(values, facts, action, manifest)
    return [float(values[name]) for name in names]


def _add_v4_features(values: dict[str, float], facts: dict[str, Any], action: dict[str, Any] | Action, manifest: dict[str, Any]) -> None:
    visibility_mode = str(facts.get("visibility_mode") or "visible")
    if f"visibility_mode.{visibility_mode}" in values:
        values[f"visibility_mode.{visibility_mode}"] = 1.0
    known_mask = [float(value) for value in list(facts.get("known_mask") or [])[:MAX_SEQUENCE_SLOTS]]
    unknown_mask = [float(value) for value in list(facts.get("unknown_mask") or [])[:MAX_SEQUENCE_SLOTS]]
    for index in range(MAX_SEQUENCE_SLOTS):
        values[f"known_mask.{index}"] = known_mask[index] if index < len(known_mask) else 0.0
        values[f"unknown_mask.{index}"] = unknown_mask[index] if index < len(unknown_mask) else 0.0
    sequence_length = float(facts.get("sequence_length") or len(facts.get("code") or facts.get("sequence") or []) or 1)
    current_index = float(facts.get("current_index", facts.get("progress", facts.get("stage", 0))) or 0)
    max_steps = float(facts.get("max_steps") or 1.0)
    remaining_steps = float(facts.get("remaining_steps", max_steps - float(facts.get("steps", 0) or 0)))
    remaining_slots = float(facts.get("remaining_sequence_slots", max(0.0, sequence_length - current_index)))
    values["revealed_slots_count"] = _clip01(float(facts.get("revealed_slots_count", sum(known_mask))) / max(1.0, sequence_length))
    values["unknown_slots_count"] = _clip01(float(facts.get("unknown_slots_count", sum(unknown_mask))) / max(1.0, sequence_length))
    values["current_slot_known"] = 1.0 if bool(facts.get("current_slot_known", True)) else 0.0
    observation_type = str(facts.get("last_observation_type", "none"))
    if f"last_observation_type.{observation_type}" in values:
        values[f"last_observation_type.{observation_type}"] = 1.0
    values["last_observation_slot"] = _clip01((float(facts.get("last_observation_slot", -1)) + 1.0) / max(1.0, sequence_length + 1.0))
    values["last_observation_value"] = _encode_observation_value(facts.get("last_observation_value"))
    values["current_index"] = _clip01(current_index / max(1.0, sequence_length))
    values["sequence_length"] = _clip01(sequence_length / float(MAX_SEQUENCE_SLOTS))
    values["remaining_steps"] = _clip01(remaining_steps / max(1.0, max_steps))
    values["remaining_sequence_slots"] = _clip01(remaining_slots / max(1.0, sequence_length))
    values["progress_fraction"] = _clip01(float(facts.get("progress_fraction", current_index / max(1.0, sequence_length))))
    values["is_last_step_index"] = 1.0 if bool(facts.get("is_last_step_index", False)) else 0.0
    values["can_finish"] = 1.0 if bool(facts.get("can_finish", False)) else 0.0
    values["can_summarize"] = 1.0 if bool(facts.get("can_summarize", False)) else 0.0
    current_required = str(facts.get("current_required_action") or "unknown")
    action_vocab = list(manifest["action_vocab"])
    if values["current_slot_known"] <= 0.0:
        current_required = "unknown"
    matched = False
    for action_key in action_vocab:
        if action_key.split("(", 1)[0] == current_required:
            values[f"current_required_action.{action_key}"] = 1.0
            matched = True
    if not matched:
        values["current_required_action.unknown"] = 1.0
    metadata = _action_metadata(action)
    values["action.meta.reveals_information"] = 1.0 if metadata["reveals_information"] else 0.0
    values["action.meta.requires_known_slot"] = 1.0 if metadata["requires_known_slot"] else 0.0
    values["action.meta.terminal_only"] = 1.0 if metadata["terminal_only"] else 0.0
    values["action.meta.consumes_progress_step"] = 1.0 if metadata["consumes_progress_step"] else 0.0


def _action_metadata(action: dict[str, Any] | Action) -> dict[str, bool]:
    name = action.name if isinstance(action, Action) else str(action["name"])
    return {
        "reveals_information": name in {"inspect", "search", "read", "observe_panel", "read_clue"},
        "requires_known_slot": name in {"press", "move", "use", "summarize"},
        "terminal_only": name in {"summarize", "move"},
        "consumes_progress_step": name not in {"inspect", "observe_panel", "read_clue"},
    }


def _encode_observation_value(value: Any) -> float:
    if value in (None, "", "unknown"):
        return 0.0
    if isinstance(value, (int, float)):
        return _clip01(float(value) / 4.0)
    text = str(value)
    if text in {"red", "search"}:
        return 0.25
    if text in {"blue", "read"}:
        return 0.5
    if text in {"green", "summarize"}:
        return 0.75
    return 1.0


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _onehot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values


def _pad(values: list[float], size: int) -> list[float]:
    if len(values) > size:
        raise ValueError(f"cannot pad vector of length {len(values)} to {size}")
    return [float(value) for value in values] + [0.0] * (size - len(values))


def _validate_sample(sample: dict[str, Any], manifest: dict[str, Any], line_no: int) -> None:
    required = ["schema_version", "input_vector", "target_vector", "target_mask", "env_name", "split", "action_key"]
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
