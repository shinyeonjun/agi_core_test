from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .memory import HarnessMemory


MODEL_USAGE_SCHEMA_VERSION = "neurokernel-model-usage-audit-v1"


def inspect_model_usage(*, project_root: str | Path = ".", db_path: str | Path = "data/harness.db") -> dict[str, Any]:
    root = Path(project_root).resolve()
    db = Path(db_path)
    runtime_snapshot = _runtime_snapshot(db)
    slots = {
        "world": _world_slot(root),
        "runtime_action": _runtime_slot(root, runtime_snapshot),
    }
    connections = _code_connections(root)
    gaps = _usage_gaps(slots, connections, runtime_snapshot)
    return {
        "status": "completed",
        "schema_version": MODEL_USAGE_SCHEMA_VERSION,
        "model_roles": {
            "world": "future_state_risk_reward_predictor",
            "runtime_action": "safety_gated_action_ranker",
        },
        "slots": slots,
        "code_connections": connections,
        "runtime_data": runtime_snapshot,
        "gaps": gaps,
        "summary": _summary(slots, gaps),
    }


def _world_slot(root: Path) -> dict[str, Any]:
    model = root / "artifacts" / "current_world_model.onnx"
    manifest = root / "artifacts" / "current_world_model.manifest.json"
    return {
        "slot": "world",
        "active_model": _file_state(model),
        "manifest": _file_state(manifest),
        "intended_use": [
            "predict next state for compatible world/env features",
            "estimate risk and reward before action execution",
            "support multi-step rollout when a compatible state encoder exists",
        ],
        "currently_connected_to": [
            "world model training CLI",
            "ONNX export",
            "gate ablation benchmark",
            "hard heldout benchmark",
            "OrangePi current model benchmark/cache pipeline",
        ],
        "live_runtime_planner": {
            "usable": False,
            "reason": "no verified encoder from live TaskSpec/OrangePi state to world model feature schema",
            "required_before_use": [
                "real_world_transition feature schema",
                "live state encoder",
                "action-conditioned next-state target",
                "world rollout benchmark gate",
            ],
        },
    }


def _runtime_slot(root: Path, runtime_snapshot: dict[str, Any]) -> dict[str, Any]:
    model = root / "artifacts" / "current_runtime_action_model.pt"
    manifest = root / "artifacts" / "current_runtime_action_model.manifest.json"
    return {
        "slot": "runtime_action",
        "active_model": _file_state(model),
        "manifest": _file_state(manifest),
        "intended_use": [
            "rank safety-gated executable actions",
            "predict success/reward/duration/failure from accumulated runtime features",
            "feed candidate_set outcomes back into training data",
        ],
        "currently_connected_to": [
            "RuntimeActionPlanner",
            "HarnessService dry_run/run/probe_counterfactual_candidates",
            "runtime replay exporter",
            "runtime feature exporter",
            "runtime benchmark and winner-only deploy pipeline",
        ],
        "live_runtime_planner": {
            "usable": bool(_file_state(model)["exists"]),
            "reason": "model file present" if model.exists() else "current_runtime_action_model.pt is missing",
            "observed_training_data": runtime_snapshot,
        },
    }


def _code_connections(root: Path) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "runtime planner uses runtime_action model",
            "path": "src/neurokernel_seed/harness/runtime_planner.py",
            "symbols": ["RuntimeActionPlanner", "runtime_policy.rank"],
            "model_slot": "runtime_action",
            "purpose": "safety-gated action ranking",
        },
        {
            "name": "harness service records runtime model decisions",
            "path": "src/neurokernel_seed/harness/service.py",
            "symbols": ["decision_policy_name", "policy_decision_summary", "record_experience"],
            "model_slot": "runtime_action",
            "purpose": "experience logging for runtime training",
        },
        {
            "name": "counterfactual runtime candidates become ranking data",
            "path": "src/neurokernel_seed/harness/service.py",
            "symbols": ["probe_counterfactual_candidates", "_experience_counterfactual_candidates"],
            "model_slot": "runtime_action",
            "purpose": "candidate_set outcome collection",
        },
        {
            "name": "world model benchmark uses learned predictor",
            "path": "src/neurokernel_seed/cli.py",
            "symbols": ["OnnxWorldModelPredictor", "ActionGate", "run_gate_ablation"],
            "model_slot": "world",
            "purpose": "model-aware benchmark gate",
        },
        {
            "name": "worker harness documents world/runtime role split",
            "path": "docs/worker_harness/world_runtime_usage.md",
            "symbols": ["world 모델", "runtime_action 모델", "planner 연결 원칙"],
            "model_slot": "world_runtime_contract",
            "purpose": "implementation worker contract",
        },
    ]
    return [_connection_state(root, check) for check in checks]


def _connection_state(root: Path, check: dict[str, Any]) -> dict[str, Any]:
    path = root / str(check["path"])
    exists = path.exists()
    text = path.read_text(encoding="utf-8", errors="replace") if exists and path.is_file() else ""
    symbols = list(check["symbols"])
    present = [symbol for symbol in symbols if symbol in text]
    return {
        **check,
        "path_exists": exists,
        "symbols_present": present,
        "connected": exists and len(present) == len(symbols),
        "line_count": _python_line_count(path) if exists and path.suffix == ".py" else len(text.splitlines()) if exists else 0,
    }


def _runtime_snapshot(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "db_exists": False,
            "tasks": 0,
            "decisions": 0,
            "executions": 0,
            "candidate_rows": 0,
            "known_candidate_outcomes": 0,
            "ranking_evaluable_groups": 0,
        }
    with HarnessMemory(db_path) as memory:
        return {
            "db_exists": True,
            "tasks": _count(memory, "SELECT COUNT(*) FROM tasks"),
            "decisions": _count(memory, "SELECT COUNT(*) FROM action_decisions"),
            "executions": _count(memory, "SELECT COUNT(*) FROM execution_results"),
            "candidate_rows": _count(memory, "SELECT COUNT(*) FROM experience_candidates"),
            "known_candidate_outcomes": _count(memory, "SELECT COUNT(*) FROM experience_candidates WHERE execution_result_known=1"),
            "ranking_evaluable_groups": _count(
                memory,
                """
                SELECT COUNT(*) FROM (
                  SELECT experience_id
                  FROM experience_candidates
                  WHERE execution_result_known=1
                  GROUP BY experience_id
                  HAVING COUNT(*) >= 2
                )
                """,
            ),
        }


def _usage_gaps(slots: dict[str, Any], connections: list[dict[str, Any]], runtime_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    if not slots["runtime_action"]["active_model"]["exists"]:
        gaps.append({"slot": "runtime_action", "severity": "medium", "reason": "runtime_model_missing", "next_step": "train and deploy current_runtime_action_model.pt"})
    if int(runtime_snapshot.get("ranking_evaluable_groups") or 0) < 10:
        gaps.append({"slot": "runtime_action", "severity": "medium", "reason": "not_enough_runtime_ranking_groups", "next_step": "run more counterfactual readonly probes"})
    if not slots["world"]["live_runtime_planner"]["usable"]:
        gaps.append({"slot": "world", "severity": "high", "reason": "world_not_live_runtime_compatible", "next_step": "connect real_world_transitions to a verified world feature schema before live rollout"})
    for connection in connections:
        if not connection["connected"]:
            gaps.append({"slot": connection["model_slot"], "severity": "high", "reason": "code_connection_incomplete", "connection": connection["name"], "next_step": f"repair {connection['path']}"})
    return gaps


def _summary(slots: dict[str, Any], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "runtime_action_used_for_live_ranking": slots["runtime_action"]["live_runtime_planner"]["usable"],
        "world_used_for_live_rollout": slots["world"]["live_runtime_planner"]["usable"],
        "gap_count": len(gaps),
        "next_priority": gaps[0]["next_step"] if gaps else "keep collecting runtime candidate outcomes and benchmark model contribution",
    }


def _file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path), "exists": True, "size_bytes": stat.st_size}


def _count(memory: HarnessMemory, query: str) -> int:
    row = memory.conn.execute(query).fetchone()
    return int(row[0] if row is not None else 0)


def _python_line_count(path: Path) -> int:
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"))
    except SyntaxError:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return len(path.read_text(encoding="utf-8-sig").splitlines())
