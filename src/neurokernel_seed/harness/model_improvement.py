from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ids import new_id
from .memory import HarnessMemory
from .work_service import WorkItemService


TERMINAL_WORK_STATUSES = {"rejected", "cancelled", "completed", "failed", "archived"}


@dataclass(frozen=True)
class ModelImprovementConfig:
    db_path: Path = Path("data/harness.db")
    interval_seconds: float = 300.0
    min_known_runtime_candidates: int = 100
    min_new_known_runtime_candidates: int = 50
    min_runtime_ranking_groups: int = 10
    target_runtime_top1: float = 0.65
    once: bool = False
    actor: str = "model_improvement_watchdog"


class ModelImprovementService:
    def __init__(self, *, db_path: str | Path, work_items: WorkItemService):
        self.db_path = Path(db_path)
        self.work_items = work_items

    def analyze(
        self,
        *,
        min_known_runtime_candidates: int = 100,
        min_new_known_runtime_candidates: int = 50,
        min_runtime_ranking_groups: int = 10,
        target_runtime_top1: float = 0.65,
    ) -> dict[str, Any]:
        with HarnessMemory(self.db_path) as memory:
            runtime = _runtime_snapshot(memory)
            open_training = _open_training_work(memory)
            latest_runtime_training = _latest_training_snapshot(memory, model_slot="runtime_action")
        previous_known = int((latest_runtime_training or {}).get("known_candidate_outcomes") or 0)
        known_delta = int(runtime["known_candidate_outcomes"]) - previous_known
        has_open_runtime = any(item["model_slot"] == "runtime_action" for item in open_training)
        runtime_ready = (
            int(runtime["known_candidate_outcomes"]) >= max(1, int(min_known_runtime_candidates))
            and int(runtime["ranking_evaluable_groups"]) >= max(0, int(min_runtime_ranking_groups))
            and (latest_runtime_training is None or known_delta >= max(1, int(min_new_known_runtime_candidates)))
            and not has_open_runtime
        )
        world_status = _world_training_status(runtime)
        return {
            "status": "completed",
            "schema_version": "neurokernel-model-improvement-analysis-v1",
            "runtime_action": {
                "ready_for_training_work": runtime_ready,
                "reason": _runtime_reason(
                    runtime,
                    latest_runtime_training=latest_runtime_training,
                    known_delta=known_delta,
                    has_open_runtime=has_open_runtime,
                    min_known_runtime_candidates=min_known_runtime_candidates,
                    min_new_known_runtime_candidates=min_new_known_runtime_candidates,
                    min_runtime_ranking_groups=min_runtime_ranking_groups,
                ),
                "snapshot": runtime,
                "previous_training_snapshot": latest_runtime_training,
                "known_candidate_delta": known_delta,
                "target_top1_action_accuracy": float(target_runtime_top1),
            },
            "world": world_status,
            "open_training_work": open_training,
        }

    def propose_training_work(
        self,
        *,
        min_known_runtime_candidates: int = 100,
        min_new_known_runtime_candidates: int = 50,
        min_runtime_ranking_groups: int = 10,
        target_runtime_top1: float = 0.65,
        actor: str = "model_improvement_watchdog",
    ) -> dict[str, Any]:
        analysis = self.analyze(
            min_known_runtime_candidates=min_known_runtime_candidates,
            min_new_known_runtime_candidates=min_new_known_runtime_candidates,
            min_runtime_ranking_groups=min_runtime_ranking_groups,
            target_runtime_top1=target_runtime_top1,
        )
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        runtime = analysis["runtime_action"]
        if runtime["ready_for_training_work"]:
            created.append(self._create_runtime_training_work(runtime, actor=actor))
        else:
            skipped.append({"model_slot": "runtime_action", "reason": runtime["reason"]})
        return {
            "status": "completed",
            "analysis": analysis,
            "created_count": len(created),
            "skipped_count": len(skipped),
            "created": created,
            "skipped": skipped,
            "actor": actor,
        }

    def _create_runtime_training_work(self, runtime: dict[str, Any], *, actor: str) -> dict[str, Any]:
        snapshot = runtime["snapshot"]
        target_top1 = float(runtime["target_top1_action_accuracy"])
        with HarnessMemory(self.db_path) as memory:
            duplicate = _find_open_training_work(memory, model_slot="runtime_action")
            if duplicate:
                return {"created": False, "reason": "open_training_work_exists", "work_item": duplicate}
            work = memory.create_work_item(
                work_id=new_id("work", "runtime_training"),
                work_type="training_pipeline",
                title="runtime_action 모델 재학습 후보",
                goal=(
                    "OrangePi에 실제로 누적된 runtime 행동 데이터를 노트북 CUDA 학습 파이프라인으로 재학습하고, "
                    "벤치에서 현재 모델보다 좋아진 경우에만 OrangePi runtime 슬롯에 배포한다."
                ),
                status="proposed",
                priority="high",
                risk_level="medium",
                linked_entity_type="model_improvement",
                linked_entity_id="runtime_action",
                route_reason=str(runtime["reason"]),
                confidence=0.82,
                metadata={
                    "execution_kind": "training_pipeline",
                    "model_slot": "runtime_action",
                    "source": "orangepi_accumulated_data",
                    "source_snapshot": snapshot,
                    "known_candidate_delta": runtime["known_candidate_delta"],
                    "quality_target": {"top1_action_accuracy": target_top1},
                    "nk_command": {
                        "action": "runtime-pipeline",
                        "args": [
                            "--source",
                            "edge",
                            "--device",
                            "cuda",
                            "--min-ranking-groups",
                            str(max(0, int(snapshot["ranking_evaluable_groups"]))),
                            "--min-top1-action-accuracy",
                            str(target_top1),
                        ],
                    },
                    "deliverables": ["runtime replay export", "runtime features", "CUDA training", "benchmark comparison", "winner-only deployment"],
                    "approval_boundary": "user_approval_required_before_training_worker_runs",
                    "guardrails": [
                        "use only accumulated OrangePi data",
                        "do not synthesize fake training rows",
                        "deploy only if benchmark says local candidate is better",
                    ],
                },
                actor=actor,
            )
            memory.add_work_note(
                str(work["work_id"]),
                actor=actor,
                note=(
                    "모델 자가개선 감시자가 runtime_action 재학습 후보를 만들었어. "
                    f"known={snapshot['known_candidate_outcomes']} ranking_groups={snapshot['ranking_evaluable_groups']}."
                ),
            )
        return {"created": True, "model_slot": "runtime_action", "work_item": work}


class ModelImprovementWatchdog:
    def __init__(self, *, config: ModelImprovementConfig, service: ModelImprovementService):
        self.config = config
        self.service = service

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"[model-improvement-watchdog] error={type(exc).__name__}: {exc}", flush=True)
            if self.config.once:
                return
            time.sleep(max(1.0, self.config.interval_seconds))

    def run_once(self) -> dict[str, Any]:
        result = self.service.propose_training_work(
            min_known_runtime_candidates=self.config.min_known_runtime_candidates,
            min_new_known_runtime_candidates=self.config.min_new_known_runtime_candidates,
            min_runtime_ranking_groups=self.config.min_runtime_ranking_groups,
            target_runtime_top1=self.config.target_runtime_top1,
            actor=self.config.actor,
        )
        print(
            f"[model-improvement-watchdog] created={result['created_count']} skipped={result['skipped_count']}",
            flush=True,
        )
        return result


def _runtime_snapshot(memory: HarnessMemory) -> dict[str, Any]:
    task_count = _count(memory, "SELECT COUNT(*) FROM tasks")
    decision_count = _count(memory, "SELECT COUNT(*) FROM action_decisions")
    execution_count = _count(memory, "SELECT COUNT(*) FROM execution_results")
    total_candidates = _count(memory, "SELECT COUNT(*) FROM experience_candidates")
    known_candidates = _count(memory, "SELECT COUNT(*) FROM experience_candidates WHERE execution_result_known=1")
    ranking_groups = _count(
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
    )
    partial_or_failed = _count(
        memory,
        "SELECT COUNT(*) FROM interaction_outcomes WHERE answer_quality IN ('partial', 'failed')",
    )
    return {
        "tasks": task_count,
        "decisions": decision_count,
        "executions": execution_count,
        "candidate_rows": total_candidates,
        "known_candidate_outcomes": known_candidates,
        "ranking_evaluable_groups": ranking_groups,
        "partial_or_failed_interactions": partial_or_failed,
    }


def _count(memory: HarnessMemory, query: str) -> int:
    row = memory.conn.execute(query).fetchone()
    return int(row[0] if row is not None else 0)


def _open_training_work(memory: HarnessMemory) -> list[dict[str, Any]]:
    rows = memory.list_work_items(limit=100, work_type="training_pipeline")
    result = []
    for row in rows:
        if str(row.get("status") or "") in TERMINAL_WORK_STATUSES:
            continue
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        result.append(
            {
                "work_id": row.get("work_id"),
                "status": row.get("status"),
                "model_slot": str(metadata.get("model_slot") or ""),
                "title": row.get("title"),
            }
        )
    return result


def _find_open_training_work(memory: HarnessMemory, *, model_slot: str) -> dict[str, Any] | None:
    for row in memory.list_work_items(limit=100, work_type="training_pipeline"):
        if str(row.get("status") or "") in TERMINAL_WORK_STATUSES:
            continue
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if str(metadata.get("model_slot") or "") == model_slot:
            return row
    return None


def _latest_training_snapshot(memory: HarnessMemory, *, model_slot: str) -> dict[str, Any] | None:
    for row in memory.list_work_items(limit=100, work_type="training_pipeline"):
        metadata = row.get("metadata_json") if isinstance(row.get("metadata_json"), dict) else {}
        if str(metadata.get("model_slot") or "") != model_slot:
            continue
        snapshot = metadata.get("source_snapshot")
        if isinstance(snapshot, dict):
            return snapshot
    return None


def _runtime_reason(
    runtime: dict[str, Any],
    *,
    latest_runtime_training: dict[str, Any] | None,
    known_delta: int,
    has_open_runtime: bool,
    min_known_runtime_candidates: int,
    min_new_known_runtime_candidates: int,
    min_runtime_ranking_groups: int,
) -> str:
    if has_open_runtime:
        return "open_runtime_training_work_exists"
    if int(runtime["known_candidate_outcomes"]) < int(min_known_runtime_candidates):
        return "not_enough_known_runtime_candidate_outcomes"
    if int(runtime["ranking_evaluable_groups"]) < int(min_runtime_ranking_groups):
        return "not_enough_ranking_evaluable_groups"
    if latest_runtime_training is not None and known_delta < int(min_new_known_runtime_candidates):
        return "not_enough_new_known_candidate_outcomes_since_last_training_work"
    return "runtime_training_work_ready"


def _world_training_status(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "ready_for_training_work": False,
        "reason": "real_world_transition_training_not_connected_yet",
        "observed_runtime_snapshot": runtime,
        "next_required_work": "connect real_world_transitions export into world model training before proposing world retraining",
    }
