from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .code_structure import inspect_code_structure
from .ids import new_id
from .improvement import ImprovementService
from .memory import HarnessMemory
from .model_improvement import ModelImprovementService
from .work_service import WorkItemService


TERMINAL_WORK_STATUSES = {"rejected", "cancelled", "completed", "failed", "archived"}


@dataclass(frozen=True)
class SelfImprovementWatchConfig:
    db_path: Path = Path("data/harness.db")
    project_root: Path = Path(".")
    interval_seconds: float = 180.0
    min_gap_count: int = 2
    lookback: int = 200
    code_paths: tuple[str, ...] = ("src", "tests")
    max_code_candidates: int = 5
    min_code_score: int = 60
    max_proposals_per_cycle: int = 3
    once: bool = False
    actor: str = "self_improvement_watchdog"


class SelfImprovementService:
    """Coordinate self-improvement signals into approval-gated work items.

    This service deliberately stops at proposing work. Development, tests, and
    activation stay behind the existing user-approval boundaries.
    """

    def __init__(
        self,
        *,
        db_path: str | Path,
        project_root: str | Path,
        improvements: ImprovementService,
        model_improvements: ModelImprovementService,
        work_items: WorkItemService,
    ):
        self.db_path = Path(db_path)
        self.project_root = Path(project_root)
        self.improvements = improvements
        self.model_improvements = model_improvements
        self.work_items = work_items

    def analyze(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        code_paths: tuple[str, ...] | list[str] = ("src", "tests"),
        max_code_candidates: int = 5,
        min_code_score: int = 60,
    ) -> dict[str, Any]:
        missing = self.improvements.analyze(min_gap_count=min_gap_count, lookback=lookback)
        model = self.model_improvements.analyze()
        pipeline = self.work_items.pipeline_status(limit=20, stale_after_seconds=300)
        memory = _memory_snapshot(self.db_path)
        code = inspect_code_structure(
            self.project_root,
            paths=list(code_paths),
            max_files=300,
            max_results=max(1, int(max_code_candidates)),
        )
        deficits = []
        deficits.extend(_missing_output_deficits(missing))
        deficits.extend(_model_deficits(model))
        deficits.extend(_pipeline_deficits(pipeline))
        deficits.extend(_code_structure_deficits(code, min_score=max(1, int(min_code_score))))
        readiness = _self_improvement_readiness(
            missing_outputs=missing,
            model_improvements=model,
            work_pipeline=pipeline,
            memory=memory,
        )
        deficits.extend(_readiness_deficits(readiness))
        deficits = _rank_deficits(deficits)
        return {
            "status": "completed",
            "schema_version": "neurokernel-self-improvement-analysis-v1",
            "autonomy_boundary": {
                "detect": "automatic",
                "propose": "automatic",
                "develop": "requires user approval on proposed work",
                "activate": "requires user approval after tests and patch review",
                "deploy_or_train": "requires explicit worker or NK training boundary",
            },
            "summary": {
                "deficit_count": len(deficits),
                "top_deficit": deficits[0] if deficits else None,
                "recommended_next": _recommended_next(deficits),
            },
            "deficits": deficits,
            "signals": {
                "missing_outputs": missing,
                "model_improvements": model,
                "work_pipeline": pipeline,
                "code_structure": code,
                "memory": memory,
                "readiness": readiness,
            },
        }

    def propose(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        code_paths: tuple[str, ...] | list[str] = ("src", "tests"),
        max_code_candidates: int = 5,
        min_code_score: int = 60,
        max_proposals_per_cycle: int = 3,
        actor: str = "self_improvement_watchdog",
    ) -> dict[str, Any]:
        analysis = self.analyze(
            min_gap_count=min_gap_count,
            lookback=lookback,
            code_paths=code_paths,
            max_code_candidates=max_code_candidates,
            min_code_score=min_code_score,
        )
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        # Reuse existing focused proposal services first. They already know how
        # to create capability proposals and training work without duplicates.
        missing_result = self.improvements.propose_missing_output_gaps(min_gap_count=min_gap_count, lookback=lookback, actor=actor)
        model_result = self.model_improvements.propose_training_work(actor=actor)
        created.extend(_created_items(missing_result))
        created.extend(_created_items(model_result))
        skipped.extend(_skipped_items(missing_result))
        skipped.extend(_skipped_items(model_result))

        remaining = max(0, int(max_proposals_per_cycle) - len(created))
        if remaining:
            for deficit in analysis["deficits"]:
                if deficit["kind"] not in {"code_structure", "world_training_connection", "work_pipeline_health"}:
                    continue
                result = self._create_deficit_work_item(deficit, actor=actor)
                if result.get("created"):
                    created.append(result)
                    remaining -= 1
                else:
                    skipped.append(result)
                if remaining <= 0:
                    break

        return {
            "status": "completed",
            "schema_version": "neurokernel-self-improvement-proposal-v1",
            "analysis": analysis,
            "created_count": len(created),
            "skipped_count": len(skipped),
            "created": created,
            "skipped": skipped,
            "actor": actor,
        }

    def _create_deficit_work_item(self, deficit: dict[str, Any], *, actor: str) -> dict[str, Any]:
        deficit_id = str(deficit["deficit_id"])
        with HarnessMemory(self.db_path) as memory:
            duplicate = _find_open_deficit_work(memory, deficit_id)
            if duplicate:
                return {"created": False, "kind": "duplicate", "deficit_id": deficit_id, "work_item": duplicate}
            work = memory.create_work_item(
                work_id=new_id("work", deficit_id),
                work_type="self_patch",
                title=str(deficit["title"]),
                goal=str(deficit["goal"]),
                status="proposed",
                priority=str(deficit.get("priority") or "medium"),
                risk_level=str(deficit.get("risk_level") or "low"),
                linked_entity_type="self_improvement_deficit",
                linked_entity_id=deficit_id,
                route_reason=str(deficit.get("reason") or ""),
                confidence=float(deficit.get("confidence") or 0.7),
                metadata={
                    "execution_kind": "self_improvement",
                    "deficit": deficit,
                    "deliverables": deficit.get("deliverables") or ["implementation_patch", "tests", "activation_candidate"],
                    "approval_boundary": "user_approval_required_before_development_worker_runs",
                    "guardrails": [
                        "do not fabricate metrics or training rows",
                        "do not bypass approval gates",
                        "preserve existing public APIs unless tests prove compatibility",
                    ],
                },
                actor=actor,
            )
            memory.add_work_note(
                str(work["work_id"]),
                actor=actor,
                note=f"자가개선 감지기가 부족 항목을 제안했습니다: {deficit['title']}",
            )
        return {"created": True, "kind": deficit["kind"], "deficit_id": deficit_id, "work_item": work}


class SelfImprovementWatchdog:
    def __init__(self, *, config: SelfImprovementWatchConfig, service: SelfImprovementService):
        self.config = config
        self.service = service

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except Exception as exc:
                print(f"[self-improvement-watchdog] error={type(exc).__name__}: {exc}", flush=True)
            if self.config.once:
                return
            time.sleep(max(1.0, self.config.interval_seconds))

    def run_once(self) -> dict[str, Any]:
        result = self.service.propose(
            min_gap_count=self.config.min_gap_count,
            lookback=self.config.lookback,
            code_paths=self.config.code_paths,
            max_code_candidates=self.config.max_code_candidates,
            min_code_score=self.config.min_code_score,
            max_proposals_per_cycle=self.config.max_proposals_per_cycle,
            actor=self.config.actor,
        )
        print(
            f"[self-improvement-watchdog] created={result['created_count']} skipped={result['skipped_count']}",
            flush=True,
        )
        return result


def _missing_output_deficits(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    deficits = []
    for gap in analysis.get("candidate_gaps") or []:
        output_key = str(gap.get("output_key") or "unknown_output")
        deficits.append(
            {
                "kind": "missing_output",
                "deficit_id": f"missing_output:{output_key}",
                "title": f"부족한 관측 능력 추가: {output_key}",
                "goal": f"Discord 요청에서 반복적으로 빠진 '{output_key}' 출력을 실제 OrangePi 상태에서 읽는 read-only 능력을 추가한다.",
                "reason": "repeated_missing_interaction_output",
                "severity": "high",
                "priority": "high",
                "risk_level": "low",
                "confidence": min(0.95, 0.6 + 0.1 * float(gap.get("count") or 1)),
                "evidence": gap,
                "score": 90 + int(gap.get("count") or 0),
            }
        )
    return deficits


def _model_deficits(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    deficits = []
    runtime = analysis.get("runtime_action") if isinstance(analysis.get("runtime_action"), dict) else {}
    if runtime.get("ready_for_training_work"):
        deficits.append(
            {
                "kind": "runtime_training",
                "deficit_id": "model:runtime_action:training_ready",
                "title": "runtime_action 모델 재학습",
                "goal": "누적된 OrangePi runtime 후보/결과 데이터로 runtime_action 모델을 재학습하고 벤치에서 이긴 경우에만 배포한다.",
                "reason": runtime.get("reason"),
                "severity": "high",
                "priority": "high",
                "risk_level": "medium",
                "confidence": 0.86,
                "evidence": runtime,
                "score": 88,
            }
        )
    world = analysis.get("world") if isinstance(analysis.get("world"), dict) else {}
    if world.get("reason") == "real_world_transition_training_not_connected_yet":
        deficits.append(
            {
                "kind": "world_training_connection",
                "deficit_id": "world:training_connection",
                "title": "world 모델 실사용 전이 학습 연결",
                "goal": "누적된 OrangePi 실사용 전이를 world 모델 학습 파이프라인에 연결하고, 벤치/게이트 후에만 후보 모델을 배포하도록 구현한다.",
                "reason": world.get("reason"),
                "severity": "high",
                "priority": "high",
                "risk_level": "medium",
                "confidence": 0.78,
                "evidence": world,
                "score": 84,
                "deliverables": ["world training data export", "world training proposal", "bench gate", "tests"],
            }
        )
    return deficits


def _pipeline_deficits(status: dict[str, Any]) -> list[dict[str, Any]]:
    summary = status.get("summary") if isinstance(status.get("summary"), dict) else {}
    attention = [str(item) for item in summary.get("attention") or []]
    queue = status.get("queue") if isinstance(status.get("queue"), dict) else {}
    deficits = []
    if queue.get("available") is False:
        deficits.append(
            {
                "kind": "work_pipeline_health",
                "deficit_id": "work_pipeline:queue_unavailable",
                "title": "자가개선 작업 큐 연결",
                "goal": "self_patch/external_work/training 작업이 승인 후 worker에 전달되도록 work queue 연결 상태를 진단하고 복구한다.",
                "reason": str(queue.get("reason") or "queue_unavailable"),
                "severity": "high",
                "priority": "high",
                "risk_level": "medium",
                "confidence": 0.8,
                "evidence": status,
                "score": 86,
            }
        )
    if attention:
        deficits.append(
            {
                "kind": "work_pipeline_health",
                "deficit_id": "work_pipeline:attention",
                "title": "자가개선 작업 파이프라인 복구",
                "goal": "stale/dead-letter/enqueue-failed 작업을 감지하고 재시도/보류/리뷰 경로를 명확히 하는 파이프라인 복구 기능을 보강한다.",
                "reason": ",".join(attention),
                "severity": "medium",
                "priority": "medium",
                "risk_level": "low",
                "confidence": 0.74,
                "evidence": status,
                "score": 70 + len(attention),
            }
        )
    return deficits


def _memory_snapshot(db_path: Path) -> dict[str, Any]:
    with HarnessMemory(db_path) as memory:
        messages = _count(memory, "SELECT COUNT(*) FROM conversation_messages")
        linked_messages = _count(memory, "SELECT COUNT(*) FROM conversation_messages WHERE linked_task_id IS NOT NULL AND linked_task_id != ''")
        interactions = _count(memory, "SELECT COUNT(*) FROM interaction_outcomes")
        known_candidates = _count(memory, "SELECT COUNT(*) FROM experience_candidates WHERE execution_result_known=1")
        total_candidates = _count(memory, "SELECT COUNT(*) FROM experience_candidates")
    return {
        "conversation_messages": messages,
        "linked_conversation_messages": linked_messages,
        "conversation_task_link_rate": round(linked_messages / messages, 4) if messages else 0.0,
        "interaction_outcomes": interactions,
        "known_candidate_outcomes": known_candidates,
        "candidate_rows": total_candidates,
    }


def _self_improvement_readiness(
    *,
    missing_outputs: dict[str, Any],
    model_improvements: dict[str, Any],
    work_pipeline: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    queue = work_pipeline.get("queue") if isinstance(work_pipeline.get("queue"), dict) else {}
    runtime = model_improvements.get("runtime_action") if isinstance(model_improvements.get("runtime_action"), dict) else {}
    runtime_snapshot = runtime.get("snapshot") if isinstance(runtime.get("snapshot"), dict) else {}
    criteria = [
        _criterion("deficit_detection", True, "자가 부족 분석 엔진이 응답했습니다.", {}),
        _criterion("approval_boundary", True, "개발/장착/학습은 승인 경계 뒤에 있습니다.", {}),
        _criterion("work_queue_available", bool(queue.get("available")), "승인된 작업을 worker 큐로 넘길 수 있어야 합니다.", queue),
        _criterion(
            "runtime_learning_data",
            int(runtime_snapshot.get("known_candidate_outcomes") or memory.get("known_candidate_outcomes") or 0) >= 100,
            "runtime_action 학습에 쓸 실제 후보 outcome이 충분히 쌓여야 합니다.",
            {"threshold": 100, "runtime_snapshot": runtime_snapshot, "memory": memory},
        ),
        _criterion(
            "conversation_task_linking",
            int(memory.get("conversation_messages") or 0) == 0 or float(memory.get("conversation_task_link_rate") or 0.0) >= 0.25,
            "Discord 대화가 task/outcome과 충분히 연결되어야 학습 데이터가 됩니다.",
            {"threshold": 0.25, "memory": memory},
        ),
        _criterion(
            "missing_output_feedback",
            missing_outputs.get("status") == "completed",
            "답변 누락/부분 실패가 capability gap으로 환류되어야 합니다.",
            missing_outputs,
        ),
        _criterion(
            "model_training_boundary",
            "runtime_action" in model_improvements and "world" in model_improvements,
            "모델 개선 판단은 runtime/world 슬롯을 구분해야 합니다.",
            model_improvements,
        ),
    ]
    passed = sum(1 for item in criteria if item["passed"])
    return {
        "schema_version": "neurokernel-self-improvement-readiness-v1",
        "score": round(passed / len(criteria), 4) if criteria else 0.0,
        "passed": passed,
        "total": len(criteria),
        "criteria": criteria,
    }


def _readiness_deficits(readiness: dict[str, Any]) -> list[dict[str, Any]]:
    deficits = []
    for item in readiness.get("criteria") or []:
        if not isinstance(item, dict) or item.get("passed"):
            continue
        criterion_id = str(item.get("id") or "unknown")
        if criterion_id == "work_queue_available":
            continue
        deficits.append(
            {
                "kind": "self_improvement_readiness",
                "deficit_id": f"readiness:{criterion_id}",
                "title": f"자가개선 readiness 보강: {criterion_id}",
                "goal": str(item.get("description") or "자가개선 readiness 실패 기준을 해결한다."),
                "reason": "readiness_criterion_failed",
                "severity": "high" if criterion_id in {"runtime_learning_data", "conversation_task_linking"} else "medium",
                "priority": "high" if criterion_id in {"runtime_learning_data", "conversation_task_linking"} else "medium",
                "risk_level": "low",
                "confidence": 0.76,
                "evidence": item,
                "score": 82,
                "deliverables": ["readiness_fix", "tests", "measured_before_after"],
            }
        )
    return deficits


def _criterion(criterion_id: str, passed: bool, description: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": criterion_id,
        "passed": bool(passed),
        "description": description,
        "evidence": evidence,
    }


def _count(memory: HarnessMemory, query: str) -> int:
    row = memory.conn.execute(query).fetchone()
    return int(row[0] if row is not None else 0)


def _code_structure_deficits(report: dict[str, Any], *, min_score: int) -> list[dict[str, Any]]:
    deficits = []
    for candidate in report.get("candidates") or []:
        score = int(candidate.get("score") or 0)
        if score < min_score:
            continue
        path = str(candidate.get("path") or "unknown")
        path_priority = 8 if path.startswith("src/") else 0
        deficits.append(
            {
                "kind": "code_structure",
                "deficit_id": f"code_structure:{path}",
                "title": f"코드 책임 분리: {path}",
                "goal": (
                    f"{path}의 긴 함수/큰 파일/과다 책임을 동작 변경 없이 작은 모듈로 분리하고, "
                    "기존 테스트와 공개 import 호환성을 유지한다."
                ),
                "reason": ",".join(str(item) for item in candidate.get("reasons") or []),
                "severity": "medium",
                "priority": "medium",
                "risk_level": "low",
                "confidence": 0.72,
                "evidence": candidate,
                "score": 60 + score + path_priority,
                "deliverables": ["refactor_patch", "regression_tests", "public_api_compatibility"],
            }
        )
    return deficits


def _rank_deficits(deficits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(deficits, key=lambda item: (int(item.get("score") or 0), str(item.get("deficit_id") or "")), reverse=True)


def _recommended_next(deficits: list[dict[str, Any]]) -> str:
    if not deficits:
        return "뚜렷한 신규 부족 항목은 없습니다. 관측 데이터를 계속 누적하세요."
    top = deficits[0]
    if top["kind"] in {"missing_output", "code_structure", "world_training_connection", "work_pipeline_health"}:
        return "상위 self_patch 후보를 Discord 승인 카드로 올리고, 승인 후 개발 워커가 구현/테스트하게 하세요."
    if top["kind"] == "runtime_training":
        return "runtime_action 학습 후보를 올리고, 노트북 CUDA 학습/벤치/승자 배포 경계를 사용하세요."
    return "상위 후보부터 승인 기반 작업으로 전환하세요."


def _find_open_deficit_work(memory: HarnessMemory, deficit_id: str) -> dict[str, Any] | None:
    for row in memory.list_work_items(limit=100):
        if str(row.get("status") or "") in TERMINAL_WORK_STATUSES:
            continue
        if str(row.get("linked_entity_type") or "") == "self_improvement_deficit" and str(row.get("linked_entity_id") or "") == deficit_id:
            return row
    return None


def _created_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in result.get("created") or [] if isinstance(item, dict)]


def _skipped_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in result.get("skipped") or [] if isinstance(item, dict)]


def serve_self_improvement_watchdog(
    *,
    config: SelfImprovementWatchConfig,
    service: SelfImprovementService,
) -> None:
    SelfImprovementWatchdog(
        config=SelfImprovementWatchConfig(
            db_path=config.db_path,
            project_root=config.project_root,
            interval_seconds=config.interval_seconds,
            min_gap_count=config.min_gap_count,
            lookback=config.lookback,
            code_paths=config.code_paths,
            max_code_candidates=config.max_code_candidates,
            min_code_score=config.min_code_score,
            max_proposals_per_cycle=config.max_proposals_per_cycle,
            once=config.once,
            actor=config.actor or f"self-improvement-watchdog-{socket.gethostname()}",
        ),
        service=service,
    ).run_forever()
