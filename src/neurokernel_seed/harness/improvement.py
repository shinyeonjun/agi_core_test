from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_catalog import ActionDefinition
from .capability_service import CapabilityProposalService
from .interaction_contract import ACTION_OUTPUTS
from .memory import HarnessMemory
from .work_service import WorkItemService


@dataclass(frozen=True)
class ImprovementWatchConfig:
    db_path: Path = Path("data/harness.db")
    interval_seconds: float = 120.0
    min_gap_count: int = 2
    lookback: int = 200
    once: bool = False
    actor: str = "improvement_watchdog"


@dataclass(frozen=True)
class MissingOutputGap:
    output_key: str
    count: int
    sample_request: str
    sample_response: str
    user_id: str | None
    channel_id: str | None
    task_id: str | None


class ImprovementAnalyzer:
    def __init__(self, *, db_path: str | Path, catalog: dict[str, ActionDefinition]):
        self.db_path = Path(db_path)
        self.catalog = catalog

    def analyze_missing_outputs(self, *, min_count: int = 2, lookback: int = 200) -> dict[str, Any]:
        covered = covered_outputs(self.catalog)
        groups = self._missing_output_groups(lookback=max(1, lookback))
        candidates = [
            gap
            for gap in groups
            if gap.count >= max(1, min_count) and gap.output_key not in covered
        ]
        return {
            "status": "completed",
            "covered_outputs": sorted(covered),
            "candidate_gaps": [gap_to_dict(gap) for gap in candidates],
            "ignored_gaps": [gap_to_dict(gap) for gap in groups if gap.output_key in covered or gap.count < max(1, min_count)],
        }

    def _missing_output_groups(self, *, lookback: int) -> list[MissingOutputGap]:
        query = """
            SELECT *
            FROM interaction_outcomes
            WHERE missing_outputs_json IS NOT NULL
              AND missing_outputs_json != '[]'
              AND answer_quality IN ('partial', 'failed')
            ORDER BY id DESC
            LIMIT ?
        """
        grouped: dict[str, MissingOutputGap] = {}
        with HarnessMemory(self.db_path) as memory:
            rows = memory.conn.execute(query, (lookback,)).fetchall()
            for row in rows:
                item = dict(row)
                missing = _json_list(item.get("missing_outputs_json"))
                for output_key in missing:
                    previous = grouped.get(output_key)
                    if previous is None:
                        grouped[output_key] = MissingOutputGap(
                            output_key=output_key,
                            count=1,
                            sample_request=str(item.get("request_text_redacted") or ""),
                            sample_response=str(item.get("response_text_redacted") or ""),
                            user_id=str(item.get("user_id") or "") or None,
                            channel_id=str(item.get("channel_id") or "") or None,
                            task_id=str(item.get("task_id") or "") or None,
                        )
                    else:
                        grouped[output_key] = MissingOutputGap(
                            output_key=previous.output_key,
                            count=previous.count + 1,
                            sample_request=previous.sample_request,
                            sample_response=previous.sample_response,
                            user_id=previous.user_id,
                            channel_id=previous.channel_id,
                            task_id=previous.task_id,
                        )
        return sorted(grouped.values(), key=lambda gap: (-gap.count, gap.output_key))


class ImprovementService:
    def __init__(
        self,
        *,
        db_path: str | Path,
        catalog: dict[str, ActionDefinition],
        work_items: WorkItemService,
    ):
        self.db_path = Path(db_path)
        self.catalog = catalog
        self.work_items = work_items

    def analyze(self, *, min_gap_count: int = 2, lookback: int = 200) -> dict[str, Any]:
        return ImprovementAnalyzer(db_path=self.db_path, catalog=self.catalog).analyze_missing_outputs(
            min_count=min_gap_count,
            lookback=lookback,
        )

    def propose_missing_output_gaps(
        self,
        *,
        min_gap_count: int = 2,
        lookback: int = 200,
        actor: str = "improvement_watchdog",
    ) -> dict[str, Any]:
        analysis = self.analyze(min_gap_count=min_gap_count, lookback=lookback)
        created: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        capability = CapabilityProposalService(db_path=self.db_path, catalog=self.catalog, work_items=self.work_items)
        for raw_gap in analysis["candidate_gaps"]:
            intent = capability_intent_for_missing_output(raw_gap)
            result = capability.create_from_intent(
                user_text=str(raw_gap.get("sample_request") or ""),
                capability_intent=intent,
                user_id=raw_gap.get("user_id"),
                channel_id=raw_gap.get("channel_id"),
                language_intent={"source": "improvement_watchdog", "missing_output": raw_gap},
            )
            if result.get("created"):
                created.append(result)
            else:
                skipped.append(result)
        return {
            "status": "completed",
            "analysis": analysis,
            "created_count": len(created),
            "skipped_count": len(skipped),
            "created": created,
            "skipped": skipped,
            "actor": actor,
        }


class ImprovementWatchdog:
    def __init__(
        self,
        *,
        config: ImprovementWatchConfig,
        service: ImprovementService,
    ):
        self.config = config
        self.service = service

    def run_forever(self) -> None:
        while True:
            self.run_once()
            if self.config.once:
                return
            time.sleep(max(1.0, self.config.interval_seconds))

    def run_once(self) -> dict[str, Any]:
        result = self.service.propose_missing_output_gaps(
            min_gap_count=self.config.min_gap_count,
            lookback=self.config.lookback,
            actor=self.config.actor,
        )
        print(
            f"[improvement-watchdog] created={result['created_count']} skipped={result['skipped_count']}",
            flush=True,
        )
        return result


def covered_outputs(catalog: dict[str, ActionDefinition]) -> set[str]:
    return {
        output
        for action_id in catalog
        for output in ACTION_OUTPUTS.get(action_id, ())
    }


def gap_to_dict(gap: MissingOutputGap) -> dict[str, Any]:
    return {
        "output_key": gap.output_key,
        "count": gap.count,
        "sample_request": gap.sample_request,
        "sample_response": gap.sample_response,
        "user_id": gap.user_id,
        "channel_id": gap.channel_id,
        "task_id": gap.task_id,
    }


def capability_intent_for_missing_output(gap: dict[str, Any]) -> dict[str, Any]:
    output_key = safe_output_key(str(gap.get("output_key") or "unknown_output"))
    action_id = f"get_{output_key}"
    label = output_key.replace("_", " ")
    return {
        "kind": "gap",
        "reply": f"I found repeated missing output: {label}. I can propose a read-only capability for approval.",
        "gap": {
            "gap_type": "missing_output",
            "requested_capability": f"Provide {label} from real accumulated system state",
            "normalized_request": f"Add read-only capability for missing output {output_key}",
            "matched_existing_actions": [],
            "confidence": min(0.95, 0.55 + 0.1 * float(gap.get("count") or 1)),
        },
        "proposal": {
            "action_id": action_id,
            "capability_name": f"Read {label}",
            "purpose": (
                f"Read the missing output '{output_key}' from the live OrangePi project state "
                "without mutating files, services, secrets, or deployment state."
            ),
            "target": "orangepi5",
            "risk_level": "low",
            "side_effect": False,
            "requires_approval": False,
            "inputs": {"type": "object", "fields": [], "required": []},
            "outputs": {
                "type": "object",
                "fields": [{"name": output_key, "type": "object", "description": f"Observed {label}"}],
                "required": [output_key],
            },
            "implementation_hint": {
                "executor": "readonly_system",
                "notes": (
                    "Implement from existing project state and manifests only. "
                    "Do not return placeholder data, do not use fallback claims, and add smoke tests."
                ),
            },
            "test_plan": [
                {"name": "returns_observed_output", "type": "unit", "assertions": [f"result contains {output_key}"]},
                {"name": "read_only", "type": "safety", "assertions": ["no writes", "no sudo", "no network secrets"]},
            ],
            "safety_notes": ["read-only proposal", "requires user approval before implementation"],
            "confidence": min(0.95, 0.55 + 0.1 * float(gap.get("count") or 1)),
            "approval_required_for_implementation": True,
            "activation_requires_tests": True,
        },
        "confidence": min(0.95, 0.55 + 0.1 * float(gap.get("count") or 1)),
        "requires_confirmation": False,
        "clarifying_question": None,
        "safety_notes": [],
    }


def safe_output_key(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value.lower()).strip("_")
    return cleaned or "unknown_output"


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    try:
        import json

        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def serve_improvement_watchdog(
    *,
    db_path: str | Path = "data/harness.db",
    project_root: str | Path = ".",
    catalog: dict[str, ActionDefinition],
    work_items: WorkItemService,
    interval_seconds: float = 120.0,
    min_gap_count: int = 2,
    lookback: int = 200,
    once: bool = False,
) -> None:
    service = ImprovementService(db_path=db_path, catalog=catalog, work_items=work_items)
    config = ImprovementWatchConfig(
        db_path=Path(db_path),
        interval_seconds=interval_seconds,
        min_gap_count=min_gap_count,
        lookback=lookback,
        once=once,
        actor=f"improvement-watchdog-{socket.gethostname()}",
    )
    ImprovementWatchdog(config=config, service=service).run_forever()
