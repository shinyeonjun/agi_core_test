from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent.config.defaults import now_kst

PIPELINE_PHASES = (
    "observe",
    "interpret",
    "retrieve",
    "plan",
    "act_or_defer",
    "verify",
    "reflect",
    "report",
)


@dataclass
class PipelinePhaseRecord:
    name: str
    status: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=now_kst)


@dataclass
class CorePipelineTrace:
    version: str = "0.17"
    phases: list[PipelinePhaseRecord] = field(default_factory=list)

    def add(self, name: str, status: str, summary: str, metadata: dict[str, Any] | None = None) -> None:
        if name not in PIPELINE_PHASES:
            raise ValueError(f"unknown pipeline phase: {name}")
        self.phases.append(PipelinePhaseRecord(name=name, status=status, summary=summary, metadata=metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        recorded = {phase.name for phase in self.phases}
        return {
            "version": self.version,
            "phase_order": list(PIPELINE_PHASES),
            "completed": [phase.name for phase in self.phases if phase.status in {"done", "deferred"}],
            "missing": [phase for phase in PIPELINE_PHASES if phase not in recorded],
            "phases": [asdict(phase) for phase in self.phases],
        }
