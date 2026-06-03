from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent.core.cognitive_engine import add_blackboard_item
from agent.core.database import connect, init_db
from agent.core.goals import create_goal
from agent.core.learner import create_reflection
from agent.memory.store import add_memory


@dataclass(frozen=True)
class ResearchPaperSeed:
    key: str
    title: str
    authors: str
    year: int
    url: str
    focus: str
    core_idea: str
    core_connection: str
    development_hook: str
    tags: tuple[str, ...]
    confidence: float = 0.78


RESEARCH_PAPER_SEEDS: tuple[ResearchPaperSeed, ...] = (
    ResearchPaperSeed(
        key="react",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        authors="Yao et al.",
        year=2022,
        url="https://arxiv.org/abs/2210.03629",
        focus="Reasoning traces and external actions are interleaved.",
        core_idea="Keep action decisions coupled to explicit reasoning and observation updates.",
        core_connection="Core should keep routing, tool action, and verification evidence in one inspectable trajectory.",
        development_hook="Use decision traces as the canonical record for why a task moved from plan to action to evidence.",
        tags=("reasoning", "acting", "traceability"),
        confidence=0.82,
    ),
    ResearchPaperSeed(
        key="reflexion",
        title="Reflexion: Language Agents with Verbal Reinforcement Learning",
        authors="Shinn et al.",
        year=2023,
        url="https://arxiv.org/abs/2303.11366",
        focus="Task feedback is stored as verbal reflection instead of model-weight updates.",
        core_idea="Turn outcomes into short reusable reflections with confidence and follow-up actions.",
        core_connection="Failed or corrected Core runs should become retrieval-ready learning records.",
        development_hook="Promote repeated high-confidence reflections into skills only after evidence accumulates.",
        tags=("reflection", "feedback", "episodic_memory"),
        confidence=0.86,
    ),
    ResearchPaperSeed(
        key="generative_agents",
        title="Generative Agents: Interactive Simulacra of Human Behavior",
        authors="Park et al.",
        year=2023,
        url="https://arxiv.org/abs/2304.03442",
        focus="Memory streams are retrieved, reflected on, and used for planning.",
        core_idea="Separate raw observations, synthesized reflections, and plans so each can be audited.",
        core_connection="Core memory, reflection, and project execution tables should remain distinct but cross-linked.",
        development_hook="Add evidence bundles that show which memories influenced each plan or response.",
        tags=("memory_stream", "reflection", "planning"),
        confidence=0.80,
    ),
    ResearchPaperSeed(
        key="voyager",
        title="Voyager: An Open-Ended Embodied Agent with Large Language Models",
        authors="Wang et al.",
        year=2023,
        url="https://arxiv.org/abs/2305.16291",
        focus="Curriculum, code-as-action, and reusable skills drive open-ended progress.",
        core_idea="Keep exploration bounded by explicit objectives, reusable skills, and measurable novelty.",
        core_connection="Core should propose bounded research notes and skill candidates, not unscoped action.",
        development_hook="Score generated goals by novelty, utility, risk, and whether they produce reusable skills.",
        tags=("curriculum", "skill_library", "novelty"),
        confidence=0.76,
    ),
    ResearchPaperSeed(
        key="memgpt",
        title="MemGPT: Towards LLMs as Operating Systems",
        authors="Packer et al.",
        year=2023,
        url="https://arxiv.org/abs/2310.08560",
        focus="Limited context is managed through virtual context and memory tiers.",
        core_idea="Treat memory as a managed resource with hot context, searchable recall, and archival summaries.",
        core_connection="Core should make memory lifecycle decisions explicit: retrieve, summarize, archive, and rehydrate.",
        development_hook="Expose memory tier status and compaction decisions in operating intelligence snapshots.",
        tags=("memory_management", "context", "operating_system"),
        confidence=0.84,
    ),
    ResearchPaperSeed(
        key="lats",
        title="Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models",
        authors="Zhou et al.",
        year=2023,
        url="https://arxiv.org/abs/2310.04406",
        focus="Tree search combines reasoning, acting, planning, and feedback.",
        core_idea="For high-impact tasks, compare candidate plans before committing to action.",
        core_connection="Core can keep single-pass execution for routine tasks and reserve plan search for risky goals.",
        development_hook="Add a lightweight candidate-plan scorer before queueing complex user-directed work.",
        tags=("planning", "tree_search", "evaluation"),
        confidence=0.78,
    ),
)


def list_research_paper_seeds() -> list[dict[str, Any]]:
    return [asdict(seed) for seed in RESEARCH_PAPER_SEEDS]


def _select_seeds(limit: int | None) -> list[ResearchPaperSeed]:
    if limit is None:
        return list(RESEARCH_PAPER_SEEDS)
    return list(RESEARCH_PAPER_SEEDS[: max(0, limit)])


def _paper_content(seed: ResearchPaperSeed) -> str:
    return "\n".join(
        [
            f"paper_key: {seed.key}",
            f"title: {seed.title}",
            f"authors: {seed.authors}",
            f"year: {seed.year}",
            f"url: {seed.url}",
            f"focus: {seed.focus}",
            f"core_idea: {seed.core_idea}",
            f"core_connection: {seed.core_connection}",
            f"development_hook: {seed.development_hook}",
        ]
    )


def _existing_memory_id(seed: ResearchPaperSeed) -> int | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM memories
            WHERE archived = 0
              AND memory_type = 'research_paper'
              AND content LIKE ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (f"%paper_key: {seed.key}%",),
        ).fetchone()
    return int(row["id"]) if row else None


def _create_research_goal(seeds: list[ResearchPaperSeed]) -> int:
    return create_goal(
        "Apply research-backed Core learning loop improvements",
        "Use collected agent papers to improve memory lifecycle, reflection promotion, planning evidence, and bounded goal generation.",
        goal_type="self_improvement_proposal",
        status="active",
        priority=0.74,
        risk_level="low",
        metadata={
            "priority_owner": "user",
            "source": "research_ingestion",
            "paper_keys": [seed.key for seed in seeds],
            "development_hooks": {seed.key: seed.development_hook for seed in seeds},
        },
    )


def ingest_research_papers(
    *,
    limit: int | None = None,
    dry_run: bool = False,
    source_event_id: int | None = None,
    goal_id: int | None = None,
) -> dict[str, Any]:
    selected = _select_seeds(limit)
    planned = [asdict(seed) for seed in selected]
    if dry_run:
        return {
            "dry_run": True,
            "papers": planned,
            "planned_memory_count": len(selected),
            "planned_reflection_count": len(selected),
            "planned_blackboard_count": len(selected),
            "planned_goal": "Apply research-backed Core learning loop improvements",
        }
    if not selected:
        return {
            "dry_run": False,
            "goal_id": None,
            "created": [],
            "skipped": [],
            "reflection_ids": [],
            "blackboard_ids": [],
            "paper_count": 0,
        }

    init_db()
    active_goal_id = goal_id or _create_research_goal(selected)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    reflection_ids: list[int] = []
    blackboard_ids: list[int] = []
    for seed in selected:
        existing_id = _existing_memory_id(seed)
        if existing_id is not None:
            skipped.append({"paper_key": seed.key, "memory_id": existing_id, "reason": "already_ingested"})
            continue
        memory_id = add_memory(
            seed.title,
            _paper_content(seed),
            memory_type="research_paper",
            tags=["research", "paper", "core_learning", *seed.tags],
            importance=0.84,
            confidence=seed.confidence,
            source_event_id=source_event_id,
        )
        reflection_id = create_reflection(
            f"Linked research paper to Core learning: {seed.title}",
            source_event_id=source_event_id,
            goal_id=active_goal_id,
            learned={
                "paper_key": seed.key,
                "core_idea": seed.core_idea,
                "core_connection": seed.core_connection,
                "development_hook": seed.development_hook,
                "memory_id": memory_id,
            },
            followup_goal={"goal_id": active_goal_id, "title": "Apply research-backed Core learning loop improvements"},
            confidence=seed.confidence,
        )
        blackboard_id = add_blackboard_item(
            "research_ingestion",
            seed.key,
            seed.development_hook,
            confidence=seed.confidence,
            tags=["research", "core_learning", *seed.tags],
            metadata={"paper_title": seed.title, "url": seed.url, "memory_id": memory_id, "reflection_id": reflection_id},
        )
        created.append({"paper_key": seed.key, "memory_id": memory_id})
        reflection_ids.append(reflection_id)
        blackboard_ids.append(blackboard_id)

    return {
        "dry_run": False,
        "goal_id": active_goal_id,
        "created": created,
        "skipped": skipped,
        "reflection_ids": reflection_ids,
        "blackboard_ids": blackboard_ids,
        "paper_count": len(selected),
    }
