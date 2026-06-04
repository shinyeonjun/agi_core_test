from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from agent.config.defaults import now_kst
from agent.core.advanced_learning import circuit_breaker_snapshot, index_failure_cases, summarize_cognitive_graph
from agent.core.cognitive_engine import curiosity_signals, utility_novelty_score
from agent.core.cognitive_graph import upsert_edge, upsert_node
from agent.core.database import connect, init_db
from agent.core.goals import create_goal
from agent.core.learner import create_reflection
from agent.core.metrics import collect_metrics
from agent.core.research_ingestion import list_research_paper_seeds
from agent.core.task_queue import enqueue_task
from agent.memory.store import add_memory, search_memories


@dataclass(frozen=True)
class ResearchQuestion:
    key: str
    title: str
    prompt: str
    source: str
    priority: float
    novelty: float
    utility: float
    risk_level: str
    evidence: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ResearchHypothesis:
    key: str
    question_key: str
    statement: str
    rationale: str
    expected_effect: str
    falsification: str
    confidence: float
    evidence: dict[str, Any]


@dataclass(frozen=True)
class ResearchExperiment:
    key: str
    question_key: str
    hypothesis_key: str
    title: str
    plan: dict[str, Any]
    variables: dict[str, Any]
    success_criteria: list[str]
    verification_commands: list[str]
    risk_level: str


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _decode(value: object, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return fallback


def _fp(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part or "") for part in parts).encode()).hexdigest()[:16]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _num(value: object, default: float = 0.5) -> float:
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return default


def _question_from_signal(signal: dict[str, Any]) -> ResearchQuestion:
    topic = str(signal.get("topic") or "unknown")
    pressure = _num(signal.get("pressure"), 0.3)
    titles = {
        "renderer_quality": "대화 답변 품질은 왜 가끔 템플릿처럼 보이나?",
        "memory_hygiene": "기억과 회고가 쌓일수록 판단 품질은 좋아지고 있나?",
        "action_reliability": "반복 실패는 어떤 실행 단계에서 생기나?",
        "goal_queue": "사용자 목표와 자율 목표가 서로 방해하지 않고 처리되나?",
        "runtime_body": "Core가 자기 실행 환경을 최신 상태로 이해하고 있나?",
        "memory_retrieval": "기억 검색 결과가 실제 답변과 작업 선택에 충분히 도움 되나?",
    }
    prompts = {
        "renderer_quality": "fallback, formatter, Discord 응답 테스트를 근거로 대화 품질 저하 원인을 찾고 작게 검증한다.",
        "memory_hygiene": "memory_rollups, reflection, search 결과를 비교해 압축이 판단에 주는 영향을 검증한다.",
        "action_reliability": "failure_cases와 circuit_breakers를 보고 같은 실패가 반복되는 실행 단계를 분리한다.",
        "goal_queue": "task_queue와 goal 우선순위를 보고 사용자 요청이 자율 루프에 밀리지 않는지 검증한다.",
        "runtime_body": "self-map과 서비스 상태가 판단에 반영되는지 확인한다.",
        "memory_retrieval": "FTS, sparse vector, rollup, graph summary가 서로 보완되는지 비교한다.",
    }
    title = titles.get(topic, str(signal.get("question") or "Core 개선 연구 질문"))
    score = utility_novelty_score({"title": title, "priority": pressure, "risk_level": "low"})
    return ResearchQuestion(_fp("signal", topic, title), title, prompts.get(topic, title), topic, max(0.25, score["score"]), _num(score["components"].get("novelty"), 0.4), _num(score["components"].get("utility"), pressure), "low", {"signal": signal, "score": score}, {"kind": "curiosity_signal"})


def _failure_question(snapshot: dict[str, Any]) -> ResearchQuestion | None:
    categories = snapshot.get("categories") or {}
    if not categories:
        return None
    dominant = Counter(categories).most_common(1)[0][0]
    count = int(categories.get(dominant) or 0)
    return ResearchQuestion(_fp("failure", dominant), f"반복 실패 {dominant}는 왜 계속 생기나?", "최근 실패 사례를 묶어서 원인, 반복 조건, 안전한 복구 실험을 만든다.", "failure_recovery", _clamp(0.45 + min(0.35, count / 100)), 0.55, 0.82, "medium", {"failure_categories": categories, "sample_count": snapshot.get("count")}, {"kind": "failure_pattern", "dominant_category": dominant})


def _research_transfer_question() -> ResearchQuestion:
    seeds = list_research_paper_seeds()
    keys = [item["key"] for item in seeds[:8]]
    return ResearchQuestion(_fp("research_transfer", ",".join(keys)), "논문 아이디어를 Core 실험으로 바꾸는 경로가 실제로 작동하나?", "연구 seed를 요약 저장하는 데서 멈추지 말고, 가설과 작은 실험, 제안 티켓으로 승격되는지 검증한다.", "research_transfer", 0.72, 0.62, 0.78, "low", {"paper_keys": keys, "paper_count": len(seeds)}, {"kind": "research_transfer", "papers": keys})


def generate_research_questions(*, limit: int = 5, persist: bool = False) -> dict[str, Any]:
    init_db()
    signals = curiosity_signals(collect_metrics(), limit=max(6, limit))
    questions = [_question_from_signal(signal) for signal in signals]
    failure_q = _failure_question(index_failure_cases(limit=60, persist=persist))
    if failure_q:
        questions.append(failure_q)
    questions.append(_research_transfer_question())
    deduped = {q.key: q for q in questions}
    ranked = sorted(deduped.values(), key=lambda q: (q.priority, q.utility, q.novelty), reverse=True)[: max(1, int(limit))]
    ids: dict[str, int] = {}
    if persist:
        ts = now_kst()
        with connect() as conn:
            for q in ranked:
                conn.execute("""
                    INSERT INTO research_questions (created_at, updated_at, question_key, title, prompt, source, status, priority, novelty, utility, risk_level, evidence_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(question_key) DO UPDATE SET updated_at=excluded.updated_at, title=excluded.title, prompt=excluded.prompt, source=excluded.source, status='open', priority=excluded.priority, novelty=excluded.novelty, utility=excluded.utility, risk_level=excluded.risk_level, evidence_json=excluded.evidence_json, metadata_json=excluded.metadata_json
                """, (ts, ts, q.key, q.title[:220], q.prompt[:1200], q.source, q.priority, q.novelty, q.utility, q.risk_level, _json(q.evidence), _json(q.metadata)))
                ids[q.key] = int(conn.execute("SELECT id FROM research_questions WHERE question_key=?", (q.key,)).fetchone()["id"])
            conn.commit()
        for q in ranked:
            upsert_node("research_idea", "research_question", ids[q.key], q.title, summary=q.prompt, importance=q.priority, confidence=0.76, risk=0.12 if q.risk_level == "medium" else 0.05, metadata={"source": q.source, "question_key": q.key})
    return {"persisted": persist, "items": [asdict(q) | ({"id": ids[q.key]} if q.key in ids else {}) for q in ranked]}

def _question_rows(limit: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM research_questions WHERE status!='archived' ORDER BY priority DESC, updated_at DESC, id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    return [dict(row) for row in rows]


def _hypothesis_for_question(q: dict[str, Any]) -> ResearchHypothesis:
    source = str(q.get("source") or "unknown")
    prompt = str(q.get("prompt") or q.get("title") or "Core 개선 질문")
    templates = {
        "renderer_quality": (
            "렌더러 실패 경로가 사용자 질문보다 내부 상태 설명을 먼저 붙이면서 답변 품질을 떨어뜨린다.",
            "fallback과 formatter가 실패를 복구할 때 내부 필드명, 처리 단계, 메타데이터를 그대로 말하면 사용자는 답변이 멈추거나 딱딱해졌다고 느낀다.",
            "fallback 응답을 질문 의도별 자연어로 제한하면 내부 냄새와 중복 응답이 줄어든다.",
        ),
        "memory_hygiene": (
            "오래된 기억과 회고를 압축하지 않으면 검색 결과가 중복되고 판단 근거가 흐려진다.",
            "raw memory가 계속 쌓이면 같은 의미의 기록이 여러 번 검색되고, rollup이나 graph summary가 없으면 중요한 장기 사실이 묻힌다.",
            "중복 기억을 압축하고 오래된 회고를 요약하면 검색 hit 품질과 답변 일관성이 올라간다.",
        ),
        "action_reliability": (
            "반복 실패는 실행 자체보다 실패 원인 분류와 복구 조건이 약해서 다시 생긴다.",
            "실패 사례가 circuit breaker와 연결되지 않으면 같은 정책 차단이나 검증 실패가 작업 큐에 계속 다시 들어온다.",
            "실패 단계와 원인을 분리하면 다음 작업은 더 작게 쪼개지고 차단 반복이 줄어든다.",
        ),
        "failure_recovery": (
            "정책 차단 실패는 금지된 행동을 계속 시도해서가 아니라 대체 경로를 못 찾을 때 반복된다.",
            "failure case를 질문, 가설, 실험으로 승격하면 차단 이유를 복구 가능한 작업 단위로 바꿀 수 있다.",
            "차단된 작업마다 안전한 대체 경로와 재시도 조건이 남으면 자율 루프가 덜 답답하게 보인다.",
        ),
        "research_transfer": (
            "논문 seed는 요약 저장에서 멈추면 지식 창고가 되고, 실험으로 연결될 때 Core 개선 재료가 된다.",
            "연구 아이디어가 질문, 가설, 실험, 제안으로 이어져야 실제 코드나 평가 루프에 영향을 준다.",
            "논문 기반 아이디어를 작은 self-improvement ticket으로 바꾸면 연구가 운영 학습으로 연결된다.",
        ),
    }
    statement, rationale, expected = templates.get(
        source,
        (
            "현재 Core 상태에서 관찰된 문제는 작은 실험으로 쪼개면 안전하게 검증할 수 있다.",
            prompt,
            "검증 가능한 작은 변경으로 만들면 성공, 실패, 보류 이유를 더 분명하게 남길 수 있다.",
        ),
    )
    falsification = "해당 변경 뒤에도 같은 실패나 품질 저하가 줄지 않으면 이 가설은 폐기하거나 더 좁게 다시 세운다."
    confidence = 0.68 if source in {"renderer_quality", "memory_hygiene", "failure_recovery"} else 0.62
    return ResearchHypothesis(_fp("hypothesis", q.get("question_key"), statement), str(q.get("question_key") or ""), statement, rationale, expected, falsification, confidence, {"question_id": q.get("id"), "question_title": q.get("title"), "source": source})

def generate_hypotheses(*, limit: int = 5, persist: bool = False) -> dict[str, Any]:
    init_db()
    questions = _question_rows(limit)
    if not questions:
        generate_research_questions(limit=limit, persist=True)
        questions = _question_rows(limit)
    hypotheses = [_hypothesis_for_question(q) for q in questions[: max(1, int(limit))]]
    ids: dict[str, int] = {}
    if persist:
        ts = now_kst()
        with connect() as conn:
            for h in hypotheses:
                qrow = conn.execute("SELECT id FROM research_questions WHERE question_key=?", (h.question_key,)).fetchone()
                qid = int(qrow["id"]) if qrow else None
                conn.execute("""
                    INSERT INTO research_hypotheses (created_at, updated_at, question_id, hypothesis_key, statement, rationale, expected_effect, falsification, confidence, status, evidence_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)
                    ON CONFLICT(hypothesis_key) DO UPDATE SET updated_at=excluded.updated_at, question_id=excluded.question_id, statement=excluded.statement, rationale=excluded.rationale, expected_effect=excluded.expected_effect, falsification=excluded.falsification, confidence=excluded.confidence, status='open', evidence_json=excluded.evidence_json, metadata_json=excluded.metadata_json
                """, (ts, ts, qid, h.key, h.statement[:1600], h.rationale[:1200], h.expected_effect[:1000], h.falsification[:1000], h.confidence, _json(h.evidence), _json({"question_key": h.question_key})))
                ids[h.key] = int(conn.execute("SELECT id FROM research_hypotheses WHERE hypothesis_key=?", (h.key,)).fetchone()["id"])
            conn.commit()
        for h in hypotheses:
            hnode = upsert_node("research_idea", "research_hypothesis", ids[h.key], h.statement[:220], summary=h.rationale, importance=0.72, confidence=h.confidence, risk=0.08, metadata={"hypothesis_key": h.key})
            qid = h.evidence.get("question_id")
            if qid:
                qnode = upsert_node("research_idea", "research_question", qid, h.evidence.get("question_title") or "research question", summary="", importance=0.66, confidence=0.7, risk=0.05)
                upsert_edge(hnode, qnode, "supports", weight=0.78, confidence=0.72, evidence={"source": "research_hypothesis"})
    return {"persisted": persist, "items": [asdict(h) | ({"id": ids[h.key]} if h.key in ids else {}) for h in hypotheses]}


def _hypothesis_rows(limit: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT h.*, q.question_key, q.title question_title, q.source question_source FROM research_hypotheses h LEFT JOIN research_questions q ON q.id=h.question_id WHERE h.status!='archived' AND h.statement NOT LIKE '%??%' ORDER BY h.confidence DESC, h.updated_at DESC, h.id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    return [dict(row) for row in rows]


def _experiment_for_hypothesis(h: dict[str, Any]) -> ResearchExperiment:
    source = str(h.get("question_source") or "unknown")
    title = str(h.get("statement") or "Core research hypothesis")[:120]
    target_tests = {
        "renderer_quality": ["python -m pytest tests/test_renderer.py tests/test_discord_control_plane.py -q"],
        "memory_hygiene": ["python -m pytest tests/test_retrieval_efficiency.py tests/test_advanced_learning.py -q"],
        "action_reliability": ["python -m pytest tests/test_failure_taxonomy.py tests/test_advanced_learning.py -q"],
        "failure_recovery": ["python -m pytest tests/test_failure_taxonomy.py tests/test_advanced_learning.py -q"],
        "research_transfer": ["python -m pytest tests/test_research_ingestion.py tests/test_research_loop.py -q"],
    }
    plan = {
        "steps": [
            "관련 코드와 최근 실행 기록에서 원인을 확인한다.",
            "가장 작은 변경 후보 하나만 만든다.",
            "지정된 테스트와 빠른 검증 명령을 실행한다.",
            "성공, 실패, 보류 이유를 evidence ledger에 남긴다.",
        ],
        "bounded_scope": True,
        "no_secret_access": True,
    }
    variables = {
        "changed_component": source,
        "control": "현재 main 동작",
        "measurement": ["test_result", "failure_count", "relevant_metric"],
    }
    success_criteria = [
        "지정된 검증 명령이 통과한다.",
        "변경 이유와 관찰 결과가 evidence ledger에 남는다.",
        "시크릿, 권한 상승, 외부 네트워크를 건드리지 않는다.",
    ]
    return ResearchExperiment(_fp("experiment", h.get("hypothesis_key"), title), str(h.get("question_key") or ""), str(h.get("hypothesis_key") or ""), f"작은 검증: {title}", plan, variables, success_criteria, target_tests.get(source, ["python -m agent.cli.agentctl test run fast --json"]), "medium" if source in {"action_reliability", "failure_recovery"} else "low")

def design_experiments(*, limit: int = 5, persist: bool = False) -> dict[str, Any]:
    init_db()
    hypotheses = _hypothesis_rows(limit)
    if not hypotheses:
        generate_hypotheses(limit=limit, persist=True)
        hypotheses = _hypothesis_rows(limit)
    experiments = [_experiment_for_hypothesis(h) for h in hypotheses[: max(1, int(limit))]]
    ids: dict[str, int] = {}
    if persist:
        ts = now_kst()
        with connect() as conn:
            for e in experiments:
                qrow = conn.execute("SELECT id FROM research_questions WHERE question_key=?", (e.question_key,)).fetchone()
                hrow = conn.execute("SELECT id FROM research_hypotheses WHERE hypothesis_key=?", (e.hypothesis_key,)).fetchone()
                qid = int(qrow["id"]) if qrow else None
                hid = int(hrow["id"]) if hrow else None
                conn.execute("""
                    INSERT INTO research_experiments (created_at, updated_at, question_id, hypothesis_id, experiment_key, title, plan_json, variables_json, success_criteria_json, verification_commands_json, status, result_json, score, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', '{}', 0.0, ?)
                    ON CONFLICT(experiment_key) DO UPDATE SET updated_at=excluded.updated_at, question_id=excluded.question_id, hypothesis_id=excluded.hypothesis_id, title=excluded.title, plan_json=excluded.plan_json, variables_json=excluded.variables_json, success_criteria_json=excluded.success_criteria_json, verification_commands_json=excluded.verification_commands_json, status='planned', risk_level=excluded.risk_level
                """, (ts, ts, qid, hid, e.key, e.title[:220], _json(e.plan), _json(e.variables), _json(e.success_criteria), _json(e.verification_commands), e.risk_level))
                ids[e.key] = int(conn.execute("SELECT id FROM research_experiments WHERE experiment_key=?", (e.key,)).fetchone()["id"])
            conn.commit()
    return {"persisted": persist, "items": [asdict(e) | ({"id": ids[e.key]} if e.key in ids else {}) for e in experiments]}

def collect_research_evidence(*, limit: int = 12, persist: bool = False) -> dict[str, Any]:
    init_db()
    metrics = collect_metrics()
    graph = summarize_cognitive_graph(limit=80, persist=False, sync=True)
    failures = index_failure_cases(limit=50, persist=False)
    circuit = circuit_breaker_snapshot(limit=50, threshold=3, persist=False)
    evidence = [
        {"evidence_type": "metrics", "source_type": "metrics", "source_id": "current", "title": "현재 Core 지표", "summary": f"eval={metrics.get('last_eval_result')} renderer_success={metrics.get('renderer_success_rate')} memory_vector={metrics.get('memory_vector_coverage')}", "payload": metrics, "confidence": 0.82},
        {"evidence_type": "graph", "source_type": "cognitive_graph", "source_id": "community_summary", "title": "인지 그래프 요약", "summary": f"communities={len(graph.get('communities') or [])}", "payload": graph, "confidence": 0.74},
        {"evidence_type": "failure", "source_type": "failure_cases", "source_id": "recent", "title": "최근 실패 사례", "summary": f"count={failures.get('count')} categories={failures.get('categories')}", "payload": failures, "confidence": 0.78},
        {"evidence_type": "circuit", "source_type": "circuit_breakers", "source_id": "open", "title": "반복 차단 감지", "summary": f"open={circuit.get('open_count')}", "payload": circuit, "confidence": 0.78},
    ]
    for memory in search_memories("research core learning improvement", limit=max(2, min(8, limit))):
        evidence.append({"evidence_type": "memory", "source_type": "memory", "source_id": str(memory.get("id")), "title": str(memory.get("title") or "memory"), "summary": str(memory.get("content") or "")[:700], "payload": memory, "confidence": _num(memory.get("confidence"), 0.65)})
    evidence = evidence[: max(1, int(limit))]
    ids: list[int] = []
    if persist:
        ts = now_kst()
        with connect() as conn:
            for item in evidence:
                cur = conn.execute("""
                    INSERT INTO research_evidence (created_at, evidence_type, source_type, source_id, title, summary, payload_json, confidence, supports_type, supports_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """, (ts, item["evidence_type"], item["source_type"], item["source_id"], item["title"][:220], item["summary"][:1600], _json(item["payload"]), item["confidence"]))
                ids.append(int(cur.lastrowid))
            conn.commit()
    return {"persisted": persist, "items": [item | ({"id": ids[idx]} if idx < len(ids) else {}) for idx, item in enumerate(evidence)]}


def _experiment_rows(limit: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT e.*, h.statement, q.title question_title, q.source question_source FROM research_experiments e LEFT JOIN research_hypotheses h ON h.id=e.hypothesis_id LEFT JOIN research_questions q ON q.id=e.question_id WHERE e.status!='archived' AND e.title NOT LIKE '%??%' AND COALESCE(h.statement, '') NOT LIKE '%??%' ORDER BY e.updated_at DESC, e.id DESC LIMIT ?", (max(1, int(limit)),)).fetchall()
    return [dict(row) for row in rows]


def propose_research_improvements(*, limit: int = 3, persist: bool = False, enqueue: bool = False) -> dict[str, Any]:
    init_db()
    experiments = _experiment_rows(limit)
    if not experiments:
        design_experiments(limit=limit, persist=True)
        experiments = _experiment_rows(limit)
    evidence = collect_research_evidence(limit=8, persist=persist)
    evidence_ids = [int(item["id"]) for item in evidence.get("items", []) if item.get("id")]
    proposals: list[dict[str, Any]] = []
    ts = now_kst()

    for exp in experiments[: max(1, int(limit))]:
        criteria = _decode(exp.get("success_criteria_json"), [])
        commands = _decode(exp.get("verification_commands_json"), [])
        question_title = str(exp.get("question_title") or exp.get("title") or "Core research question")
        title = f"연구 기반 개선: {question_title}"[:220]
        worker_prompt = "\n".join([
            "Agent Core research-backed self-improvement task.",
            "",
            f"Research question: {question_title}",
            f"Hypothesis: {exp.get('statement') or ''}",
            f"Experiment: {exp.get('title') or ''}",
            "",
            "Success criteria:",
            *[f"- {item}" for item in criteria],
            "",
            "Verification commands:",
            *[f"- {cmd}" for cmd in commands],
            "",
            "Boundaries:",
            "- Keep the change narrow and testable.",
            "- Do not read secrets, .env, SSH keys, tokens, or credentials.",
            "- Do not use sudo, apt, systemd, or external network without approval.",
            "- Report evidence, failed assumptions, and remaining risk.",
        ])
        proposal = {
            "title": title,
            "proposal_type": "self_improvement_code",
            "status": "proposed",
            "priority": 0.76,
            "risk_level": exp.get("risk_level") or "low",
            "worker_prompt": worker_prompt,
            "success_criteria": criteria,
            "verification_commands": commands,
            "evidence_ids": evidence_ids[:6],
            "experiment_id": exp.get("id"),
            "hypothesis_id": exp.get("hypothesis_id"),
            "question_id": exp.get("question_id"),
            "experiment_key": exp.get("experiment_key"),
        }
        proposals.append(proposal)

    if persist and proposals:
        with connect() as conn:
            for proposal in proposals:
                conn.execute("""
                    INSERT INTO research_proposals (created_at, updated_at, question_id, hypothesis_id, experiment_id, title, proposal_type, status, priority, risk_level, worker_prompt, success_criteria_json, evidence_ids_json, metadata_json, queued_task_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """, (ts, ts, proposal["question_id"], proposal["hypothesis_id"], proposal["experiment_id"], proposal["title"], proposal["proposal_type"], proposal["status"], proposal["priority"], proposal["risk_level"], proposal["worker_prompt"], _json(proposal["success_criteria"]), _json(proposal["evidence_ids"]), _json({"verification_commands": proposal["verification_commands"]})))
                proposal["id"] = int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])
            conn.commit()

    if enqueue:
        queued_updates: list[tuple[int, int]] = []
        for proposal in proposals:
            goal_id = create_goal(
                proposal["title"],
                proposal["worker_prompt"],
                goal_type="self_improvement_proposal",
                status="active",
                priority=0.86,
                risk_level=str(proposal["risk_level"]),
                metadata={
                    "source": "research_loop",
                    "proposal_id": proposal.get("id"),
                    "experiment_id": proposal.get("experiment_id"),
                    "requires_native_loop": True,
                },
                dedupe=True,
            )
            task_id = enqueue_task(
                "autonomous",
                goal_id=goal_id,
                task_kind="code_change",
                title=proposal["title"],
                source="research_loop",
                priority=0.82,
                payload={
                    "worker_prompt": proposal["worker_prompt"],
                    "research_proposal_id": proposal.get("id"),
                    "requires_native_loop": True,
                },
                idempotency_key=f"research_proposal:{proposal.get('experiment_key')}",
            )
            proposal["goal_id"] = goal_id
            proposal["task_id"] = task_id
            proposal["status"] = "queued"
            if persist and proposal.get("id"):
                queued_updates.append((task_id, int(proposal["id"])))
        if queued_updates:
            with connect() as conn:
                for task_id, proposal_id in queued_updates:
                    conn.execute("UPDATE research_proposals SET status='queued', queued_task_id=?, updated_at=? WHERE id=?", (task_id, now_kst(), proposal_id))
                conn.commit()

    if persist and proposals:
        create_reflection("연구 루프가 개선 제안을 만들었다", learned={"proposal_count": len(proposals), "enqueued": enqueue}, confidence=0.74)
        add_memory("research loop proposals", f"Research loop generated {len(proposals)} proposals. enqueued={enqueue}", memory_type="research_result", tags=["research_loop", "self_improvement"], importance=0.72, confidence=0.72)
    return {"persisted": persist, "enqueued": enqueue, "items": proposals}

def run_research_cycle(*, limit: int = 3, persist: bool = False, enqueue: bool = False) -> dict[str, Any]:
    questions = generate_research_questions(limit=limit, persist=persist)
    hypotheses = generate_hypotheses(limit=limit, persist=persist)
    experiments = design_experiments(limit=limit, persist=persist)
    evidence = collect_research_evidence(limit=max(4, limit * 2), persist=persist)
    proposals = propose_research_improvements(limit=limit, persist=persist, enqueue=enqueue)
    return {
        "persisted": persist,
        "enqueued": enqueue,
        "counts": {
            "questions": len(questions.get("items", [])),
            "hypotheses": len(hypotheses.get("items", [])),
            "experiments": len(experiments.get("items", [])),
            "evidence": len(evidence.get("items", [])),
            "proposals": len(proposals.get("items", [])),
        },
        "questions": questions.get("items", []),
        "hypotheses": hypotheses.get("items", []),
        "experiments": experiments.get("items", []),
        "evidence": evidence.get("items", []),
        "proposals": proposals.get("items", []),
    }
