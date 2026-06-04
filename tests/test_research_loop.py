import json

from agent.cli.agentctl import main
from agent.core.database import connect, init_db
from agent.core.research_loop import design_experiments, generate_hypotheses, generate_research_questions, propose_research_improvements, run_research_cycle
from agent.core.task_queue import list_tasks
from agent.memory.store import add_memory


def test_research_cycle_persists_question_hypothesis_experiment_and_proposal():
    init_db()
    add_memory("research core learning", "research should become hypothesis and experiment", memory_type="research_paper", tags=["research", "core_learning"], importance=0.8)

    result = run_research_cycle(limit=2, persist=True)

    assert result["counts"]["questions"] >= 1
    assert result["counts"]["hypotheses"] >= 1
    assert result["counts"]["experiments"] >= 1
    assert result["counts"]["evidence"] >= 1
    assert result["counts"]["proposals"] >= 1
    with connect() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM research_questions").fetchone()["c"] >= 1
        assert conn.execute("SELECT COUNT(*) c FROM research_hypotheses").fetchone()["c"] >= 1
        assert conn.execute("SELECT COUNT(*) c FROM research_experiments").fetchone()["c"] >= 1
        assert conn.execute("SELECT COUNT(*) c FROM research_evidence").fetchone()["c"] >= 1
        assert conn.execute("SELECT COUNT(*) c FROM research_proposals").fetchone()["c"] >= 1


def test_research_pipeline_can_enqueue_bounded_worker_task():
    init_db()
    generate_research_questions(limit=1, persist=True)
    generate_hypotheses(limit=1, persist=True)
    design_experiments(limit=1, persist=True)

    result = propose_research_improvements(limit=1, persist=True, enqueue=True)

    assert result["enqueued"] is True
    assert result["items"][0]["task_id"]
    tasks = list_tasks(limit=10, queue_type="autonomous")
    task = next(row for row in tasks if int(row["id"]) == int(result["items"][0]["task_id"]))
    assert task["task_kind"] == "code_change"
    assert task["payload"]["requires_native_loop"] is True
    assert "Do not read secrets" in task["payload"]["worker_prompt"]


def test_research_cli_cycle_and_questions(capsys):
    assert main(["research", "questions", "--limit", "2", "--persist"]) == 0
    questions = json.loads(capsys.readouterr().out)
    assert questions["persisted"] is True
    assert len(questions["items"]) >= 1

    assert main(["research", "cycle", "--limit", "1"]) == 0
    cycle = json.loads(capsys.readouterr().out)
    assert cycle["persisted"] is False
    assert cycle["counts"]["questions"] == 1


def test_research_proposal_does_not_enqueue_without_explicit_flag():
    init_db()
    run_research_cycle(limit=1, persist=True, enqueue=False)

    tasks = list_tasks(limit=20, queue_type="autonomous")
    assert all(row["source"] != "research_loop" for row in tasks)

def test_research_cycle_outputs_readable_korean_templates():
    init_db()

    result = run_research_cycle(limit=2, persist=False, enqueue=False)

    serialized = json.dumps(result, ensure_ascii=False)
    assert "??" not in serialized
    assert "작은 검증" in serialized
    assert "지정된 검증 명령이 통과한다" in serialized

