from neurokernel_seed.harness.action_catalog import default_action_catalog
from neurokernel_seed.harness.runtime_planner import RuntimeActionPlanner, decision_policy_name, policy_decision_summary
from neurokernel_seed.harness.task_spec import TaskSpec


class FakeRuntimePolicy:
    def rank(self, *, task, decisions):
        return {
            "model_used": True,
            "reason": "ranked_by_runtime_action_model",
            "ranked_actions": ["get_memory_usage", "list_artifacts"],
            "top_action": "get_memory_usage",
            "scores": {
                "list_artifacts": {"model_used": True, "score": 0.1},
                "get_memory_usage": {"model_used": True, "score": 0.9},
            },
        }


def test_runtime_planner_uses_runtime_model_after_safety_gate():
    planner = RuntimeActionPlanner(catalog=default_action_catalog(), runtime_policy=FakeRuntimePolicy())
    task = TaskSpec(
        task_id="task_runtime_plan",
        goal="pick readonly action",
        target="orangepi5",
        allowed_actions=("list_artifacts", "get_memory_usage"),
        risk_level="low",
        requires_approval=False,
        mode="readonly",
    )

    plan = planner.plan(task=task, candidates=["list_artifacts", "get_memory_usage"])

    assert plan.chosen_action == "get_memory_usage"
    assert decision_policy_name(plan.policy) == "runtime_model_ranked_safety_gated"
    assert policy_decision_summary(plan.policy)["top_action"] == "get_memory_usage"
    assert plan.decisions[1]["model_score"]["score"] == 0.9


def test_runtime_planner_never_lets_model_bypass_approval_gate():
    planner = RuntimeActionPlanner(catalog=default_action_catalog(), runtime_policy=FakeRuntimePolicy())
    task = TaskSpec(
        task_id="task_runtime_plan_write",
        goal="write file",
        target="orangepi5",
        allowed_actions=("write_file",),
        risk_level="low",
        requires_approval=False,
    )

    plan = planner.plan(task=task, candidates=["write_file"])

    assert plan.chosen_action == "write_file"
    assert plan.chosen["safety"]["decision"] == "requires_approval"
