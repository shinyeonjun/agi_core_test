from __future__ import annotations

import argparse
import json
from typing import Sequence

from agent import __version__
from agent.core.approvals import ApprovalStore
from agent.core.database import get_schema_version, init_db
from agent.core.events import list_events, log_event
from agent.core.goals import list_goals, mark_goal_done
from agent.core.learner import list_reflections, list_skills, upsert_skill, update_after_turn
from agent.core.pipeline import run_talk
from agent.core.policy import ActionProposal, PolicyEngine
from agent.core.state import load_state, save_state
from agent.eval.harness import list_eval_runs, list_tasks, run_suite
from agent.memory.store import add_memory, list_memories, rebuild_memory_fts, search_memories
from agent.ops.backup import create_backup
from agent.scheduler.tick import run_tick
from agent.tools.system_readonly import READ_ONLY_COMMANDS, run_readonly, system_snapshot


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def cmd_init(_args: argparse.Namespace) -> int:
    init_db()
    state = load_state()
    save_state(state)
    rebuild_memory_fts()
    print(f"agent-core {__version__} initialized schema={get_schema_version()}")
    return 0


def cmd_state(_args: argparse.Namespace) -> int:
    print_json(load_state())
    return 0


def cmd_talk(args: argparse.Namespace) -> int:
    print(run_talk(args.message)["text"])
    return 0


def cmd_tick(_args: argparse.Namespace) -> int:
    result = run_tick()
    print(result["message"])
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    print_json(list_events(args.limit))
    return 0


def cmd_goals(args: argparse.Namespace) -> int:
    print_json(list_goals(args.limit, include_archived=args.all))
    return 0


def cmd_goal_done(args: argparse.Namespace) -> int:
    ok = mark_goal_done(args.goal_id)
    print("goal marked done" if ok else "goal not found")
    return 0 if ok else 1


def cmd_memories(args: argparse.Namespace) -> int:
    print_json(list_memories(args.limit))
    return 0


def cmd_memory_add(args: argparse.Namespace) -> int:
    memory_id = add_memory(args.title, args.content, args.type, args.tags or [], args.importance, args.confidence)
    print(f"memory added: #{memory_id}")
    return 0


def cmd_memory_search(args: argparse.Namespace) -> int:
    print_json(search_memories(args.query, args.limit))
    return 0


def cmd_memory_rebuild_fts(_args: argparse.Namespace) -> int:
    rebuild_memory_fts()
    print("memory FTS rebuild complete")
    return 0


def _policy_response(proposal: ActionProposal, approval_id: int | None = None) -> dict[str, object]:
    result = proposal.to_dict()
    result["approval_id"] = approval_id
    result["queued_for_approval"] = approval_id is not None
    result["will_execute"] = False
    return result


def cmd_policy_check(args: argparse.Namespace) -> int:
    init_db()
    engine = PolicyEngine()
    store = ApprovalStore()
    decision = engine.classify_decision(args.text, action_type=args.action_type)
    engine.record_decision(decision)
    proposal = engine.classify_text(args.text, action_type=args.action_type)
    approval_id = None
    if proposal.requires_approval and proposal.denied_reason is None:
        approval_id = store.create_approval(proposal)
    result = _policy_response(proposal, approval_id)
    log_event("policy", "policy_check", args.text, result, importance=0.6)
    print_json(result)
    return 0


def cmd_approvals(args: argparse.Namespace) -> int:
    init_db()
    store = ApprovalStore()
    rows = store.list(status=None, limit=args.limit) if args.all else store.list_pending()[: args.limit]
    print_json(rows)
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    init_db()
    ok = ApprovalStore().approve(args.approval_id)
    log_event("approval", "approval_approved" if ok else "approval_approve_failed", str(args.approval_id), {"ok": ok}, 0.7)
    print("approved" if ok else "not found or not pending")
    return 0 if ok else 1


def cmd_reject(args: argparse.Namespace) -> int:
    init_db()
    ok = ApprovalStore().reject(args.approval_id)
    log_event("approval", "approval_rejected" if ok else "approval_reject_failed", str(args.approval_id), {"ok": ok}, 0.7)
    print("rejected" if ok else "not found or not pending")
    return 0 if ok else 1


def cmd_reflections(args: argparse.Namespace) -> int:
    print_json(list_reflections(args.limit))
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    print_json(list_skills(args.limit))
    return 0


def cmd_skill_add(args: argparse.Namespace) -> int:
    skill_id = upsert_skill(args.name, args.trigger, args.procedure, args.tags or [])
    print(f"skill upsert complete: #{skill_id}")
    return 0


def cmd_eval_list(args: argparse.Namespace) -> int:
    print_json([task.__dict__ for task in list_tasks(args.suite)])
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    result = run_suite(args.suite)
    print_json(result)
    return 1 if result["result"] in {"FAIL", "UNSAFE"} else 0


def cmd_eval_runs(args: argparse.Namespace) -> int:
    print_json(list_eval_runs(args.limit))
    return 0


def cmd_tool_list(_args: argparse.Namespace) -> int:
    print_json(sorted(READ_ONLY_COMMANDS))
    return 0


def cmd_tool_run(args: argparse.Namespace) -> int:
    print_json(run_readonly(args.name))
    return 0


def cmd_snapshot(_args: argparse.Namespace) -> int:
    print_json(system_snapshot())
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    print_json(create_backup(args.label))
    return 0


def cmd_audit(_args: argparse.Namespace) -> int:
    checks = {
        "schema_version": get_schema_version(),
        "state_version": load_state().get("version"),
        "policy_df": PolicyEngine().classify_text("df -h").to_dict(),
        "policy_apt": PolicyEngine().classify_text("apt-get install nginx").to_dict(),
        "policy_rm": PolicyEngine().classify_text("rm    -rf    /").to_dict(),
        "policy_ssh": PolicyEngine().classify_text("cat ~/.ssh/id_rsa").to_dict(),
        "pending_approvals": ApprovalStore().list_pending(),
    }
    print_json(checks)
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    if args.area == "bridge":
        from agent.bridge.auth import DiscordAuthConfig, classify_context
        from agent.bridge.formatter import split_for_discord, strip_bot_mention
        config = DiscordAuthConfig(allowed_user_ids={"1"}, allowed_channel_ids={"10"}, user_cooldown_seconds=0)
        result = {
            "allowed_dm": classify_context(user_id="1", channel_id="dm", is_dm=True, was_mention=False, author_is_bot=False, config=config) == "conversation",
            "denied_user": classify_context(user_id="2", channel_id="10", is_dm=False, was_mention=False, author_is_bot=False, config=config) == "denied",
            "allowed_channel": classify_context(user_id="1", channel_id="10", is_dm=False, was_mention=False, author_is_bot=False, config=config) == "conversation",
            "mention_strip": strip_bot_mention("<@123> hello") == "hello",
            "chunking_ok": all(len(chunk) <= 1800 for chunk in split_for_discord("a" * 4000, 1800)),
        }
        result["ok"] = all(result.values())
        print_json(result)
        return 0 if result["ok"] else 1
    if args.area == "memory":
        memory_id = add_memory("digital agi korean memory", "Core remembers \ub514\uc9c0\ud138 AGI context.", memory_type="project_context", tags=["digital_agi", "core"], importance=0.93)
        results = search_memories("\ub514\uc9c0\ud138 AGI", limit=5)
        result = {"ok": any(row["id"] == memory_id for row in results), "memory_id": memory_id, "result_count": len(results)}
        print_json(result)
        return 0 if result["ok"] else 1
    if args.area == "reflection":
        before = len(list_reflections(1000))
        learner = update_after_turn("\uc544\ub2c8 \ub2e4\uc2dc \ud574\uc918", None, None, {"selected_goal_id": None})
        after = len(list_reflections(1000))
        skills = list_skills(100)
        result = {
            "ok": learner["feedback"] == "negative" and after > before and any(skill["name"] == "core_talk_pipeline" for skill in skills),
            "feedback": learner["feedback"],
            "reflection_created": after > before,
        }
        print_json(result)
        return 0 if result["ok"] else 1
    if args.area == "tool":
        uptime = run_readonly("uptime")
        snapshot = system_snapshot()
        result = {"ok": uptime["requires_approval"] is False and uptime["risk_level"] == "medium" and "snapshot_id" in snapshot, "uptime_rc": uptime["returncode"], "snapshot_id": snapshot.get("snapshot_id")}
        print_json(result)
        return 0 if result["ok"] else 1
    raise ValueError(f"unknown self-check area: {args.area}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentctl")
    parser.add_argument("--version", action="version", version=f"agent-core {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init"); p.set_defaults(func=cmd_init)
    p = sub.add_parser("state"); p.set_defaults(func=cmd_state)
    p = sub.add_parser("talk"); p.add_argument("message"); p.set_defaults(func=cmd_talk)
    p = sub.add_parser("tick"); p.set_defaults(func=cmd_tick)
    p = sub.add_parser("events"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_events)
    p = sub.add_parser("goals"); p.add_argument("--limit", type=int, default=20); p.add_argument("--all", action="store_true"); p.set_defaults(func=cmd_goals)

    p = sub.add_parser("goal"); goal_sub = p.add_subparsers(dest="goal_command", required=True)
    p_done = goal_sub.add_parser("done"); p_done.add_argument("goal_id", type=int); p_done.set_defaults(func=cmd_goal_done)

    p = sub.add_parser("memories"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_memories)
    p = sub.add_parser("memory"); memory_sub = p.add_subparsers(dest="memory_command", required=True)
    p_add = memory_sub.add_parser("add"); p_add.add_argument("title"); p_add.add_argument("content"); p_add.add_argument("--type", default="fact"); p_add.add_argument("--tags", nargs="*"); p_add.add_argument("--importance", type=float, default=0.5); p_add.add_argument("--confidence", type=float, default=0.7); p_add.set_defaults(func=cmd_memory_add)
    p_search = memory_sub.add_parser("search"); p_search.add_argument("query"); p_search.add_argument("--limit", type=int, default=10); p_search.set_defaults(func=cmd_memory_search)
    p_fts = memory_sub.add_parser("rebuild-fts"); p_fts.set_defaults(func=cmd_memory_rebuild_fts)

    p = sub.add_parser("policy-check"); p.add_argument("text"); p.add_argument("--action-type", default="shell_text"); p.set_defaults(func=cmd_policy_check)
    p = sub.add_parser("approvals"); p.add_argument("--all", action="store_true"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_approvals)
    p = sub.add_parser("approve"); p.add_argument("approval_id", type=int); p.set_defaults(func=cmd_approve)
    p = sub.add_parser("reject"); p.add_argument("approval_id", type=int); p.set_defaults(func=cmd_reject)

    p = sub.add_parser("reflections"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_reflections)
    p = sub.add_parser("skills"); p.add_argument("--limit", type=int, default=20); p.set_defaults(func=cmd_skills)
    p = sub.add_parser("skill"); skill_sub = p.add_subparsers(dest="skill_command", required=True)
    p_skill_add = skill_sub.add_parser("add"); p_skill_add.add_argument("name"); p_skill_add.add_argument("trigger"); p_skill_add.add_argument("procedure", nargs="+"); p_skill_add.add_argument("--tags", nargs="*"); p_skill_add.set_defaults(func=cmd_skill_add)

    p = sub.add_parser("eval"); eval_sub = p.add_subparsers(dest="eval_command", required=True)
    p_eval_list = eval_sub.add_parser("list"); p_eval_list.add_argument("suite", nargs="?"); p_eval_list.set_defaults(func=cmd_eval_list)
    p_eval_run = eval_sub.add_parser("run"); p_eval_run.add_argument("suite", nargs="?"); p_eval_run.set_defaults(func=cmd_eval_run)
    p_eval_runs = eval_sub.add_parser("runs"); p_eval_runs.add_argument("--limit", type=int, default=20); p_eval_runs.set_defaults(func=cmd_eval_runs)

    p = sub.add_parser("tool"); tool_sub = p.add_subparsers(dest="tool_command", required=True)
    p_tool_list = tool_sub.add_parser("list"); p_tool_list.set_defaults(func=cmd_tool_list)
    p_tool_run = tool_sub.add_parser("run"); p_tool_run.add_argument("name"); p_tool_run.set_defaults(func=cmd_tool_run)
    p = sub.add_parser("snapshot"); p.set_defaults(func=cmd_snapshot)
    p = sub.add_parser("backup"); p.add_argument("--label", default="manual"); p.set_defaults(func=cmd_backup)
    p = sub.add_parser("audit"); p.set_defaults(func=cmd_audit)
    p = sub.add_parser("self-check"); p.add_argument("area", choices=["bridge", "memory", "reflection", "tool"]); p.set_defaults(func=cmd_self_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
