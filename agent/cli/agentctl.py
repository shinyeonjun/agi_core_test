from __future__ import annotations

import argparse
import json
from typing import Sequence

from agent import __version__
from agent.core.approvals import ApprovalStore
from agent.core.database import init_db
from agent.core.decision import build_talk_decision
from agent.core.decisions import record_decision
from agent.core.events import list_events, log_event
from agent.core.goals import list_goals, mark_goal_done
from agent.core.policy import ActionProposal, PolicyEngine
from agent.core.state import load_state, mark_user_interaction, save_state
from agent.memory.store import add_memory, list_memories, search_memories
from agent.renderer.fallback_renderer import render
from agent.renderer.validator import validate_output
from agent.scheduler.tick import run_tick


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def cmd_init(_args: argparse.Namespace) -> int:
    init_db()
    state = load_state()
    save_state(state)
    print("agent-core v0.2-pre \ucd08\uae30\ud654 \uc644\ub8cc")
    return 0


def cmd_state(_args: argparse.Namespace) -> int:
    print_json(load_state())
    return 0


def cmd_talk(args: argparse.Namespace) -> int:
    init_db()
    mark_user_interaction()
    user_event_id = log_event("user", "user_message", args.message, importance=0.8)
    decision = build_talk_decision(args.message)
    decision["source_event_id"] = user_event_id
    decision_row_id = record_decision(decision)
    log_event(
        "core",
        "decision_created",
        json.dumps(decision, ensure_ascii=False),
        {"goal_id": decision["selected_goal_id"], "decision_id": decision_row_id},
        0.7,
    )
    output = render(decision)
    validation = validate_output(output, decision.get("must_include"), decision.get("must_not_include"))
    if not validation["ok"]:
        output = "v0.1 fallback renderer \uac80\uc99d \uc2e4\ud328. Core decision\uc740 \uc800\uc7a5\ub410\uc9c0\ub9cc \ucd9c\ub825\uc740 \ucd95\uc57d\ud588\uc5b4."
    log_event("core", "assistant_output", output, {"validation": validation}, 0.7)
    print(output)
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
    print("\uc644\ub8cc \ucc98\ub9ac\ub428" if ok else "\ub300\uc0c1 goal\uc744 \ucc3e\uc9c0 \ubabb\ud568")
    return 0 if ok else 1


def cmd_memories(args: argparse.Namespace) -> int:
    print_json(list_memories(args.limit))
    return 0


def cmd_memory_add(args: argparse.Namespace) -> int:
    memory_id = add_memory(
        title=args.title,
        content=args.content,
        memory_type=args.type,
        tags=args.tags or [],
        importance=args.importance,
        confidence=args.confidence,
    )
    print(f"memory \ucd94\uac00 \uc644\ub8cc: #{memory_id}")
    return 0


def cmd_memory_search(args: argparse.Namespace) -> int:
    print_json(search_memories(args.query, args.limit))
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
    if args.all:
        rows = store.list(status=None, limit=args.limit)
    else:
        rows = store.list_pending()[: args.limit]
    print_json(rows)
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    init_db()
    store = ApprovalStore()
    ok = store.approve(args.approval_id)
    log_event("approval", "approval_approved" if ok else "approval_approve_failed", str(args.approval_id), {"ok": ok}, 0.7)
    print("approved" if ok else "not found or not pending")
    return 0 if ok else 1


def cmd_reject(args: argparse.Namespace) -> int:
    init_db()
    store = ApprovalStore()
    ok = store.reject(args.approval_id)
    log_event("approval", "approval_rejected" if ok else "approval_reject_failed", str(args.approval_id), {"ok": ok}, 0.7)
    print("rejected" if ok else "not found or not pending")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentctl")
    parser.add_argument("--version", action="version", version=f"agent-core {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("state")
    p.set_defaults(func=cmd_state)

    p = sub.add_parser("talk")
    p.add_argument("message")
    p.set_defaults(func=cmd_talk)

    p = sub.add_parser("tick")
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("events")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("goals")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_goals)

    p = sub.add_parser("goal")
    goal_sub = p.add_subparsers(dest="goal_command", required=True)
    p_done = goal_sub.add_parser("done")
    p_done.add_argument("goal_id", type=int)
    p_done.set_defaults(func=cmd_goal_done)

    p = sub.add_parser("memories")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_memories)

    p = sub.add_parser("memory")
    memory_sub = p.add_subparsers(dest="memory_command", required=True)
    p_add = memory_sub.add_parser("add")
    p_add.add_argument("title")
    p_add.add_argument("content")
    p_add.add_argument("--type", default="fact")
    p_add.add_argument("--tags", nargs="*")
    p_add.add_argument("--importance", type=float, default=0.5)
    p_add.add_argument("--confidence", type=float, default=0.7)
    p_add.set_defaults(func=cmd_memory_add)

    p_search = memory_sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.set_defaults(func=cmd_memory_search)

    p = sub.add_parser("policy-check")
    p.add_argument("text")
    p.add_argument("--action-type", default="shell_text")
    p.set_defaults(func=cmd_policy_check)

    p = sub.add_parser("approvals")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_approvals)

    p = sub.add_parser("approve")
    p.add_argument("approval_id", type=int)
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("reject")
    p.add_argument("approval_id", type=int)
    p.set_defaults(func=cmd_reject)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
