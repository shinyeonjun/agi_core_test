from __future__ import annotations

import argparse
import json
from typing import Sequence
from uuid import uuid4

from agent import __version__
from agent.bridge.reports import notify_action, notify_activity_summary, notify_daily_summary, notify_observation_dashboard, notify_test_summary, notify_test_update
from agent.core.autonomy import arm_catastrophic_destruction, disarm_catastrophic_destruction, get_autonomy_state, set_autonomy_profile
from agent.core.approvals import ApprovalStore
from agent.core.database import get_schema_version, init_db
from agent.core.events import list_events, log_event
from agent.core.goals import cleanup_noise_goals, list_goals, mark_goal_done
from agent.core.goal_generator import add_root_objective, generate_goal_candidates, list_goal_candidates, list_root_objectives, seed_default_objectives, set_objective_enabled
from agent.core.learner import list_reflections, list_skills, upsert_skill, update_after_turn
from agent.core.metrics import collect_metrics
from agent.core.observability import action_failure_breakdown, action_observation, latest_decision_traces, observability_snapshot, task_observation
from agent.core.operating_intelligence import action_critics, memory_hygiene_candidates, operating_snapshot, ranked_goals, refresh_goal_priorities, skill_candidates
from agent.core.pipeline import run_talk
from agent.core.policy import ActionProposal, PolicyEngine
from agent.core.self_map import latest_self_map, refresh_self_map, self_map_brief
from agent.core.state import load_state, save_state
from agent.core.style import add_style_example, apply_style_feedback, get_active_style_profile, list_style_examples, list_style_feedback, seed_default_style_profile, style_directives
from agent.core.task_lifecycle import list_task_lifecycle, task_lifecycle_summary
from agent.core.task_queue import doctor_tasks, list_tasks as list_queued_tasks, task_status_counts
from agent.eval.harness import list_eval_runs, list_tasks as list_eval_tasks, run_suite
from agent.language.engine import get_language_engine, interpret_user_message, language_cache_stats, list_interpretation_logs
from agent.memory.store import add_memory, list_memories, rebuild_memory_fts, search_memories
from agent.ops.backup import create_backup
from agent.scheduler.tick import run_tick
from agent.tools.system_readonly import READ_ONLY_COMMANDS, run_readonly, system_snapshot
from agent.tools.full_device import get_action_run, list_action_runs, run_action
from agent.lab.codex_bridge import write_codex_lab_context
from agent.lab.planner import lab_report, run_lab_tick, run_lab_tick_if_enabled, run_user_task, sync_open_goals_to_tasks
from agent.lab.proposals import list_action_proposals
from agent.workspace.executor import create_project_spec, create_status_report, ensure_workspace, write_text_artifact
from agent.workspace.store import list_project_specs, list_workspace_artifacts


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def cmd_init(_args: argparse.Namespace) -> int:
    init_db()
    seed_default_style_profile()
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


def cmd_goal_generate(args: argparse.Namespace) -> int:
    print_json(generate_goal_candidates(dry_run=bool(args.dry_run)))
    return 0


def cmd_goal_candidates(args: argparse.Namespace) -> int:
    print_json(list_goal_candidates(limit=args.limit, status=args.status))
    return 0


def cmd_goal_cleanup_noise(_args: argparse.Namespace) -> int:
    print_json(cleanup_noise_goals())
    return 0


def cmd_objective(args: argparse.Namespace) -> int:
    if args.objective_command == "seed":
        print_json(seed_default_objectives())
        return 0
    if args.objective_command == "list":
        print_json(list_root_objectives(include_disabled=bool(args.all), limit=args.limit))
        return 0
    if args.objective_command == "add":
        objective_id = add_root_objective(args.title, args.description, args.type, args.priority, args.cooldown_seconds)
        print_json({"id": objective_id})
        return 0
    if args.objective_command == "enable":
        ok = set_objective_enabled(args.objective_id, True)
        print("enabled" if ok else "not found")
        return 0 if ok else 1
    if args.objective_command == "disable":
        ok = set_objective_enabled(args.objective_id, False)
        print("disabled" if ok else "not found")
        return 0 if ok else 1
    raise ValueError(f"unknown objective command: {args.objective_command}")


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


def cmd_style(args: argparse.Namespace) -> int:
    if args.style_command == "show":
        profile = get_active_style_profile()
        print_json({
            "id": profile.get("id"),
            "name": profile.get("name"),
            "confidence": profile.get("confidence"),
            "profile": profile.get("profile"),
            "directives": style_directives(profile),
        })
        return 0
    if args.style_command == "feedback":
        result = apply_style_feedback(args.text)
        print_json(result or {"applied": False, "reason": "no_style_feedback_detected"})
        return 0
    if args.style_command == "feedbacks":
        print_json(list_style_feedback(args.limit))
        return 0
    if args.style_command == "examples":
        print_json(list_style_examples(args.limit, label=args.label))
        return 0
    if args.style_command == "add-example":
        example_id = add_style_example(
            label=args.label,
            input_text=args.input,
            good_response=args.good,
            bad_response=args.bad,
            reason=args.reason,
            tags=args.tags or [],
        )
        print_json({"id": example_id})
        return 0
    raise ValueError(f"unknown style command: {args.style_command}")


def cmd_language(args: argparse.Namespace) -> int:
    if args.language_command == "interpret":
        print_json(interpret_user_message(args.text, {"surface": "cli"}))
        return 0
    if args.language_command == "logs":
        print_json(list_interpretation_logs(args.limit))
        return 0
    if args.language_command == "engine":
        engine = get_language_engine()
        print_json({"engine": getattr(engine, "name", "unknown")})
        return 0
    if args.language_command == "cache-stats":
        print_json(language_cache_stats())
        return 0
    raise ValueError(f"unknown language command: {args.language_command}")


def cmd_eval_list(args: argparse.Namespace) -> int:
    print_json([task.__dict__ for task in list_eval_tasks(args.suite)])
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    result = run_suite(args.suite, isolated=not args.live_db)
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


def cmd_notify(args: argparse.Namespace) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    if args.notify_command == "test-summary":
        print_json(notify_test_summary(dry_run=dry_run))
        return 0
    if args.notify_command == "test-update":
        print_json(notify_test_update(dry_run=dry_run))
        return 0
    if args.notify_command == "daily-summary":
        print_json(notify_daily_summary(dry_run=dry_run))
        return 0
    if args.notify_command == "activity-summary":
        print_json(notify_activity_summary(dry_run=dry_run, force=bool(getattr(args, "force", False))))
        return 0
    if args.notify_command == "observation-dashboard":
        print_json(notify_observation_dashboard(dry_run=dry_run, force=bool(getattr(args, "force", False))))
        return 0
    if args.notify_command == "action":
        print_json(notify_action(args.action_id, dry_run=dry_run))
        return 0
    raise ValueError(f"unknown notify command: {args.notify_command}")


def cmd_metrics(args: argparse.Namespace) -> int:
    metrics = collect_metrics()
    if args.json:
        print_json(metrics)
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")
    return 0


def cmd_self_map(args: argparse.Namespace) -> int:
    if args.self_map_command == "refresh":
        result = refresh_self_map()
        if args.full:
            print_json(result)
        else:
            print_json({key: result[key] for key in ["id", "changed", "fingerprint", "summary"]})
        return 0
    if args.self_map_command == "show":
        item = latest_self_map()
        if not item:
            print_json({"available": False, "reason": "no_self_map_yet"})
            return 0
        print_json(item if args.full else self_map_brief())
        return 0
    raise ValueError(f"unknown self-map command: {args.self_map_command}")


def cmd_autonomy(args: argparse.Namespace) -> int:
    if args.autonomy_command == "show":
        print_json(get_autonomy_state())
        return 0
    if args.autonomy_command == "set":
        print_json(set_autonomy_profile(args.profile))
        return 0
    if args.autonomy_command == "arm-destruction":
        print_json(arm_catastrophic_destruction(args.ttl))
        return 0
    if args.autonomy_command == "disarm-destruction":
        print_json(disarm_catastrophic_destruction())
        return 0
    raise ValueError(f"unknown autonomy command: {args.autonomy_command}")


def cmd_action(args: argparse.Namespace) -> int:
    if args.action_command == "run":
        print_json(run_action(args.command, cwd=args.cwd, timeout_seconds=args.timeout, goal_id=args.goal_id, use_shell=args.shell))
        return 0
    if args.action_command == "history":
        print_json(list_action_runs(args.limit))
        return 0
    if args.action_command == "show":
        item = get_action_run(args.action_id)
        print_json(item or {})
        return 0 if item else 1
    raise ValueError(f"unknown action command: {args.action_command}")


def cmd_lab(args: argparse.Namespace) -> int:
    if args.lab_command == "tick":
        print_json(run_lab_tick())
        return 0
    if args.lab_command == "tick-if-enabled":
        print_json(run_lab_tick_if_enabled(notify=bool(getattr(args, "notify", False))))
        return 0
    if args.lab_command == "proposals":
        print_json(list_action_proposals(args.limit, args.status))
        return 0
    if args.lab_command == "report":
        print_json(lab_report(args.limit))
        return 0
    if args.lab_command == "codex-plan":
        print_json(write_codex_lab_context(args.limit))
        return 0
    raise ValueError(f"unknown lab command: {args.lab_command}")


def cmd_tasks(args: argparse.Namespace) -> int:
    if args.tasks_command == "list":
        print_json(list_queued_tasks(args.limit, status=args.status, queue_type=args.queue_type))
        return 0
    if args.tasks_command == "counts":
        print_json(task_status_counts())
        return 0
    if args.tasks_command == "sync":
        print_json(sync_open_goals_to_tasks())
        return 0
    if args.tasks_command == "run-user":
        print_json(run_user_task(args.task_id))
        return 0
    if args.tasks_command == "doctor":
        print_json(doctor_tasks(max_age_seconds=args.max_age_seconds))
        return 0
    if args.tasks_command == "lifecycle":
        task = next((row for row in list_queued_tasks(200) if int(row.get("id", -1)) == int(args.task_id)), None)
        if task:
            print_json(task_lifecycle_summary(task, limit=args.limit))
        else:
            print_json({"task_id": args.task_id, "events": list_task_lifecycle(args.task_id, limit=args.limit), "available": False})
        return 0
    raise ValueError(f"unknown tasks command: {args.tasks_command}")


def cmd_observe(args: argparse.Namespace) -> int:
    if args.observe_command == "snapshot":
        print_json(observability_snapshot(args.limit))
        return 0
    if args.observe_command == "actions":
        actions = list_action_runs(args.limit)
        print_json({"items": [action_observation(row) for row in actions], "failure_breakdown": action_failure_breakdown(actions)})
        return 0
    if args.observe_command == "tasks":
        print_json({"items": [task_observation(row) for row in list_queued_tasks(args.limit)]})
        return 0
    if args.observe_command == "decisions":
        print_json({"items": latest_decision_traces(args.limit)})
        return 0
    raise ValueError(f"unknown observe command: {args.observe_command}")


def cmd_intelligence(args: argparse.Namespace) -> int:
    if args.intelligence_command == "snapshot":
        print_json(operating_snapshot(persist=bool(args.persist), refresh_priorities=bool(args.refresh)))
        return 0
    if args.intelligence_command == "priorities":
        result = refresh_goal_priorities(limit=args.limit) if args.refresh else {"items": ranked_goals(limit=args.limit), "changed": 0}
        print_json(result)
        return 0
    if args.intelligence_command == "critics":
        print_json({"items": action_critics(limit=args.limit)})
        return 0
    if args.intelligence_command == "memory":
        print_json({"items": memory_hygiene_candidates(limit=args.limit)})
        return 0
    if args.intelligence_command == "skills":
        print_json({"items": skill_candidates(limit=args.limit)})
        return 0
    raise ValueError(f"unknown intelligence command: {args.intelligence_command}")


def cmd_workspace(args: argparse.Namespace) -> int:
    if args.workspace_command == "init":
        print_json(ensure_workspace())
        return 0
    if args.workspace_command == "report":
        print_json(create_status_report(args.title, metrics=collect_metrics()))
        return 0
    if args.workspace_command == "write":
        print_json(write_text_artifact(args.section, args.path, args.content, "file", args.title))
        return 0
    if args.workspace_command == "project":
        print_json(create_project_spec(args.title, args.objective, args.notes or []))
        return 0
    if args.workspace_command == "artifacts":
        print_json(list_workspace_artifacts(args.limit, args.type))
        return 0
    if args.workspace_command == "projects":
        print_json(list_project_specs(args.limit, args.status))
        return 0
    raise ValueError(f"unknown workspace command: {args.workspace_command}")


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
        nonce = uuid4().hex[:8]
        title = f"digital agi korean memory {nonce}"
        content = f"Core remembers \ub514\uc9c0\ud138 AGI context. nonce={nonce}"
        memory_id = add_memory(title, content, memory_type="project_context", tags=["digital_agi", "core", nonce], importance=0.93)
        results = search_memories(nonce, limit=5)
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
    if args.area == "workspace":
        workspace = ensure_workspace()
        artifact = write_text_artifact("scratch", f"self-check-{uuid4().hex[:8]}.txt", "workspace self-check ok\n", "self_check", "workspace self-check")
        result = {"ok": artifact["relative_path"].startswith("scratch/"), "root": workspace["root"], "artifact_id": artifact["id"], "relative_path": artifact["relative_path"]}
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
    p_goal_generate = goal_sub.add_parser("generate"); p_goal_generate.add_argument("--dry-run", action="store_true"); p_goal_generate.set_defaults(func=cmd_goal_generate)
    p_goal_candidates = goal_sub.add_parser("candidates"); p_goal_candidates.add_argument("--limit", type=int, default=20); p_goal_candidates.add_argument("--status"); p_goal_candidates.set_defaults(func=cmd_goal_candidates)
    p_goal_cleanup = goal_sub.add_parser("cleanup-noise"); p_goal_cleanup.set_defaults(func=cmd_goal_cleanup_noise)

    p = sub.add_parser("objective"); objective_sub = p.add_subparsers(dest="objective_command", required=True)
    p_objective_seed = objective_sub.add_parser("seed"); p_objective_seed.set_defaults(func=cmd_objective)
    p_objective_list = objective_sub.add_parser("list"); p_objective_list.add_argument("--all", action="store_true"); p_objective_list.add_argument("--limit", type=int, default=50); p_objective_list.set_defaults(func=cmd_objective)
    p_objective_add = objective_sub.add_parser("add"); p_objective_add.add_argument("title"); p_objective_add.add_argument("description"); p_objective_add.add_argument("--type", required=True); p_objective_add.add_argument("--priority", type=float, default=0.5); p_objective_add.add_argument("--cooldown-seconds", type=int, default=21600); p_objective_add.set_defaults(func=cmd_objective)
    p_objective_enable = objective_sub.add_parser("enable"); p_objective_enable.add_argument("objective_id", type=int); p_objective_enable.set_defaults(func=cmd_objective)
    p_objective_disable = objective_sub.add_parser("disable"); p_objective_disable.add_argument("objective_id", type=int); p_objective_disable.set_defaults(func=cmd_objective)

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
    p = sub.add_parser("style"); style_sub = p.add_subparsers(dest="style_command", required=True)
    p_style_show = style_sub.add_parser("show"); p_style_show.set_defaults(func=cmd_style)
    p_style_feedback = style_sub.add_parser("feedback"); p_style_feedback.add_argument("text"); p_style_feedback.set_defaults(func=cmd_style)
    p_style_feedbacks = style_sub.add_parser("feedbacks"); p_style_feedbacks.add_argument("--limit", type=int, default=20); p_style_feedbacks.set_defaults(func=cmd_style)
    p_style_examples = style_sub.add_parser("examples"); p_style_examples.add_argument("--limit", type=int, default=10); p_style_examples.add_argument("--label"); p_style_examples.set_defaults(func=cmd_style)
    p_style_example = style_sub.add_parser("add-example"); p_style_example.add_argument("--label", required=True); p_style_example.add_argument("--input"); p_style_example.add_argument("--good"); p_style_example.add_argument("--bad"); p_style_example.add_argument("--reason"); p_style_example.add_argument("--tags", nargs="*"); p_style_example.set_defaults(func=cmd_style)
    p = sub.add_parser("language"); language_sub = p.add_subparsers(dest="language_command", required=True)
    p_language_interpret = language_sub.add_parser("interpret"); p_language_interpret.add_argument("text"); p_language_interpret.set_defaults(func=cmd_language)
    p_language_logs = language_sub.add_parser("logs"); p_language_logs.add_argument("--limit", type=int, default=20); p_language_logs.set_defaults(func=cmd_language)
    p_language_engine = language_sub.add_parser("engine"); p_language_engine.set_defaults(func=cmd_language)
    p_language_cache = language_sub.add_parser("cache-stats"); p_language_cache.set_defaults(func=cmd_language)

    p = sub.add_parser("eval"); eval_sub = p.add_subparsers(dest="eval_command", required=True)
    p_eval_list = eval_sub.add_parser("list"); p_eval_list.add_argument("suite", nargs="?"); p_eval_list.set_defaults(func=cmd_eval_list)
    p_eval_run = eval_sub.add_parser("run"); p_eval_run.add_argument("suite", nargs="?"); p_eval_run.add_argument("--live-db", action="store_true"); p_eval_run.set_defaults(func=cmd_eval_run)
    p_eval_runs = eval_sub.add_parser("runs"); p_eval_runs.add_argument("--limit", type=int, default=20); p_eval_runs.set_defaults(func=cmd_eval_runs)

    p = sub.add_parser("tool"); tool_sub = p.add_subparsers(dest="tool_command", required=True)
    p_tool_list = tool_sub.add_parser("list"); p_tool_list.set_defaults(func=cmd_tool_list)
    p_tool_run = tool_sub.add_parser("run"); p_tool_run.add_argument("name"); p_tool_run.set_defaults(func=cmd_tool_run)
    p = sub.add_parser("snapshot"); p.set_defaults(func=cmd_snapshot)
    p = sub.add_parser("backup"); p.add_argument("--label", default="manual"); p.set_defaults(func=cmd_backup)
    p = sub.add_parser("metrics"); p.add_argument("--json", action="store_true"); p.set_defaults(func=cmd_metrics)
    p = sub.add_parser("self-map"); self_map_sub = p.add_subparsers(dest="self_map_command", required=True)
    p_self_map_refresh = self_map_sub.add_parser("refresh"); p_self_map_refresh.add_argument("--full", action="store_true"); p_self_map_refresh.set_defaults(func=cmd_self_map)
    p_self_map_show = self_map_sub.add_parser("show"); p_self_map_show.add_argument("--full", action="store_true"); p_self_map_show.set_defaults(func=cmd_self_map)
    p = sub.add_parser("notify"); notify_sub = p.add_subparsers(dest="notify_command", required=True)
    p_notify_summary = notify_sub.add_parser("test-summary"); p_notify_summary.add_argument("--dry-run", action="store_true"); p_notify_summary.set_defaults(func=cmd_notify)
    p_notify_update = notify_sub.add_parser("test-update"); p_notify_update.add_argument("--dry-run", action="store_true"); p_notify_update.set_defaults(func=cmd_notify)
    p_notify_daily = notify_sub.add_parser("daily-summary"); p_notify_daily.add_argument("--dry-run", action="store_true"); p_notify_daily.set_defaults(func=cmd_notify)
    p_notify_activity = notify_sub.add_parser("activity-summary"); p_notify_activity.add_argument("--dry-run", action="store_true"); p_notify_activity.add_argument("--force", action="store_true"); p_notify_activity.set_defaults(func=cmd_notify)
    p_notify_observation = notify_sub.add_parser("observation-dashboard"); p_notify_observation.add_argument("--dry-run", action="store_true"); p_notify_observation.add_argument("--force", action="store_true"); p_notify_observation.set_defaults(func=cmd_notify)
    p_notify_action = notify_sub.add_parser("action"); p_notify_action.add_argument("action_id", type=int); p_notify_action.add_argument("--dry-run", action="store_true"); p_notify_action.set_defaults(func=cmd_notify)
    p = sub.add_parser("autonomy"); autonomy_sub = p.add_subparsers(dest="autonomy_command", required=True)
    p_auto_show = autonomy_sub.add_parser("show"); p_auto_show.set_defaults(func=cmd_autonomy)
    p_auto_set = autonomy_sub.add_parser("set"); p_auto_set.add_argument("profile", choices=["safe", "workspace", "full_device_lab"]); p_auto_set.set_defaults(func=cmd_autonomy)
    p_auto_arm = autonomy_sub.add_parser("arm-destruction"); p_auto_arm.add_argument("--ttl", type=int, default=300); p_auto_arm.set_defaults(func=cmd_autonomy)
    p_auto_disarm = autonomy_sub.add_parser("disarm-destruction"); p_auto_disarm.set_defaults(func=cmd_autonomy)
    p = sub.add_parser("action"); action_sub = p.add_subparsers(dest="action_command", required=True)
    p_action_run = action_sub.add_parser("run"); p_action_run.add_argument("command"); p_action_run.add_argument("--cwd"); p_action_run.add_argument("--timeout", type=int, default=15); p_action_run.add_argument("--goal-id", type=int); p_action_run.add_argument("--shell", action="store_true"); p_action_run.set_defaults(func=cmd_action)
    p_action_history = action_sub.add_parser("history"); p_action_history.add_argument("--limit", type=int, default=20); p_action_history.set_defaults(func=cmd_action)
    p_action_show = action_sub.add_parser("show"); p_action_show.add_argument("action_id", type=int); p_action_show.set_defaults(func=cmd_action)
    p = sub.add_parser("lab"); lab_sub = p.add_subparsers(dest="lab_command", required=True)
    p_lab_tick = lab_sub.add_parser("tick"); p_lab_tick.set_defaults(func=cmd_lab)
    p_lab_tick_enabled = lab_sub.add_parser("tick-if-enabled"); p_lab_tick_enabled.add_argument("--notify", action="store_true"); p_lab_tick_enabled.set_defaults(func=cmd_lab)
    p_lab_proposals = lab_sub.add_parser("proposals"); p_lab_proposals.add_argument("--limit", type=int, default=20); p_lab_proposals.add_argument("--status"); p_lab_proposals.set_defaults(func=cmd_lab)
    p_lab_report = lab_sub.add_parser("report"); p_lab_report.add_argument("--limit", type=int, default=10); p_lab_report.set_defaults(func=cmd_lab)
    p_lab_codex = lab_sub.add_parser("codex-plan"); p_lab_codex.add_argument("--limit", type=int, default=10); p_lab_codex.set_defaults(func=cmd_lab)
    p = sub.add_parser("tasks"); tasks_sub = p.add_subparsers(dest="tasks_command", required=True)
    p_tasks_list = tasks_sub.add_parser("list"); p_tasks_list.add_argument("--limit", type=int, default=20); p_tasks_list.add_argument("--status"); p_tasks_list.add_argument("--queue-type", choices=["user", "autonomous"]); p_tasks_list.set_defaults(func=cmd_tasks)
    p_tasks_counts = tasks_sub.add_parser("counts"); p_tasks_counts.set_defaults(func=cmd_tasks)
    p_tasks_sync = tasks_sub.add_parser("sync"); p_tasks_sync.set_defaults(func=cmd_tasks)
    p_tasks_run_user = tasks_sub.add_parser("run-user"); p_tasks_run_user.add_argument("task_id", type=int); p_tasks_run_user.set_defaults(func=cmd_tasks)
    p_tasks_doctor = tasks_sub.add_parser("doctor"); p_tasks_doctor.add_argument("--max-age-seconds", type=int, default=1800); p_tasks_doctor.set_defaults(func=cmd_tasks)
    p_tasks_lifecycle = tasks_sub.add_parser("lifecycle"); p_tasks_lifecycle.add_argument("task_id", type=int); p_tasks_lifecycle.add_argument("--limit", type=int, default=50); p_tasks_lifecycle.set_defaults(func=cmd_tasks)
    p = sub.add_parser("observe"); observe_sub = p.add_subparsers(dest="observe_command", required=True)
    p_observe_snapshot = observe_sub.add_parser("snapshot"); p_observe_snapshot.add_argument("--limit", type=int, default=10); p_observe_snapshot.set_defaults(func=cmd_observe)
    p_observe_actions = observe_sub.add_parser("actions"); p_observe_actions.add_argument("--limit", type=int, default=10); p_observe_actions.set_defaults(func=cmd_observe)
    p_observe_tasks = observe_sub.add_parser("tasks"); p_observe_tasks.add_argument("--limit", type=int, default=10); p_observe_tasks.set_defaults(func=cmd_observe)
    p_observe_decisions = observe_sub.add_parser("decisions"); p_observe_decisions.add_argument("--limit", type=int, default=5); p_observe_decisions.set_defaults(func=cmd_observe)
    p = sub.add_parser("intelligence"); intelligence_sub = p.add_subparsers(dest="intelligence_command", required=True)
    p_int_snapshot = intelligence_sub.add_parser("snapshot"); p_int_snapshot.add_argument("--persist", action="store_true"); p_int_snapshot.add_argument("--refresh", action="store_true"); p_int_snapshot.set_defaults(func=cmd_intelligence)
    p_int_priorities = intelligence_sub.add_parser("priorities"); p_int_priorities.add_argument("--limit", type=int, default=20); p_int_priorities.add_argument("--refresh", action="store_true"); p_int_priorities.set_defaults(func=cmd_intelligence)
    p_int_critics = intelligence_sub.add_parser("critics"); p_int_critics.add_argument("--limit", type=int, default=20); p_int_critics.set_defaults(func=cmd_intelligence)
    p_int_memory = intelligence_sub.add_parser("memory"); p_int_memory.add_argument("--limit", type=int, default=20); p_int_memory.set_defaults(func=cmd_intelligence)
    p_int_skills = intelligence_sub.add_parser("skills"); p_int_skills.add_argument("--limit", type=int, default=20); p_int_skills.set_defaults(func=cmd_intelligence)
    p = sub.add_parser("workspace"); workspace_sub = p.add_subparsers(dest="workspace_command", required=True)
    p_ws_init = workspace_sub.add_parser("init"); p_ws_init.set_defaults(func=cmd_workspace)
    p_ws_report = workspace_sub.add_parser("report"); p_ws_report.add_argument("--title", default="Workspace status report"); p_ws_report.set_defaults(func=cmd_workspace)
    p_ws_write = workspace_sub.add_parser("write"); p_ws_write.add_argument("section"); p_ws_write.add_argument("path"); p_ws_write.add_argument("content"); p_ws_write.add_argument("--title"); p_ws_write.set_defaults(func=cmd_workspace)
    p_ws_project = workspace_sub.add_parser("project"); p_ws_project.add_argument("title"); p_ws_project.add_argument("objective"); p_ws_project.add_argument("--notes", nargs="*"); p_ws_project.set_defaults(func=cmd_workspace)
    p_ws_artifacts = workspace_sub.add_parser("artifacts"); p_ws_artifacts.add_argument("--limit", type=int, default=20); p_ws_artifacts.add_argument("--type"); p_ws_artifacts.set_defaults(func=cmd_workspace)
    p_ws_projects = workspace_sub.add_parser("projects"); p_ws_projects.add_argument("--limit", type=int, default=20); p_ws_projects.add_argument("--status"); p_ws_projects.set_defaults(func=cmd_workspace)
    p = sub.add_parser("audit"); p.set_defaults(func=cmd_audit)
    p = sub.add_parser("self-check"); p.add_argument("area", choices=["bridge", "memory", "reflection", "tool", "workspace"]); p.set_defaults(func=cmd_self_check)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
