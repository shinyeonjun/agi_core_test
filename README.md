# agent-core

Digital AGI-oriented Local Stateful Agent Core for Orange Pi 5. This project does not claim AGI. It builds a testable runtime for persistent state, long-term memory, goals, policy, approvals, reflection, skills, evaluation, safe tool use, and renderer isolation.

## Current Scope

- v0.11-alpha Agent OS kernel for Orange Pi
- Discord conversational control plane
- Codex-backed language interpretation and rendering with rule fallback
- Language interpretation cache and redacted interpretation logs
- Goal deduplication and cooldown
- Reflection plus skill learner
- SQLite FTS5 memory search with LIKE fallback
- Evaluation harness
- CodexRenderer sanitized isolation boundary with fallback
- Read-only ToolExecutor and system snapshot
- WorkspaceExecutor bounded to `/home/ubuntu/agent_workspace`
- ProjectSpec and workspace artifact tracking
- Runtime self-map loop for host/service/git/config-presence awareness
- Split task queue for immediate user tasks and scheduled autonomous tasks
- Task lifecycle tracking with queued/planning/executing/verifying/reporting/learned phases
- SQLite task leases, idempotency keys, delayed scheduling fields, and task doctor recovery
- Operating intelligence snapshots for goal priority, action critic, memory hygiene, skill candidates, and next improvements
- Explicit DB migration status table and `agentctl db migrate/check`
- systemd unit templates

Version alignment:

- package: `0.11.0a0`
- schema: `0.11.0-alpha`
- runtime scope: `v0.11-alpha`

## Basic Commands

```bash
cd /home/ubuntu/agent_core
source venv/bin/activate
agentctl init
agentctl state
agentctl talk "Core next step?"
agentctl tick
agentctl policy-check "apt-get install nginx"
agentctl eval run policy
agentctl audit
agentctl db check
agentctl db migrate
agentctl metrics
agentctl self-map refresh
agentctl self-map show
agentctl workspace init
agentctl workspace report --title "Daily workspace status"
agentctl autonomy show
agentctl action history
agentctl tasks counts
agentctl tasks list --queue-type user
agentctl intelligence snapshot --persist --refresh
```

## Discord

Put `DISCORD_BOT_TOKEN`, `DISCORD_ALLOWED_USER_IDS`, and optionally `DISCORD_ALLOWED_CHANNEL_IDS` in `.env`.

```bash
python -m agent.bridge.discord_bot --check-config
python -m agent.bridge.discord_bot
```

Discord connects to Core only. Discord input never executes shell commands directly. Risky requests are separated into policy and approval flow.

## Full Device Lab

Default autonomy profile is `safe`. `full_device_lab` must be enabled explicitly before `agentctl action run` can execute local commands. Even in lab mode, secret access, credential exfiltration, remote script execution, network scanning, payment, and cloud creation patterns stay denied.

```bash
agentctl autonomy show
agentctl autonomy set full_device_lab
agentctl action run "printf lab-ok"
agentctl action history
agentctl autonomy set safe
```

## Runtime Self-Map

`self-map` is a safe runtime body map. It records where Core is running, recent git state, service/timer state, autonomy profile, schema/eval summary, and config presence booleans. It does not read or store `.env` values, tokens, private keys, or passwords.

The normal tick loop refreshes it on cooldown, so Discord summaries and chat rendering can use recent verified runtime context without re-running heavy checks on every message.

```bash
agentctl self-map refresh
agentctl self-map show
```

## Task Queue Split

User-directed work and autonomous scheduled work are intentionally separate.

- Discord user directives enqueue `user` tasks and the chat bridge runs that user task immediately when policy allows it.
- The lab scheduler only claims `autonomous` tasks. It does not consume user tasks.
- Memory, events, reflections, goals, self-map, policy, and approvals remain shared state.

```bash
agentctl tasks counts
agentctl tasks list --queue-type user
agentctl tasks list --queue-type autonomous
agentctl tasks run-user <task_id>
agentctl tasks doctor
```

Task queue rows carry a lease (`locked_until`, `locked_by`), idempotency key, scheduling fields (`not_before`, `due_at`), and retry budget. This keeps user-triggered work and scheduler work separate while still sharing memory, goals, approvals, events, reflections, and operating reviews.

## Operating Intelligence

`agentctl intelligence` exposes Core's current operating review loop. It does not make the model fine-tune itself. It records and reports structured signals that help the Core decide what to improve next.

```bash
agentctl intelligence snapshot --persist --refresh
agentctl intelligence priorities --refresh
agentctl intelligence critics
agentctl intelligence memory
agentctl intelligence skills
```

The Discord observation dashboard includes the same high-level signals in Korean so the summary channel reads like a control room instead of raw logs.

## Database Migrations

Fresh databases are created from `agent/memory/schema.sql`. Existing databases are repaired through idempotent migrations recorded in `schema_migrations`.

```bash
agentctl db check
agentctl db migrate
```

Migrations are intentionally conservative. They add missing columns, indexes, and tables without reading secrets or rewriting user data.

## Codex Runtime Tuning

Chat and language interpretation use faster Codex settings. Code work should keep the stronger default model.

```bash
AGENT_CODEX_LANGUAGE_MODEL=gpt-5.3-codex-spark
AGENT_CODEX_LANGUAGE_REASONING=low
AGENT_CODEX_LANGUAGE_TIMEOUT=20
AGENT_CODEX_RENDERER_MODEL=gpt-5.3-codex-spark
AGENT_CODEX_RENDERER_REASONING=low
AGENT_CODEX_RENDERER_TIMEOUT=20
AGENT_CODEX_WORK_MODEL=gpt-5.5
AGENT_CODEX_WORK_REASONING=medium
```
