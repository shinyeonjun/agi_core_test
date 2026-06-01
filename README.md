# agent-core

Digital AGI-oriented Local Stateful Agent Core for Orange Pi 5. This project does not claim AGI. It builds a testable runtime for persistent state, long-term memory, goals, policy, approvals, reflection, skills, evaluation, safe tool use, and renderer isolation.

## Current Scope

- v0.6-alpha Discord conversational bridge
- Goal deduplication and cooldown
- Reflection plus skill learner
- SQLite FTS5 memory search with LIKE fallback
- Evaluation harness
- CodexRenderer sanitized isolation boundary with fallback
- Read-only ToolExecutor and system snapshot
- systemd unit templates

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
```

## Discord

Put `DISCORD_BOT_TOKEN`, `DISCORD_ALLOWED_USER_IDS`, and optionally `DISCORD_ALLOWED_CHANNEL_IDS` in `.env`.

```bash
python -m agent.bridge.discord_bot --check-config
python -m agent.bridge.discord_bot
```

Discord connects to Core only. Discord input never executes shell commands directly. Risky requests are separated into policy and approval flow.
