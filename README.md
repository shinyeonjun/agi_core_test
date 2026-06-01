# agent-core

Digital AGI-oriented Local Stateful Agent Core for Orange Pi 5. This project does not claim AGI. It builds a testable runtime for persistent state, long-term memory, goals, policy, approvals, reflection, skills, evaluation, safe tool use, and renderer isolation.

## Current Scope

- v0.7-alpha Discord conversational bridge
- Goal deduplication and cooldown
- Reflection plus skill learner
- SQLite FTS5 memory search with LIKE fallback
- Evaluation harness
- CodexRenderer sanitized isolation boundary with fallback
- Read-only ToolExecutor and system snapshot
- WorkspaceExecutor bounded to `/home/ubuntu/agent_workspace`
- ProjectSpec and workspace artifact tracking
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
agentctl metrics
agentctl workspace init
agentctl workspace report --title "Daily workspace status"
agentctl autonomy show
agentctl action history
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
