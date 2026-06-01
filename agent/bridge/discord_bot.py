from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.config.defaults import env_path


def load_env_file(path: Path | None = None) -> None:
    path = path or env_path()
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def check_config() -> int:
    load_env_file()
    config = DiscordAuthConfig.from_env()
    problems = []
    if not os.environ.get("DISCORD_BOT_TOKEN"):
        problems.append("DISCORD_BOT_TOKEN missing")
    if not config.allowed_user_ids:
        problems.append("DISCORD_ALLOWED_USER_IDS missing")
    print("Discord config:")
    print(f"- allowed_users={len(config.allowed_user_ids)}")
    print(f"- allowed_channels={len(config.allowed_channel_ids)}")
    print(f"- chat_channel_configured={bool(config.chat_channel_id)}")
    print(f"- approval_channel_configured={bool(config.approval_channel_id)}")
    print(f"- max_response_chars={config.max_response_chars}")
    if problems:
        print("CONFIG_FAIL: " + ", ".join(problems))
        return 1
    print("CONFIG_OK")
    return 0


def run_bot() -> int:
    load_env_file()
    try:
        import discord
    except ImportError:
        print("discord.py is not installed. Run `pip install discord.py` inside the venv.")
        return 1
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("DISCORD_BOT_TOKEN is missing from .env.")
        return 1
    config = DiscordAuthConfig.from_env()
    intents = discord.Intents.default()
    intents.message_content = True
    intents.dm_messages = True
    intents.guild_messages = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"agent-core discord bridge logged in as {client.user}")

    @client.event
    async def on_message(message):
        bot_id = str(client.user.id) if client.user else ""
        was_mention = bool(bot_id and (f"<@{bot_id}>" in message.content or f"<@!{bot_id}>" in message.content))
        event = DiscordEvent(str(message.guild.id) if message.guild else None, str(message.channel.id), str(message.author.id), str(message.id), message.guild is None, was_mention, message.content, bool(message.author.bot))
        for chunk in route_discord_event(event, config):
            await message.reply(chunk, mention_author=False)

    client.run(token)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.bridge.discord_bot")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    return check_config() if args.check_config else run_bot()


if __name__ == "__main__":
    raise SystemExit(main())
