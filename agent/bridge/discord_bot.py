from __future__ import annotations

import argparse
import atexit
import os
from pathlib import Path

from agent.bridge.auth import DiscordAuthConfig
from agent.bridge.router import DiscordEvent, route_discord_event
from agent.config.defaults import data_dir, env_path


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "posix":
        return pid == os.getpid()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _release_single_instance_lock(path: Path) -> None:
    try:
        if path.exists() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def _acquire_single_instance_lock() -> Path | None:
    lock_path = Path(os.environ.get("AGENT_DISCORD_LOCK_PATH", data_dir() / "discord_bot.pid")).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing_pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                existing_pid = 0
            if _pid_alive(existing_pid):
                return None
            try:
                lock_path.unlink()
            except OSError:
                return None
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        atexit.register(_release_single_instance_lock, lock_path)
        return lock_path


def run_bot() -> int:
    load_env_file()
    lock_path = _acquire_single_instance_lock()
    if lock_path is None:
        print("agent-core discord bridge is already running; refusing duplicate instance.")
        return 2
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
