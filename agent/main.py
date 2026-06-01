from __future__ import annotations

import argparse

from agent.cli.agentctl import main as agentctl_main
from agent.scheduler.tick import run_tick


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.main")
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "tick", "daemon"])
    args, rest = parser.parse_known_args()

    if args.mode == "tick":
        result = run_tick()
        print(result["message"])
        return 0
    if args.mode == "daemon":
        print("agentd daemon is not implemented in v0.1. Use agentctl tick.")
        return 0
    return agentctl_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
