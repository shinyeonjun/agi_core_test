from __future__ import annotations

from collections.abc import Callable

from .base import MicroWorld
from .lock_world import LockWorld
from .memory_maze import MemoryMaze
from .model_needed import LockTrapWorld, MazeHazardWorld, ToolPreconditionWorld
from .tool_world import ToolWorld

EnvFactory = Callable[[], MicroWorld]

_REGISTRY: dict[str, EnvFactory] = {
    "lock.train": lambda: LockWorld("lock.train", "train", (1, 2, 3), 6),
    "lock.test": lambda: LockWorld("lock.test", "test", (2, 4, 1), 7),
    "tool.train": lambda: ToolWorld("tool.train", "train", 5),
    "tool.test": lambda: ToolWorld("tool.test", "test", 6),
    "maze.train": lambda: MemoryMaze("maze.train", "train", "blue", 7),
    "maze.test": lambda: MemoryMaze("maze.test", "test", "green", 8),
    "lock.trap_probe": lambda: LockTrapWorld("lock.trap_probe", "probe", (2, 4, 1), 8, True),
    "maze.hazard_probe": lambda: MazeHazardWorld("maze.hazard_probe", "probe", "green", 7, ("red", "blue", "green"), True),
    "tool.precondition_probe": lambda: ToolPreconditionWorld("tool.precondition_probe", "probe", 6, ("read", "summarize"), False),
}


def make_env(name: str) -> MicroWorld:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown environment: {name}") from exc


def list_envs(split: str = "all") -> list[str]:
    names = sorted(_REGISTRY)
    if split == "all":
        return names
    return [name for name in names if name.endswith(f".{split}")]
