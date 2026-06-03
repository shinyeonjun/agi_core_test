from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from agent.config.defaults import project_root


DEFAULT_DEV_REQUIREMENTS = ("pytest",)
PYTHON_INSTALL_TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class DependencyCheck:
    requirement: str
    distribution: str
    installed: bool
    version: str | None
    source: str


def _requirement_name(requirement: str) -> str:
    text = requirement.strip()
    text = re.split(r"\s*[<>=!~]=?", text, maxsplit=1)[0]
    text = text.split("[", 1)[0]
    return text.strip()


def _dependency_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    configured = project_root()
    if (configured / "pyproject.toml").exists():
        return configured
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists():
        return cwd
    return configured


def project_python_requirements(root: Path | None = None, *, include_dev: bool = True) -> list[dict[str, str]]:
    root = _dependency_root(root)
    pyproject = root / "pyproject.toml"
    items: list[dict[str, str]] = []
    if pyproject.exists():
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        for requirement in data.get("project", {}).get("dependencies", []) or []:
            name = _requirement_name(str(requirement))
            if name:
                items.append({"requirement": str(requirement), "distribution": name, "source": "project"})
    if include_dev:
        for requirement in DEFAULT_DEV_REQUIREMENTS:
            items.append({"requirement": requirement, "distribution": _requirement_name(requirement), "source": "dev"})
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in items:
        key = (item["distribution"].lower(), item["source"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def check_python_dependencies(root: Path | None = None, *, include_dev: bool = True) -> list[dict[str, Any]]:
    checks: list[DependencyCheck] = []
    for item in project_python_requirements(root, include_dev=include_dev):
        distribution = item["distribution"]
        try:
            version = importlib.metadata.version(distribution)
            installed = True
        except importlib.metadata.PackageNotFoundError:
            version = None
            installed = False
        checks.append(
            DependencyCheck(
                requirement=item["requirement"],
                distribution=distribution,
                installed=installed,
                version=version,
                source=item["source"],
            )
        )
    return [asdict(check) for check in checks]


def missing_python_requirements(checks: Sequence[dict[str, Any]]) -> list[str]:
    return [str(item["requirement"]) for item in checks if not item.get("installed")]


def dependency_doctor(*, root: Path | None = None, include_dev: bool = True) -> dict[str, Any]:
    checks = check_python_dependencies(root, include_dev=include_dev)
    missing = missing_python_requirements(checks)
    return {
        "ok": not missing,
        "scope": "python_venv",
        "checks": checks,
        "missing": missing,
        "can_auto_install": bool(missing),
        "requires_approval": False,
        "blocked_actions": [
            "apt/system package install",
            "systemd changes",
            "secret/token based package sources",
        ],
        "install_command": [sys.executable, "-m", "pip", "install", *missing] if missing else [],
        "explanation": "venv 안의 Python 패키지만 자동 복구 대상이다. apt나 systemd 변경은 승인 게이트로 보낸다.",
    }


def install_missing_python_dependencies(*, root: Path | None = None, include_dev: bool = True) -> dict[str, Any]:
    root = _dependency_root(root)
    plan = dependency_doctor(root=root, include_dev=include_dev)
    command = plan.get("install_command") or []
    if not command:
        return {**plan, "installed": False, "returncode": 0, "stdout": "", "stderr": ""}
    completed = subprocess.run(
        [str(part) for part in command],
        cwd=root,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=PYTHON_INSTALL_TIMEOUT_SECONDS,
        check=False,
    )
    after = dependency_doctor(root=root, include_dev=include_dev)
    return {
        **after,
        "installed": completed.returncode == 0 and after["ok"],
        "returncode": completed.returncode,
        "stdout": (completed.stdout or "")[-2000:],
        "stderr": (completed.stderr or "")[-2000:],
        "command": command,
    }
