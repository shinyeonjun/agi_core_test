from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_IGNORED_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "baselines",
    "data",
    "dist",
    "htmlcov",
    "language_workspace",
    "logs",
    "model_transrate",
    "node_modules",
    "venv",
    ".venv",
}


@dataclass(frozen=True)
class CodeStructureThresholds:
    long_file_lines: int = 300
    very_long_file_lines: int = 700
    large_function_lines: int = 80
    many_functions: int = 18
    many_classes: int = 8


def inspect_code_structure(
    project_root: str | Path,
    *,
    paths: list[str] | None = None,
    max_files: int = 250,
    max_results: int = 20,
    thresholds: CodeStructureThresholds | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"project_root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"project_root is not a directory: {root}")

    threshold = thresholds or CodeStructureThresholds()
    candidates: list[dict[str, Any]] = []
    scanned_files = 0
    parse_errors: list[dict[str, str]] = []
    for file_path in _iter_python_files(root, paths=paths, max_files=max_files):
        scanned_files += 1
        try:
            report = _inspect_python_file(root, file_path, threshold)
        except SyntaxError as exc:
            parse_errors.append({"path": _relative_path(root, file_path), "error": str(exc)})
            continue
        if report["reasons"]:
            candidates.append(report)

    candidates.sort(key=lambda item: (item["score"], item["line_count"]), reverse=True)
    limited_candidates = candidates[: max(1, max_results)]
    return {
        "schema_version": "neurokernel-code-structure-report-v1",
        "project_root": str(root),
        "scanned_files": scanned_files,
        "candidate_count": len(candidates),
        "candidates": limited_candidates,
        "parse_errors": parse_errors[:20],
        "thresholds": {
            "long_file_lines": threshold.long_file_lines,
            "very_long_file_lines": threshold.very_long_file_lines,
            "large_function_lines": threshold.large_function_lines,
            "many_functions": threshold.many_functions,
            "many_classes": threshold.many_classes,
        },
        "summary": _summary(limited_candidates, scanned_files=scanned_files, total_candidates=len(candidates)),
    }


def _iter_python_files(root: Path, *, paths: list[str] | None, max_files: int) -> list[Path]:
    selected_roots = [_resolve_child(root, item) for item in paths] if paths else [root / "src", root / "tests"]
    if not any(path.exists() for path in selected_roots):
        selected_roots = [root]
    files: list[Path] = []
    for start in selected_roots:
        if not start.exists():
            continue
        if start.is_file() and start.suffix == ".py":
            files.append(start)
            continue
        if start.is_dir():
            for file_path in start.rglob("*.py"):
                if _is_ignored(root, file_path):
                    continue
                files.append(file_path)
                if len(files) >= max_files:
                    return sorted(files)
    return sorted(files[:max_files])


def _resolve_child(root: Path, value: str) -> Path:
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {value}") from exc
    return candidate


def _is_ignored(root: Path, path: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in DEFAULT_IGNORED_NAMES for part in relative_parts)


def _inspect_python_file(root: Path, file_path: Path, thresholds: CodeStructureThresholds) -> dict[str, Any]:
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    tree = ast.parse(text, filename=str(file_path))
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    max_function_lines = max((_node_line_count(node) for node in functions), default=0)
    max_class_lines = max((_node_line_count(node) for node in classes), default=0)
    reasons: list[str] = []
    score = 0
    if len(lines) >= thresholds.very_long_file_lines:
        reasons.append("very_long_file")
        score += 50
    elif len(lines) >= thresholds.long_file_lines:
        reasons.append("long_file")
        score += 25
    if max_function_lines >= thresholds.large_function_lines:
        reasons.append("large_function")
        score += 20
    if len(functions) >= thresholds.many_functions:
        reasons.append("many_functions")
        score += 15
    if len(classes) >= thresholds.many_classes:
        reasons.append("many_classes")
        score += 15
    return {
        "path": _relative_path(root, file_path),
        "line_count": len(lines),
        "class_count": len(classes),
        "function_count": len(functions),
        "import_count": len(imports),
        "max_function_lines": max_function_lines,
        "max_class_lines": max_class_lines,
        "reasons": reasons,
        "score": score,
        "top_level_symbols": _top_level_symbols(tree),
        "suggested_next_step": _suggest_next_step(reasons),
    }


def _node_line_count(node: ast.AST) -> int:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return 0
    return max(0, end - start + 1)


def _top_level_symbols(tree: ast.Module, *, limit: int = 12) -> list[str]:
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        if len(symbols) >= limit:
            break
    return symbols


def _suggest_next_step(reasons: list[str]) -> str:
    if "large_function" in reasons:
        return "extract_small_helpers_without_changing_behavior"
    if "many_functions" in reasons or "many_classes" in reasons:
        return "split_by_responsibility_and_keep_public_imports_stable"
    if "very_long_file" in reasons or "long_file" in reasons:
        return "map_responsibilities_before_splitting_file"
    return "no_refactor_candidate"


def _summary(candidates: list[dict[str, Any]], *, scanned_files: int, total_candidates: int) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for item in candidates:
        for reason in item["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "scanned_files": scanned_files,
        "reported_candidates": len(candidates),
        "total_candidates": total_candidates,
        "top_candidate": candidates[0]["path"] if candidates else None,
        "reason_counts": reason_counts,
    }


def _relative_path(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()
