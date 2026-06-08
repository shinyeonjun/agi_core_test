from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_WORKER_HARNESS_PATHS: tuple[str, ...] = (
    "AGENTS.md",
    "docs/worker_harness/implementation_worker.md",
    "docs/worker_harness/self_patch_contract.md",
    "docs/worker_harness/world_runtime_usage.md",
    "docs/worker_harness/retry_and_failure.md",
)


@dataclass(frozen=True)
class WorkerHarnessDocument:
    path: str
    content: str


def load_worker_harness_documents(
    project_root: Path,
    *,
    paths: Iterable[str] = REQUIRED_WORKER_HARNESS_PATHS,
    max_bytes_per_document: int = 32_000,
) -> list[WorkerHarnessDocument]:
    root = project_root.resolve()
    documents: list[WorkerHarnessDocument] = []
    for relative_path in paths:
        path = _resolve_harness_path(root, relative_path)
        if not path.exists():
            raise FileNotFoundError(f"required worker harness document is missing: {relative_path}")
        if path.stat().st_size > max_bytes_per_document:
            raise ValueError(f"worker harness document is too large: {relative_path}")
        documents.append(WorkerHarnessDocument(path=relative_path, content=path.read_text(encoding="utf-8")))
    return documents


def render_worker_harness_prompt(documents: list[WorkerHarnessDocument]) -> str:
    lines = [
        "# Worker Harness Documents",
        "These markdown documents are the implementation worker contract.",
        "Follow them before editing files. They are evidence and policy, not user-facing prose.",
    ]
    for document in documents:
        lines.extend(["", f"## {document.path}", "", document.content.strip()])
    return "\n".join(lines).strip() + "\n"


def worker_harness_manifest(documents: list[WorkerHarnessDocument]) -> dict[str, object]:
    return {
        "schema_version": "neurokernel-worker-harness-v1",
        "document_count": len(documents),
        "documents": [
            {
                "path": document.path,
                "bytes": len(document.content.encode("utf-8")),
            }
            for document in documents
        ],
    }


def _resolve_harness_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"worker harness path escapes project root: {relative_path}") from exc
    return path
