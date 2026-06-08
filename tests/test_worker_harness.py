from pathlib import Path

import pytest

from neurokernel_seed.harness.worker_harness import (
    load_worker_harness_documents,
    render_worker_harness_prompt,
    worker_harness_manifest,
)


def test_load_worker_harness_documents_reads_required_contracts(tmp_path):
    _write_worker_harness_docs(tmp_path)

    documents = load_worker_harness_documents(tmp_path)
    prompt = render_worker_harness_prompt(documents)
    manifest = worker_harness_manifest(documents)

    assert len(documents) == 5
    assert "## AGENTS.md" in prompt
    assert "## docs/worker_harness/self_patch_contract.md" in prompt
    assert manifest["schema_version"] == "neurokernel-worker-harness-v1"
    assert manifest["document_count"] == 5


def test_load_worker_harness_documents_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes project root"):
        load_worker_harness_documents(tmp_path, paths=["../AGENTS.md"])


def test_load_worker_harness_documents_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="required worker harness document is missing"):
        load_worker_harness_documents(tmp_path)


def _write_worker_harness_docs(project: Path) -> None:
    (project / "AGENTS.md").write_text("# Test Agent Harness\n", encoding="utf-8")
    docs = project / "docs" / "worker_harness"
    docs.mkdir(parents=True)
    for name in (
        "implementation_worker.md",
        "self_patch_contract.md",
        "world_runtime_usage.md",
        "retry_and_failure.md",
    ):
        (docs / name).write_text(f"# {name}\n", encoding="utf-8")
