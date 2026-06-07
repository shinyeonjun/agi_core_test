import argparse
import json
from pathlib import Path

from neurokernel_seed import nk_cli


def _write_ablation(run_dir: Path, *, hybrid_veto: float, prior: float, signal_count: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_ablation_result.json").write_text(
        json.dumps(
            {
                "aggregate": {
                    "macro_success_rate": {
                        "model_only": 0.5,
                        "prior_only": prior,
                        "hybrid": max(prior, hybrid_veto - 0.1),
                        "hybrid_veto": hybrid_veto,
                    },
                    "model_needed_signal_group_count": signal_count,
                    "hybrid_veto_gain_over_prior": hybrid_veto - prior,
                    "hybrid_gain_over_prior": 0.0,
                    "groups_with_model_needed_signal": ["model_needed_probe"] if signal_count else [],
                },
                "interpretation": {"verdict": "hybrid_adds_value"},
            }
        ),
        encoding="utf-8",
    )


def test_nk_top_ranks_training_runs_by_benchmark_quality(tmp_path):
    _write_ablation(tmp_path / "mid", hybrid_veto=0.9, prior=0.7, signal_count=1)
    _write_ablation(tmp_path / "best", hybrid_veto=1.0, prior=0.6, signal_count=3)
    _write_ablation(tmp_path / "weak", hybrid_veto=0.8, prior=0.8, signal_count=0)

    result = nk_cli._run_top_models(argparse.Namespace(run_dir=str(tmp_path), limit=2))

    assert [item["run_name"] for item in result["top"]] == ["best", "mid"]
    assert result["top"][0]["hybrid_veto_gain_over_prior"] == 0.4


def test_nk_help_exposes_menu_commands():
    parser = nk_cli._build_parser()
    help_text = parser.format_help()
    assert "train-use" in help_text
    assert "top" in help_text
