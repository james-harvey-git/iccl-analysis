from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from iccl.data.diagnostics import analyze_sequence_distribution


def make_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 5,
            "data": {
                "name": "hyperteacher",
                "input_dim": 4,
                "output_dim": 4,
                "hidden_dims": [4],
                "use_bias": True,
                "num_modules": {"min": 4, "max": 7, "held_out": [6]},
                "scale": 3.0,
                "weighting": "discrete",
                "sequence": {
                    "curriculum_sampler": "constructive",
                    "hotness": 2,
                    "surplus_tasks": 2,
                    "demos_per_task": {"min": 2, "max": 4, "scope": "per_task"},
                    "signal_boundaries": True,
                    "require_identifiable": True,
                    "require_full_rank": False,
                },
                "distribution_diagnostic": {
                    "num_sequences": 60,
                    "bootstrap_seed": 11,
                    "bootstrap_replicates": 40,
                },
            },
            "training": {"batch_size": 8},
        }
    )


def test_distribution_diagnostic_reports_unbiased_surplus_categories(tmp_path: Path) -> None:
    report = analyze_sequence_distribution(make_cfg(), tmp_path)
    assert report["generated_sequences"] == 60
    assert report["failure_count"] == 0
    assert report["held_out_module_occurrences"] == 0
    assert "6" not in report["distributions"]["M"]["counts"]
    assert report["invariant_failures"] == {
        "coverage": 0,
        "connectedness": 0,
        "full_rank": 0,
    }
    assert report["batch_padding"]["expansion"] >= 1.0
    assert 0.0 <= report["batch_padding"]["padding_fraction"] < 1.0

    for relative in ("generation_relative", "presentation_relative"):
        categories = report["categories"][relative]
        for estimator in (
            "sequence_uniform",
            "surplus_task_uniform",
            "loss_token_weighted",
        ):
            total = sum(
                categories[category][estimator]["estimate"]
                for category in (
                    "novel_support",
                    "seen_support_new_weights",
                    "exact_repeat",
                )
            )
            assert total == pytest.approx(1.0)

    assert (tmp_path / "sequence_distribution.json").exists()
    assert (tmp_path / "sequence_distribution.csv").exists()


def test_rejection_diagnostic_classifies_the_declared_surplus_tasks(tmp_path: Path) -> None:
    cfg = make_cfg()
    cfg.data.sequence.curriculum_sampler = "rejection"
    cfg.data.distribution_diagnostic.num_sequences = 20
    report = analyze_sequence_distribution(cfg, tmp_path)
    estimates = report["categories"]["generation_relative"]
    total = sum(
        estimates[category]["surplus_task_uniform"]["estimate"]
        for category in (
            "novel_support",
            "seen_support_new_weights",
            "exact_repeat",
        )
    )
    assert total == pytest.approx(1.0)
    assert all(row["attempts"] >= row["num_sequences"] for row in report["rejection_efficiency"])


def test_zero_surplus_reports_the_conditional_probability_as_undefined(tmp_path: Path) -> None:
    cfg = make_cfg()
    cfg.data.sequence.surplus_tasks = 0
    cfg.data.distribution_diagnostic.num_sequences = 8
    report = analyze_sequence_distribution(cfg, tmp_path)
    generation = report["categories"]["generation_relative"]
    assert generation["p_surplus_zero"] == 1.0
    assert generation["novel_support"]["sequence_uniform"]["estimate"] is None
