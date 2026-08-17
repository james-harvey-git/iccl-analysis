from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from iccl.data.controls import (
    build_paired_composition_controls,
    build_paired_retention_control,
)
from iccl.data.curriculum import PhaseConfig, SequenceConfig
from iccl.data.dataset import sequence_rng
from iccl.data.sequences import (
    SequenceSample,
    build_sequence,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig
from iccl.models.model import GDNModel
from iccl.training.metrics import (
    _append_generalization_gaps,
    demo_mse,
    demo_nmse,
    evaluate_suites,
    load_eval_suites,
)

TEACHER = TeacherConfig(
    input_dim=4,
    output_dim=4,
    hidden_dims=(4,),
    use_bias=True,
    num_modules=8,
    scale=3.0,
    weighting="discrete",
)

SEQUENCE = SequenceConfig(
    phases=(PhaseConfig(num_tasks=8, hotness=(2, 2)),),
    demos_per_task=4,
    signal_boundaries=True,
    require_identifiable=True,
)


@pytest.fixture(scope="module")
def family() -> HyperTeacher:
    return HyperTeacher(TEACHER, max_hotness=3)


def suite_from(samples: list[SequenceSample]) -> dict[str, Any]:
    suite = {
        key: np.stack([getattr(s, key) for s in samples])
        for key in ("tokens", "token_type", "targets", "loss_mask")
    }
    for key in ("task_spans", "demo_counts", "base_mse", "num_curriculum_tasks"):
        suite[key] = np.stack([s.info[key] for s in samples])
    return suite


def capability_metadata(name: str, demo_counts: tuple[int, ...]) -> dict[str, object]:
    capability, condition, slice_name, status, modules, tasks, _ = name.split("__")[:7]
    return {
        "suite": name,
        "capability": capability,
        "condition": condition,
        "structural_slice": slice_name,
        "variant": "",
        "module_count_status": status,
        "num_modules": int(modules[1:]),
        "num_tasks": int(tasks[1:]),
        "num_surplus_tasks": int(tasks[1:]) - (int(modules[1:]) - 1),
        "demo_counts": demo_counts,
        "history_prediction_tokens": sum(demo_counts),
        "history_serialized_tokens": 2 * sum(demo_counts) + len(demo_counts),
        "pair_group": f"{capability}__{slice_name}",
        "config": {"sequence": {"curriculum_sampler": "constructive"}, "weighting": "discrete"},
    }


def test_demo_nmse_hand_built() -> None:
    # One sequence, two tasks x two demos with boundaries: layout is
    # [B x y x y B x y x y], x-positions 1, 3, 6, 8.
    d_out = 2
    targets = np.zeros((1, 10, d_out), dtype=np.float32)
    preds = np.zeros_like(targets)
    preds[0, 1] = [1.0, -1.0]  # sq err mean 1
    preds[0, 3] = [2.0, 0.0]  # sq err mean 2
    preds[0, 6] = [0.0, 0.0]  # sq err mean 0
    preds[0, 8] = [2.0, 2.0]  # sq err mean 4
    suite = {
        "targets": targets,
        "task_spans": np.array([[[1, 5], [6, 10]]]),
        "demo_counts": np.array([[2, 2]]),
        "base_mse": np.array([[[2.0, 2.0], [1.0, 3.0]]]),  # per-task means 2, 2
    }
    mse = demo_mse(preds, suite)
    nmse = demo_nmse(preds, suite)
    np.testing.assert_allclose(mse, [[[1.0, 2.0], [0.0, 4.0]]])
    np.testing.assert_allclose(nmse, [[[0.5, 1.0], [0.0, 2.0]]])


def test_demo_errors_nan_pad_variable_demo_counts() -> None:
    targets = np.zeros((2, 8, 2), dtype=np.float32)
    preds = np.zeros_like(targets)
    counts = np.array([[2, 1], [1, 2]])
    spans = np.array([[[1, 5], [6, 8]], [[1, 3], [4, 8]]])
    for sequence, position, error in (
        (0, 1, 1.0),
        (0, 3, 2.0),
        (0, 6, 4.0),
        (1, 1, 9.0),
        (1, 4, 16.0),
        (1, 6, 25.0),
    ):
        preds[sequence, position] = np.sqrt(error)
    suite = {
        "targets": targets,
        "task_spans": spans,
        "demo_counts": counts,
        "base_mse": np.full((2, 2, 2), 2.0),
    }

    mse = demo_mse(preds, suite)
    expected = np.array([[[1.0, 2.0], [4.0, np.nan]], [[9.0, np.nan], [16.0, 25.0]]])
    np.testing.assert_allclose(mse, expected, equal_nan=True)
    np.testing.assert_allclose(demo_nmse(preds, suite), expected / 2.0, equal_nan=True)


def test_predicting_the_mean_scores_unit_nmse(family: HyperTeacher) -> None:
    sample = build_sequence(family, SEQUENCE, sequence_rng(0, 0))
    suite = suite_from([sample])
    preds = np.zeros((1, *sample.targets.shape), dtype=np.float32)
    for k, (start, _) in enumerate(sample.info["task_spans"]):
        positions = start + 2 * np.arange(sample.info["demo_counts"][k])
        preds[0, positions] = sample.targets[positions].mean(axis=0)
    nmse = demo_nmse(preds, suite)
    per_task = np.nanmean(nmse, axis=2)
    np.testing.assert_allclose(per_task, np.ones_like(per_task), rtol=1e-5)


def test_demo_positions_are_x_tokens(family: HyperTeacher) -> None:
    sample = build_sequence(family, SEQUENCE, sequence_rng(0, 1))
    for k, (start, _) in enumerate(sample.info["task_spans"]):
        positions = start + 2 * np.arange(sample.info["demo_counts"][k])
        assert (sample.loss_mask[positions] == 1.0).all()
    assert sample.loss_mask.sum() == sample.info["demo_counts"].sum()


def test_load_eval_suites_missing_dir_points_at_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        load_eval_suites(tmp_path)


def test_variable_capability_metrics_emit_structural_rows() -> None:
    family = HyperTeacher(
        TeacherConfig(
            input_dim=4,
            output_dim=4,
            hidden_dims=(4,),
            use_bias=True,
            num_modules=4,
            scale=3.0,
            weighting="discrete",
        ),
        max_hotness=2,
    )
    cfg = SequenceConfig(
        phases=(),
        demos_per_task=2,
        signal_boundaries=True,
        require_identifiable=True,
        curriculum_sampler="constructive",
        hotness=2,
        surplus_tasks=1,
    )
    names = {
        condition: f"{capability}__{condition}__fixed_surplus__seen__m04__t04__b0008"
        for capability, conditions in {
            "icl": ["ordinary"],
            "composition": ["constituent", "matched_prefix", "no_history"],
            "retention": ["repeat", "novel", "shared"],
        }.items()
        for condition in conditions
    }
    samples: dict[str, list[SequenceSample]] = {name: [] for name in names.values()}
    for i in range(2):
        samples[names["ordinary"]].append(
            build_sequence(family, cfg, sequence_rng(4, i), fixed_demo_counts=(2,) * 4)
        )
        triplet = build_paired_composition_controls(
            family,
            cfg,
            sequence_rng(4, 10 + i),
            target_demos=3,
            constituent_task_exposures=1,
            fixed_demo_counts=(2,) * 4,
        )
        for condition, sample in zip(
            ("constituent", "matched_prefix", "no_history"), triplet, strict=True
        ):
            samples[names[condition]].append(sample)

        rng = sequence_rng(4, 20 + i)
        repeat = build_sequence(
            family,
            cfg,
            rng,
            revisit_demos=3,
            include_world=True,
            fixed_demo_counts=(2,) * 4,
        )
        samples[names["repeat"]].append(repeat)
        samples[names["novel"]].append(
            build_paired_retention_control(family, repeat, rng, mode="novel")
        )
        samples[names["shared"]].append(
            build_paired_retention_control(family, repeat, rng, mode="shared")
        )

    suites = {name: suite_from(group) for name, group in samples.items()}
    for name, suite in suites.items():
        suite["__meta__"] = capability_metadata(name, (2,) * 4)
    torch.manual_seed(0)
    model = GDNModel(d_in=4, d_out=4, d_model=32, n_layers=2, n_heads=2, d_ffw=64)
    report = evaluate_suites(
        model,
        suites,
        torch.device("cpu"),
        bootstrap_seed=3,
        bootstrap_replicates=50,
    )

    composition = "composition/fixed_surplus__seen__m04__t04__b0008"
    retention = "retention/fixed_surplus__seen__m04__t04__b0008"
    assert np.isfinite(report.scalars[f"{composition}/benefit_mean"])
    assert np.isfinite(report.scalars[f"{retention}/savings_mean"])
    assert np.isfinite(report.scalars[f"{retention}/episodic_savings_mean"])
    assert f"{composition}/benefit_curve" in report.curves
    assert f"{retention}/savings_curve" in report.curves
    assert {row["capability"] for row in report.summary_rows} == {
        "icl",
        "composition",
        "retention",
    }
    assert all(row["slice"] == "fixed_surplus" for row in report.summary_rows)
    assert all(row["ci_low"] <= row["ci_high"] for row in report.summary_rows)
    savings = next(row["value"] for row in report.summary_rows if row["metric"] == "savings_mean")
    episodic = next(
        row["value"] for row in report.summary_rows if row["metric"] == "episodic_savings_mean"
    )
    module = next(
        row["value"] for row in report.summary_rows if row["metric"] == "module_savings_mean"
    )
    assert savings == pytest.approx(episodic + module)


def test_generalization_gap_definitions_use_matched_seen_cells() -> None:
    rows = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "metric": "nmse_aulc",
            "slice": "matched_task_count",
            "status": status,
            "M": modules,
            "value": value,
        }
        for status, modules, value in (
            ("seen", 4, 1.0),
            ("heldout", 6, 2.5),
            ("seen", 8, 3.0),
            ("ood", 10, 4.0),
        )
    ]
    scalars: dict[str, float] = {}
    _append_generalization_gaps(rows, scalars)
    interpolation = next(row for row in rows if row["condition"] == "interpolation_gap")
    ood = next(row for row in rows if row["condition"] == "ood_gap")
    assert interpolation["value"] == pytest.approx(0.5)
    assert ood["value"] == pytest.approx(1.0)
    assert any("interpolation_gap" in key for key in scalars)
    assert any("ood_gap" in key for key in scalars)
