from pathlib import Path

import numpy as np
import pytest
import torch

from iccl.data.dataset import sequence_rng
from iccl.data.sequences import (
    FinalTaskConfig,
    PhaseConfig,
    SequenceConfig,
    SequenceSample,
    build_paired_control,
    build_sequence,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig
from iccl.models.model import GDNModel
from iccl.training.metrics import (
    demo_nmse,
    evaluate_suites,
    load_eval_suites,
    retention_metrics,
    suite_scalars,
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


def suite_from(samples: list[SequenceSample]) -> dict[str, np.ndarray]:
    suite = {
        key: np.stack([getattr(s, key) for s in samples])
        for key in ("tokens", "token_type", "targets", "loss_mask")
    }
    for key in ("task_spans", "demo_counts", "base_mse"):
        suite[key] = np.stack([s.info[key] for s in samples])
    return suite


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
    nmse = demo_nmse(preds, suite)
    np.testing.assert_allclose(nmse, [[[0.5, 1.0], [0.0, 2.0]]])
    scalars = suite_scalars(nmse, suite["demo_counts"])
    assert scalars["nmse_first_demo"] == pytest.approx(0.25)
    assert scalars["nmse_last_demo"] == pytest.approx(1.5)
    assert scalars["nmse_mean"] == pytest.approx(0.875)


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


def test_retention_metrics_pairing() -> None:
    # Two tasks of 3 demos plus a 2-demo revisit; revisit errors are half the
    # original's at each paired demo index, base variance 1 everywhere.
    nmse = np.array([[[1.0, 0.5, 0.25], [0.9, 0.9, 0.9], [0.5, 0.25, np.nan]]])
    counts = np.array([[3, 3, 2]])
    scalars, curves = retention_metrics(nmse, counts)
    assert scalars["revisit_first_demo"] == pytest.approx(0.5)
    assert scalars["original_last_demo"] == pytest.approx(0.25)
    assert scalars["forgetting"] == pytest.approx(0.25)
    assert scalars["savings_first_demo"] == pytest.approx(0.5)
    np.testing.assert_allclose(curves["original_curve"], [1.0, 0.5, 0.25])
    np.testing.assert_allclose(curves["relearning_curve"], [0.5, 0.25])
    np.testing.assert_allclose(curves["savings_curve"], [0.5, 0.25])


def test_evaluate_suites_end_to_end(family: HyperTeacher) -> None:
    torch.manual_seed(0)
    model = GDNModel(d_in=4, d_out=4, d_model=32, n_layers=2, n_heads=2, d_ffw=64)
    final = FinalTaskConfig(mode="composite", hotness=2, num_demos=3)
    in_dist, composite, control, retention = [], [], [], []
    for i in range(2):
        in_dist.append(build_sequence(family, SEQUENCE, sequence_rng(0, i)))
        rng = sequence_rng(0, 10 + i)
        seq = build_sequence(family, SEQUENCE, rng, final_task=final, include_world=True)
        composite.append(seq)
        control.append(build_paired_control(family, SEQUENCE, seq, final, rng))
        retention.append(build_sequence(family, SEQUENCE, sequence_rng(0, 20 + i), revisit_demos=3))
    suites = {
        "in_dist": suite_from(in_dist),
        "composite": suite_from(composite),
        "composite_control": suite_from(control),
        "retention": suite_from(retention),
    }

    scalars, curves = evaluate_suites(model, suites, torch.device("cpu"))

    for name in suites:
        assert np.isfinite(scalars[f"{name}/nmse_first_demo"])
        assert np.isfinite(scalars[f"{name}/nmse_last_demo"])
        assert curves[f"{name}/learning_curve"].ndim == 1
        assert curves[f"{name}/task_position_curve"].shape == (
            suites[name]["demo_counts"].shape[1],
        )
    assert np.isfinite(scalars["composite/benefit_last_demo"])
    assert curves["composite/benefit_curve"].shape == (final.num_demos,)
    assert np.isfinite(scalars["retention/forgetting"])
    assert curves["retention/savings_curve"].shape == (3,)


def test_load_eval_suites_missing_dir_points_at_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        load_eval_suites(tmp_path)
