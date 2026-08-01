from pathlib import Path

import numpy as np
import pytest
import torch

from iccl.data.dataset import sequence_rng
from iccl.data.export import export_suite
from iccl.data.sequences import (
    FinalTaskConfig,
    PhaseConfig,
    SequenceConfig,
    SequenceSample,
    build_paired_control,
    build_paired_retention_control,
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
    for key in ("task_spans", "demo_counts", "base_mse", "num_curriculum_tasks"):
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


def _retention_nmse(final_block: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """A one-sequence retention-shaped suite: two curriculum tasks of 3 demos
    plus a 3-demo final block with the given per-demo errors."""
    nmse = np.array([[[1.0, 0.5, 0.25], [0.9, 0.9, 0.9], final_block]])
    return nmse, np.array([[3, 3, 3]])


def test_retention_metrics_against_position_matched_controls() -> None:
    nmse, counts = _retention_nmse([0.8, 0.4, 0.2])
    novel = _retention_nmse([1.0, 0.9, 0.8])
    shared = _retention_nmse([0.9, 0.6, 0.5])
    scalars, curves = retention_metrics(nmse, counts, {"novel": novel, "shared": shared})

    np.testing.assert_allclose(curves["relearning_curve"], [0.8, 0.4, 0.2])
    np.testing.assert_allclose(curves["control_curve"], [1.0, 0.9, 0.8])
    np.testing.assert_allclose(curves["savings_curve"], [0.2, 0.5, 0.6])
    np.testing.assert_allclose(curves["episodic_savings_curve"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(curves["module_savings_curve"], [0.1, 0.3, 0.3])
    # The split is an allocation of a fixed total, so the two terms must sum to it.
    np.testing.assert_allclose(
        curves["episodic_savings_curve"] + curves["module_savings_curve"],
        curves["savings_curve"],
    )

    assert scalars["savings_demo0"] == pytest.approx(0.2)
    assert scalars["savings_one_demo"] == pytest.approx(0.5)
    assert scalars["savings_mean"] == pytest.approx(13 / 30)
    assert scalars["episodic_savings_one_demo"] == pytest.approx(0.2)
    assert scalars["relearning_last_demo"] == pytest.approx(0.2)
    assert scalars["control_last_demo"] == pytest.approx(0.8)
    # Descriptive only: the first visit's own final demo, at task position 0.
    assert scalars["original_last_demo"] == pytest.approx(0.25)
    # The revisit crosses 0.5 at demo 1; the novel control never does, so it is
    # censored at the block length.
    assert scalars["demos_to_threshold_revisit"] == pytest.approx(1.0)
    assert scalars["demos_to_threshold_control"] == pytest.approx(3.0)
    assert scalars["demos_to_threshold_delta"] == pytest.approx(2.0)


def test_retention_metrics_without_shared_control_omits_the_split() -> None:
    nmse, counts = _retention_nmse([0.8, 0.4, 0.2])
    scalars, curves = retention_metrics(nmse, counts, {"novel": _retention_nmse([1.0, 0.9, 0.8])})
    assert "savings_curve" in curves
    assert "episodic_savings_curve" not in curves
    assert "module_savings_curve" not in curves
    assert not any(key.startswith(("episodic_", "module_")) for key in scalars)


def test_retention_metrics_require_the_novel_control() -> None:
    nmse, counts = _retention_nmse([0.8, 0.4, 0.2])
    with pytest.raises(KeyError, match="make_eval_sets"):
        retention_metrics(nmse, counts, {"shared": _retention_nmse([0.9, 0.6, 0.5])})


def test_evaluate_suites_end_to_end(family: HyperTeacher) -> None:
    torch.manual_seed(0)
    model = GDNModel(d_in=4, d_out=4, d_model=32, n_layers=2, n_heads=2, d_ffw=64)
    final = FinalTaskConfig(mode="composite", hotness=2, num_demos=3)
    in_dist, composite, control, retention = [], [], [], []
    novel, shared = [], []
    for i in range(2):
        in_dist.append(build_sequence(family, SEQUENCE, sequence_rng(0, i)))
        rng = sequence_rng(0, 10 + i)
        seq = build_sequence(family, SEQUENCE, rng, final_task=final, include_world=True)
        composite.append(seq)
        control.append(build_paired_control(family, SEQUENCE, seq, final, rng))
        rng = sequence_rng(0, 20 + i)
        revisit = build_sequence(family, SEQUENCE, rng, revisit_demos=3, include_world=True)
        retention.append(revisit)
        novel.append(build_paired_retention_control(family, revisit, rng, mode="novel"))
        shared.append(build_paired_retention_control(family, revisit, rng, mode="shared"))
    suites = {
        "in_dist": suite_from(in_dist),
        "composite": suite_from(composite),
        "composite_control": suite_from(control),
        "retention": suite_from(retention),
        "retention_control": suite_from(novel),
        "retention_control_shared": suite_from(shared),
    }

    scalars, curves = evaluate_suites(model, suites, torch.device("cpu"))

    # Pure-curriculum suites carry the generic in-context-learning curves over
    # their 8 curriculum tasks.
    assert curves["in_dist/learning_curve"].ndim == 1
    assert curves["in_dist/task_position_curve"].shape == (8,)
    assert np.isfinite(scalars["in_dist/nmse_last_demo"])

    # The composite suite reports the novel final task's few-shot curve with
    # history, its control the no-history baseline, and their difference as the
    # benefit of history — all over the final task's demos, not the curriculum.
    assert curves["composite/learning_curve"].shape == (final.num_demos,)
    assert curves["composite_control/learning_curve"].shape == (final.num_demos,)
    assert curves["composite/benefit_curve"].shape == (final.num_demos,)
    assert np.isfinite(scalars["composite/nmse_last_demo"])
    assert np.isfinite(scalars["composite/benefit_last_demo"])

    # Special-task suites do not emit the redundant curriculum curves.
    assert "composite/task_position_curve" not in curves
    assert "retention/learning_curve" not in curves
    assert "retention/task_position_curve" not in curves
    assert "retention_control/learning_curve" not in curves
    assert "retention_control_shared/task_position_curve" not in curves

    # Retention reports the revisit against its position-matched controls; a
    # comparison against its own first visit is not part of the metric surface.
    assert curves["retention/savings_curve"].shape == (3,)
    assert curves["retention/module_savings_curve"].shape == (3,)
    assert np.isfinite(scalars["retention/savings_one_demo"])
    assert np.isfinite(scalars["retention/episodic_savings_mean"])
    assert not any(
        key in scalars
        for key in ("retention/forgetting", "retention/revisit_first_demo", "savings_first_demo")
    )


def test_load_eval_suites_missing_dir_points_at_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        load_eval_suites(tmp_path)


def test_load_eval_suites_rejects_a_retention_set_without_its_control(
    tmp_path: Path, family: HyperTeacher
) -> None:
    retention = [build_sequence(family, SEQUENCE, sequence_rng(0, 0), revisit_demos=3)]
    export_suite(retention, tmp_path / "retention", {"suite": "retention"})
    with pytest.raises(FileNotFoundError, match="retention_control"):
        load_eval_suites(tmp_path)
