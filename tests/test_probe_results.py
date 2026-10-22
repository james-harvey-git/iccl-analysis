import numpy as np
import pytest

from iccl.analysis.probe_results import (
    aggregate_exposure,
    aggregate_scores,
    exposure_scores,
    paired_control_scores,
    score_predictions,
)


def condition(world, control="trained"):
    return {
        "phase": "development",
        "population": "test_familiar",
        "position": 5,
        "capture": "final_answer",
        "family": "state",
        "selection": "all",
        "control": control,
        "train_count": 32,
        "world": world,
    }


def test_scores_constant_coordinates_and_stable_top_pair():
    targets = np.array([[0.5, 0.7, 0], [0.8, 0.5, 0]])
    scores = score_predictions(targets, targets, np.ones(3))
    assert scores["r2_macro"] is None
    assert scores["undefined_coordinates"] == [2]
    assert scores["active_pair_accuracy"] == 1
    tied = score_predictions(targets, np.zeros_like(targets), np.ones(3))
    assert tied["active_pair_accuracy"] == 1
    assert scores["fixed_scale_vector_mse"] == 0


def test_equal_world_average_and_paired_differences():
    rows = []
    for world, count in ((0, 10), (1, 10000)):
        for control in ("trained", "untrained"):
            value = world + (0.2 if control == "trained" else 0)
            rows.append(
                condition(world, control)
                | {
                    "scores": {
                        "r2_macro": value,
                        "active_pair_accuracy": value,
                        "active_coefficient_rmse": value,
                        "fixed_scale_vector_mse": value,
                        "trials": count,
                    }
                }
            )
    settings = {"seed": 0, "replicates": 100, "confidence": 0.95}
    aggregate = aggregate_scores(rows, worlds=2, **settings)
    primary = next(r for r in aggregate if r["control"] == "trained" and r["metric"] == "r2_macro")
    assert primary["mean"] == pytest.approx(0.7)
    paired = paired_control_scores(rows, **settings)
    assert all(r["mean"] == pytest.approx(0.2) for r in paired)
    assert all(r["lower"] == pytest.approx(0.2) for r in paired)


def test_exposure_bins_keep_original_training_scale_and_common_world_cohort():
    z = np.array([[0.5, 0.7, 0], [0, 0.6, 0.8], [0.7, 0, 0.5], [0.8, 0, 0.7]])
    scores = exposure_scores(
        z,
        z + 0.1,
        np.array([[1, 0], [0, 1], [1, 2], [2, 1]]),
        np.array([0.1, 0.2, 0.3]),
        [0, 1, 2, 3],
    )
    assert {(r["e_min"], r["e_max"]) for r in scores} == {(0, 1), (1, 2)}
    assert scores[0]["scores"]["fixed_scale_vector_mse"] == pytest.approx(
        0.01 * np.mean(1 / np.array([0.1, 0.2, 0.3]))
    )
    rows = [
        condition(w) | cell
        for w in range(3)
        for cell in scores
        if not (w == 2 and cell["e_min"] == 1)
    ]
    aggregated = aggregate_exposure(
        rows, worlds=3, seed=0, replicates=50, confidence=0.95, min_trials=2, min_worlds=2
    )
    valid = [r for r in aggregated if r["metric"] == "fixed_scale_vector_mse"]
    assert all(r["world_ids"] == [0, 1] for r in valid)
