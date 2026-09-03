from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iccl.evaluation.metrics import (
    _aggregate,
    _evaluate,
    demo_mse,
    demo_nmse,
    load_eval_suites,
    token_mse,
)


def test_demo_nmse_hand_built() -> None:
    targets = np.zeros((1, 10, 2), dtype=np.float32)
    preds = np.zeros_like(targets)
    preds[0, 1] = [1.0, -1.0]
    preds[0, 3] = [2.0, 0.0]
    preds[0, 6] = [0.0, 0.0]
    preds[0, 8] = [2.0, 2.0]
    suite = {
        "targets": targets,
        "task_spans": np.array([[[1, 5], [6, 10]]]),
        "demo_counts": np.array([[2, 2]]),
        "base_mse": np.array([[[2.0, 2.0], [1.0, 3.0]]]),
    }
    np.testing.assert_allclose(demo_mse(preds, suite), [[[1.0, 2.0], [0.0, 4.0]]])
    np.testing.assert_allclose(demo_nmse(preds, suite), [[[0.5, 1.0], [0.0, 2.0]]])


def test_token_mse_pools_prediction_tokens_globally() -> None:
    targets = np.zeros((2, 3, 1), dtype=np.float32)
    preds = np.array([[[1.0], [100.0], [100.0]], [[3.0], [3.0], [3.0]]])
    suite = {"targets": targets, "loss_mask": np.array([[1, 0, 0], [1, 1, 1]])}
    assert token_mse(preds, suite) == pytest.approx(7.0)


def test_stratified_aggregate_weights_positions_equally_and_is_deterministic() -> None:
    values = np.array([0.0, 0.0, 0.0, 10.0])
    strata = np.array([0, 0, 0, 1])
    first = _aggregate(values, seed=7, replicates=50, strata=strata)
    second = _aggregate(values, seed=7, replicates=50, strata=strata)
    assert float(first[0]) == pytest.approx(5.0)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def metadata(capability: str, condition: str, pair_group: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "capability": capability,
        "condition": condition,
        "cell_id": "m04__t04__d002",
        "family_memberships": ["canonical", "task_variation"],
        "module_count_status": "seen",
        "num_modules": 4,
        "num_tasks": 4,
        "num_surplus_tasks": 1,
        "demos_per_task": 2,
        "config": {
            "sequence": {"curriculum_sampler": "constructive"},
            "weighting": "discrete",
        },
    }
    if pair_group is not None:
        value["pair_group"] = pair_group
    return value


def capability_fixture() -> tuple[
    dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    count, tasks, demos = 5, 4, 2
    suites: dict[str, dict[str, Any]] = {
        "icl": {"__meta__": metadata("icl", "ordinary")},
    }
    mses = {"icl": np.arange(count * tasks * demos).reshape(count, tasks, demos) / 10}
    nmses = {"icl": mses["icl"].copy()}

    for condition in ("constituent", "matched_prefix", "no_history"):
        name = f"composition_{condition}"
        suites[name] = {"__meta__": metadata("composition", condition, "composition-pair")}
        value = {"constituent": 2.0, "matched_prefix": 5.0, "no_history": 7.0}[condition]
        mses[name] = np.full((count, tasks + 1, demos), value)
        nmses[name] = mses[name].copy()

    positions = np.array([0, 0, 1, 2, 3])
    delays = tasks - 1 - positions
    for condition in ("repeat", "shared", "novel"):
        name = f"retention_{condition}"
        suites[name] = {
            "__meta__": metadata("retention", condition, "retention-pair"),
            "original_task_position": positions,
            "intervening_tasks": delays,
        }
        values = np.zeros((count, tasks + 1, demos), dtype=np.float64)
        for task in range(tasks):
            values[:, task] = 10 + task
        values[:, -1] = 1.0
        if condition == "shared":
            values[:, -1] += delays[:, None]
        elif condition == "novel":
            values[:, -1] += 3 * delays[:, None]
        mses[name] = values
        nmses[name] = values.copy()
    return suites, mses, nmses


def test_requested_capability_metrics_match_hand_computations() -> None:
    suites, mses, nmses = capability_fixture()
    raw = {
        f"{name}/{kind}": errors
        for name in suites
        for kind, errors in (("mse", mses[name]), ("nmse", nmses[name]))
    }
    report = _evaluate(
        suites,
        mses,
        nmses,
        raw,
        bootstrap_seed=11,
        bootstrap_replicates=30,
    )

    metrics = {row["metric"]: row["value"] for row in report.summary_rows}
    assert "episode_nmse_mean" not in metrics
    assert metrics["benefit_mean"] == pytest.approx(3.0)
    assert metrics["savings_mean"] == pytest.approx(4.5)
    assert metrics["episodic_savings_mean"] == pytest.approx(1.5)
    assert metrics["module_savings_mean"] == pytest.approx(3.0)
    assert metrics["savings_mean"] == pytest.approx(
        metrics["episodic_savings_mean"] + metrics["module_savings_mean"]
    )

    episode = [row for row in report.curve_rows if row["curve_type"] == "episode_learning"]
    assert [row["x_value"] for row in episode] == [0, 1, 2, 3]
    original = [
        row
        for row in report.curve_rows
        if row["curve_type"] == "retention_learning" and row["condition"] == "original"
    ]
    assert all(row["nmse"] == pytest.approx(11.5) for row in original)

    total_delay = [
        row
        for row in report.curve_rows
        if row["curve_type"] == "retention_delay" and row["retention_component"] == "total"
    ]
    assert [row["x_value"] for row in total_delay] == [0, 1, 2, 3]
    assert [row["nmse"] for row in total_delay] == pytest.approx([0, 3, 6, 9])
    assert "retention/m04__t04__d002/total_nmse" in report.raw_errors
    np.testing.assert_array_equal(
        report.raw_errors["retention/m04__t04__d002/intervening_tasks"],
        [3, 3, 2, 1, 0],
    )


def test_retention_decomposition_is_omitted_without_the_shared_control() -> None:
    suites, mses, nmses = capability_fixture()
    del suites["retention_shared"], mses["retention_shared"], nmses["retention_shared"]
    report = _evaluate(
        suites,
        mses,
        nmses,
        {},
        bootstrap_seed=11,
        bootstrap_replicates=10,
    )
    metrics = {row["metric"] for row in report.summary_rows}
    assert "savings_mean" in metrics
    assert "episodic_savings_mean" not in metrics
    assert "module_savings_mean" not in metrics


def test_load_eval_suites_missing_dir_points_at_generation_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        load_eval_suites(tmp_path)
