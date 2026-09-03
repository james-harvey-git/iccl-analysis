from typing import Any

import numpy as np
import pytest

from iccl.evaluation.metrics import _evaluate


def metadata(family: str, condition: str) -> dict[str, Any]:
    return {
        "capability": "retention_position",
        "condition": condition,
        "diagnostic_family": family,
        "pair_group": f"retention-position-{family}",
        "cell_id": "m04__t04__d002",
        "family_memberships": ["position_diagnostic"],
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


def add_conditions(
    suites: dict[str, dict[str, Any]],
    mses: dict[str, np.ndarray],
    nmses: dict[str, np.ndarray],
    family: str,
    total_savings: np.ndarray,
    **arrays: np.ndarray,
) -> None:
    rows = len(total_savings)
    for condition, fraction in (("repeat", 0.0), ("shared", 0.25), ("novel", 1.0)):
        name = f"{family}_{condition}"
        suites[name] = {"__meta__": metadata(family, condition)} | arrays
        errors = np.zeros((rows, 5, 2), dtype=np.float64)
        errors[:, -1] = fraction * total_savings[:, None]
        mses[name] = errors
        nmses[name] = errors.copy()


def diagnostic_fixture() -> tuple[
    dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    suites: dict[str, dict[str, Any]] = {}
    mses: dict[str, np.ndarray] = {}
    nmses: dict[str, np.ndarray] = {}

    paired_values = np.array([[10, 1, 2, 8], [12, 2, 2, 9], [11, 3, 1, 7]], dtype=np.float64)
    paired_positions = np.tile(np.arange(4), 3)
    paired_groups = np.repeat(np.arange(3), 4)
    add_conditions(
        suites,
        mses,
        nmses,
        "paired_permutation",
        paired_values.reshape(-1),
        original_task_position=paired_positions,
        intervening_tasks=3 - paired_positions,
        position_group_id=paired_groups,
        pair_id=np.arange(12),
        rehearsal_mode=np.full(12, "natural"),
        support_status=np.full(12, "connected_id"),
        target_support=np.tile([0, 1], (12, 1)),
    )

    modes = np.array(["none", "one", "both"])
    rehearsal_groups = np.repeat(np.arange(3), 6)
    rehearsal_positions = np.tile(np.repeat([0, 1], 3), 3)
    rehearsal_modes = np.tile(modes, 6)
    mode_values = {"none": 1.0, "one": 2.0, "both": 4.0}
    rehearsal_values = np.array(
        [
            mode_values[mode] + 0.1 * position
            for position, mode in zip(rehearsal_positions, rehearsal_modes, strict=True)
        ]
    )
    statuses = np.array(
        [
            "disconnected_ood" if position == 0 and mode == "none" else "connected_id"
            for position, mode in zip(rehearsal_positions, rehearsal_modes, strict=True)
        ]
    )
    add_conditions(
        suites,
        mses,
        nmses,
        "controlled_rehearsal",
        rehearsal_values,
        original_task_position=rehearsal_positions,
        intervening_tasks=3 - rehearsal_positions,
        position_group_id=rehearsal_groups,
        pair_id=np.arange(18),
        rehearsal_mode=rehearsal_modes,
        support_status=statuses,
        target_support=np.tile([0, 1], (18, 1)),
    )
    return suites, mses, nmses


def test_paired_position_contrasts_are_computed_within_world() -> None:
    suites, mses, nmses = diagnostic_fixture()
    report = _evaluate(suites, mses, nmses, {}, bootstrap_seed=3, bootstrap_replicates=30)
    reseeded = _evaluate(suites, mses, nmses, {}, bootstrap_seed=99, bootstrap_replicates=30)
    assert report.scalars == reseeded.scalars
    for key in report.curves:
        np.testing.assert_array_equal(report.curves[key], reseeded.curves[key])
    rows = {
        (row["metric"], row["retention_component"]): row
        for row in report.summary_rows
        if row["diagnostic_family"] == "paired_permutation"
    }
    assert rows[("primacy_excess_mean", "total")]["value"] == pytest.approx(
        np.mean([8.5, 10.0, 9.0])
    )
    assert rows[("recency_excess_mean", "total")]["value"] == pytest.approx(
        np.mean([6.5, 7.0, 5.0])
    )
    assert rows[("edge_excess_mean", "total")]["value"] == pytest.approx(np.mean([7.5, 8.5, 7.0]))

    position = [
        row
        for row in report.curve_rows
        if row["curve_type"] == "retention_position" and row["retention_component"] == "total"
    ]
    assert [row["x_value"] for row in position] == [0, 1, 2, 3]
    assert [row["nmse"] for row in position] == pytest.approx([11, 2, 5 / 3, 8])
    assert all(row["n_sequences"] == 3 for row in position)


def test_rehearsal_effects_are_paired_and_ood_status_is_preserved() -> None:
    suites, mses, nmses = diagnostic_fixture()
    report = _evaluate(suites, mses, nmses, {}, bootstrap_seed=7, bootstrap_replicates=20)
    effects = [
        row
        for row in report.summary_rows
        if row["metric"] == "rehearsal_effect_mean" and row["retention_component"] == "total"
    ]
    assert len(effects) == 4
    assert {
        (row["original_task_position"], row["rehearsal_mode"]): row["value"] for row in effects
    } == pytest.approx({(0, "one"): 1, (0, "both"): 3, (1, "one"): 1, (1, "both"): 3})
    assert {row["support_status"] for row in effects if row["original_task_position"] == 0} == {
        "includes_disconnected_ood"
    }

    no_rehearsal = [
        row
        for row in report.curve_rows
        if row["curve_type"] == "retention_rehearsal"
        and row["retention_component"] == "total"
        and row["rehearsal_mode"] == "none"
    ]
    assert no_rehearsal[0]["support_status"] == "disconnected_ood"
    assert no_rehearsal[1]["support_status"] == "connected_id"


def test_diagnostic_raw_output_contains_pairing_metadata_and_exact_decomposition() -> None:
    suites, mses, nmses = diagnostic_fixture()
    report = _evaluate(suites, mses, nmses, {}, bootstrap_seed=0, bootstrap_replicates=0)
    for family in ("paired_permutation", "controlled_rehearsal"):
        prefix = f"retention_position/{family}"
        assert f"{prefix}/position_group_id" in report.raw_errors
        assert f"{prefix}/target_support" in report.raw_errors
        total = report.raw_errors[f"{prefix}/total_nmse"]
        episodic = report.raw_errors[f"{prefix}/episodic_nmse"]
        module = report.raw_errors[f"{prefix}/module_nmse"]
        np.testing.assert_allclose(total, episodic + module)


def test_binary_weighting_reports_only_total_savings() -> None:
    suites, mses, nmses = diagnostic_fixture()
    for mapping in (suites, mses, nmses):
        for name in [name for name in mapping if name.endswith("_shared")]:
            del mapping[name]
    report = _evaluate(suites, mses, nmses, {}, bootstrap_seed=0, bootstrap_replicates=0)
    components = {row["retention_component"] for row in report.curve_rows}
    assert components == {"total"}
