import json
from pathlib import Path

import numpy as np

from iccl.analysis.plotting import paired_bootstrap_curves, plot_bayes_oracle_comparison


def test_paired_bootstrap_preserves_a_sequence_level_difference() -> None:
    right = np.arange(24, dtype=np.float32).reshape(6, 4)
    left = right + np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    first = paired_bootstrap_curves(left, right, replicates=200, seed=7)
    second = paired_bootstrap_curves(left, right, replicates=200, seed=7)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(first["difference_mean"], [0.1, 0.2, 0.3, 0.4], atol=3e-7)
    np.testing.assert_allclose(first["difference_ci_low"], first["difference_mean"], atol=1e-6)
    np.testing.assert_allclose(first["difference_ci_high"], first["difference_mean"], atol=1e-6)


def test_comparison_plot_writes_figures_arrays_and_provenance(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    shape = (6, 2, 4)
    oracle_nmse = rng.uniform(0.05, 0.3, size=shape).astype(np.float32)
    oracle_raw = (oracle_nmse * 0.5).astype(np.float32)
    gdn_nmse = (oracle_nmse + rng.normal(0.1, 0.02, size=shape)).astype(np.float32)
    gdn_raw = (oracle_raw + rng.normal(0.05, 0.01, size=shape)).astype(np.float32)
    coverage = np.array(
        [
            [False, True],
            [False, True],
            [False, True],
            [False, False],
            [False, False],
            [False, False],
        ]
    )
    survivor_before = np.broadcast_to(np.array([216, 1, 1, 1], dtype=np.int32), shape).copy()
    oracle = {
        "nmse": oracle_nmse,
        "raw_mse": oracle_raw,
        "demo_counts": np.full((6, 2), 4, dtype=np.int64),
        "task_positions": np.array([1, 2], dtype=np.int16),
        "all_current_modules_seen": coverage,
        "survivor_count_before": survivor_before,
        "entropy_before": np.log(survivor_before).astype(np.float32),
        "unique_identified_after": survivor_before == 1,
        "true_posterior_mass_after": (1.0 / survivor_before).astype(np.float32),
        "predictive_covariance_trace": rng.uniform(0.0, 1.0, size=shape).astype(np.float32),
    }

    written = plot_bayes_oracle_comparison(
        gdn_nmse,
        gdn_raw,
        oracle,
        suite_name="in_dist",
        task_position=2,
        checkpoint_step=40,
        output_dir=tmp_path,
        bootstrap_replicates=50,
        bootstrap_seed=3,
        include_raw_mse=True,
        provenance={"checkpoint_reference": "checkpoint.pt"},
    )
    assert len(written) == 10
    assert all(path.exists() for path in written)
    assert sum(path.suffix == ".png" for path in written) == 4
    assert sum(path.suffix == ".pdf" for path in written) == 4

    arrays_path = next(path for path in written if path.name.endswith("plotted-arrays.npz"))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert arrays["demo_index"].tolist() == [0, 1, 2, 3]
        assert arrays["nmse_all_modules_seen_sample_size"] == 3
        assert arrays["nmse_module_unseen_sample_size"] == 3
        np.testing.assert_allclose(
            arrays["nmse_pooled_difference_mean"],
            np.mean(gdn_nmse[:, 1] - oracle_nmse[:, 1], axis=0),
        )

    provenance_path = next(path for path in written if path.suffix == ".json")
    provenance = json.loads(provenance_path.read_text())
    assert provenance["task_position"] == 2
    assert provenance["task_index_zero_based"] == 1
    assert provenance["confidence_intervals"].startswith("paired sequence-level")
