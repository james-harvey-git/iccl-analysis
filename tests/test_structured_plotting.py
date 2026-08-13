from pathlib import Path

import numpy as np

from iccl.analysis.structured_observer.plotting import (
    paired_bootstrap_methods,
    plot_structured_observer_comparison,
)


def test_shared_bootstrap_preserves_exact_paired_difference() -> None:
    base = np.arange(20, dtype=np.float64).reshape(5, 4)
    result = paired_bootstrap_methods(
        {
            "gdn": base + 3.0,
            "full_history": base,
            "current_task": base + 2.0,
        },
        differences=(("current_task", "full_history"), ("gdn", "full_history")),
        replicates=100,
        seed=4,
    )
    assert np.array_equal(result["current_task_minus_full_history_mean"], np.full(4, 2.0))
    assert np.array_equal(result["current_task_minus_full_history_ci_low"], np.full(4, 2.0))
    assert np.array_equal(result["gdn_minus_full_history_ci_high"], np.full(4, 3.0))


def test_comparison_writes_figures_arrays_and_provenance(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    shape = (6, 3, 4)
    gdn = rng.uniform(0.2, 1.0, size=shape).astype(np.float32)
    full = rng.uniform(0.1, 0.8, size=shape).astype(np.float32)
    current = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
    cache = {
        "sequence_indices": np.arange(6, dtype=np.int32),
        "task_positions": np.arange(1, 4, dtype=np.int16),
        "demo_counts": np.full((6, 3), 4, dtype=np.int64),
        "full_nmse": full,
        "current_task_nmse": current,
        "full_raw_mse": full / 2,
        "current_task_raw_mse": current / 2,
        "all_current_modules_seen": rng.random((6, 3)) > 0.5,
        "full_ess_after_by_seed": np.full((2, 6, 3, 4), 12.0, dtype=np.float32),
        "full_max_weight_after_by_seed": np.full(
            (2, 6, 3, 4), 0.1, dtype=np.float32
        ),
        "full_unique_prefixes_after_by_seed": np.full(
            (2, 6, 3, 4), 8, dtype=np.int32
        ),
        "full_resampled_by_seed": np.zeros((2, 6, 3, 4), dtype=bool),
        "full_predictive_covariance_trace_by_seed": np.ones(
            (2, 6, 3, 4), dtype=np.float32
        ),
        "current_task_predictive_covariance_trace": np.ones(
            (6, 3, 4), dtype=np.float32
        ),
        "full_task_end_rejuvenation_acceptance_by_seed": np.full(
            (2, 6, 3), 0.4, dtype=np.float32
        ),
        "full_algorithmic_prediction_std": np.full(
            (6, 3, 4, 2), 0.03, dtype=np.float32
        ),
    }
    written = plot_structured_observer_comparison(
        gdn,
        gdn / 2,
        cache,
        suite_name="in_dist",
        task_position=3,
        checkpoint_step=400_000,
        output_root=tmp_path,
        bootstrap_replicates=50,
        bootstrap_seed=0,
        include_raw_mse=True,
        provenance={"test": True},
    )
    output_dir = tmp_path / "pilot-400k-steps"
    assert output_dir.is_dir()
    assert any(path.suffix == ".png" for path in written)
    assert any(path.suffix == ".pdf" for path in written)
    assert any(path.name.endswith("plotted-arrays.npz") for path in written)
    assert any(path.name.endswith("provenance.json") for path in written)
    assert all(path.exists() for path in written)
