import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from iccl.analysis.structured_observer.cache import (
    KernelSettings,
    StructuredObserverSettings,
    cache_identity,
    default_cache_path,
    load_cache,
    merge_seed_caches,
    structured_suite_spec,
    write_cache,
)
from iccl.analysis.structured_observer.smc import SMCConfig


def _metadata() -> dict[str, object]:
    return {
        "suite": "in_dist",
        "config": {
            "input_dim": 2,
            "output_dim": 1,
            "hidden_dims": [4],
            "use_bias": True,
            "num_modules": 3,
            "scale": 1.5,
            "weighting": "discrete",
            "sequence": {
                "phases": [{"num_tasks": 3, "hotness": [2, 2]}],
                "signal_boundaries": True,
                "require_identifiable": True,
                "task_graph": "random",
            },
        },
    }


def _settings() -> StructuredObserverSettings:
    return StructuredObserverSettings(
        modes=("full_history", "current_task"),
        device="cpu",
        kernel=KernelSettings(
            num_features=8,
            seed=0,
            dtype="float64",
            relative_jitter=1e-6,
            max_relative_jitter=1e-3,
        ),
        smc=SMCConfig(
            num_particles=4,
            ess_fraction=0.5,
            task_end_rejuvenation_sweeps=0,
        ),
        smc_seeds=(0,),
        sequence_limit=1,
    )


def test_suite_spec_rejects_non_in_distribution_suite() -> None:
    metadata = _metadata()
    metadata["suite"] = "composite"
    with pytest.raises(ValueError, match="in_dist"):
        structured_suite_spec(metadata)


def test_cache_round_trip_and_identity_validation(tmp_path: Path) -> None:
    suite_path = tmp_path / "in_dist.npz"
    metadata_path = tmp_path / "in_dist.meta.json"
    np.savez_compressed(suite_path, placeholder=np.array([1]))
    metadata_path.write_text(json.dumps(_metadata()))
    identity = cache_identity(suite_path, metadata_path, _metadata(), _settings())
    cache_path = default_cache_path(tmp_path / "cache", suite_path, identity)
    arrays = {"predictions": np.arange(6, dtype=np.float32).reshape(2, 3)}
    write_cache(cache_path, arrays, identity, feature_bank_hash="abc")
    loaded, metadata = load_cache(cache_path, expected_identity=identity)
    assert np.array_equal(loaded["predictions"], arrays["predictions"])
    assert metadata["feature_bank_sha256"] == "abc"

    stale = dict(identity)
    stale["schema_version"] = 999
    with pytest.raises(ValueError, match="stale"):
        load_cache(cache_path, expected_identity=stale)


def test_disjoint_seed_caches_merge_into_monolithic_seed_axis(tmp_path: Path) -> None:
    suite_path = tmp_path / "in_dist.npz"
    metadata_path = tmp_path / "in_dist.meta.json"
    np.savez_compressed(suite_path, placeholder=np.array([1]))
    metadata_path.write_text(json.dumps(_metadata()))
    base_identity = cache_identity(suite_path, metadata_path, _metadata(), _settings())
    cache_paths = []
    for seed, value in [(1, 1.0), (0, 0.0)]:
        identity = deepcopy(base_identity)
        identity["settings"]["smc_seeds"] = [seed]
        predictions = np.full((1, 1, 1, 2, 1), value, dtype=np.float32)
        arrays = {
            "sequence_indices": np.array([0], dtype=np.int32),
            "task_positions": np.array([1], dtype=np.int16),
            "demo_counts": np.array([[2]], dtype=np.int64),
            "smc_seeds": np.array([seed], dtype=np.int64),
            "targets": np.zeros((1, 1, 2, 1), dtype=np.float32),
            "base_mse": np.ones((1, 1, 1), dtype=np.float32),
            "full_predictions_by_seed": predictions,
            "full_raw_mse_by_seed": np.full(
                (1, 1, 1, 2), value**2, dtype=np.float32
            ),
            "full_nmse_by_seed": np.full(
                (1, 1, 1, 2), value**2, dtype=np.float32
            ),
            "full_predictions_mean": predictions[0],
            "full_algorithmic_prediction_std": np.zeros_like(predictions[0]),
            "full_raw_mse": np.full((1, 1, 2), value**2, dtype=np.float32),
            "full_nmse": np.full((1, 1, 2), value**2, dtype=np.float32),
        }
        path = tmp_path / f"seed-{seed}.npz"
        write_cache(path, arrays, identity, feature_bank_hash="same-bank")
        cache_paths.append(path)

    output_path = tmp_path / "merged.npz"
    merged_path = merge_seed_caches(cache_paths, output_path)
    arrays, metadata = load_cache(merged_path)
    assert np.array_equal(arrays["smc_seeds"], [0, 1])
    assert np.array_equal(
        arrays["full_predictions_by_seed"][:, 0, 0, 0, 0],
        [0.0, 1.0],
    )
    assert np.allclose(arrays["full_predictions_mean"], 0.5)
    assert np.allclose(arrays["full_raw_mse"], 0.25)
    assert np.allclose(arrays["full_nmse"], 0.25)
    assert metadata["identity"]["settings"]["smc_seeds"] == [0, 1]
