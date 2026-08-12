import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iccl.analysis.bayes_oracle import (
    OracleConfig,
    cache_identity,
    candidate_forward,
    compute_oracle,
    enumerate_candidates,
    generate_or_reuse_cache,
    load_oracle_cache,
    load_suite_metadata,
    module_pool_from_suite,
)
from iccl.data.dataset import sequence_rng
from iccl.data.export import export_suite, load_suite
from iccl.data.sequences import PhaseConfig, SequenceConfig, build_sequence
from iccl.data.teacher import HyperTeacher, TeacherConfig, teacher_forward

TEACHER = TeacherConfig(
    input_dim=3,
    output_dim=3,
    hidden_dims=(4,),
    use_bias=True,
    num_modules=4,
    scale=3.0,
    weighting="discrete",
)
SEQUENCE = SequenceConfig(
    phases=(PhaseConfig(num_tasks=3, hotness=(2, 2)),),
    demos_per_task=3,
    signal_boundaries=True,
    require_identifiable=True,
)


def metadata(out_dir: Path) -> dict[str, Any]:
    return {
        "suite": "in_dist",
        "seed": 0,
        "config": {
            "name": "hyperteacher",
            "input_dim": TEACHER.input_dim,
            "output_dim": TEACHER.output_dim,
            "hidden_dims": list(TEACHER.hidden_dims),
            "use_bias": TEACHER.use_bias,
            "num_modules": TEACHER.num_modules,
            "scale": TEACHER.scale,
            "weighting": TEACHER.weighting,
            "sequence": {
                "phases": [{"num_tasks": 3, "hotness": [2, 2]}],
                "demos_per_task": 3,
                "signal_boundaries": True,
                "require_identifiable": True,
            },
            "eval_sets": {
                "out_dir": str(out_dir),
                "composite": {"hotness": 2, "num_demos": 2},
            },
        },
    }


@pytest.fixture()
def frozen_suite(tmp_path: Path) -> tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]]:
    family = HyperTeacher(TEACHER, max_hotness=2)
    samples = [
        build_sequence(family, SEQUENCE, sequence_rng(10, index), include_world=True)
        for index in range(3)
    ]
    base = tmp_path / "in_dist"
    export_suite(samples, base, metadata(tmp_path))
    suite_path = base.with_suffix(".npz")
    metadata_path = base.with_suffix(".meta.json")
    return suite_path, metadata_path, load_suite(base), load_suite_metadata(metadata_path)


def _task_targets(suite: dict[str, np.ndarray]) -> np.ndarray:
    counts = suite["demo_counts"]
    targets = np.full(
        (*counts.shape, int(counts.max()), suite["targets"].shape[-1]),
        np.nan,
        dtype=np.float32,
    )
    for sequence in range(counts.shape[0]):
        for task in range(counts.shape[1]):
            count = int(counts[sequence, task])
            start = int(suite["task_spans"][sequence, task, 0])
            positions = start + 2 * np.arange(count)
            targets[sequence, task, :count] = suite["targets"][sequence, positions]
    return targets


def test_pilot_candidate_space_has_1008_unique_latents() -> None:
    candidates = enumerate_candidates(8, 2)
    assert candidates.shape == (math.comb(8, 2) * 6**2, 8) == (1008, 8)
    assert len(np.unique(candidates, axis=0)) == len(candidates)
    np.testing.assert_array_equal(candidates[0, :2], [0.5, 0.5])
    np.testing.assert_array_equal(candidates[-1, -2:], [1.0, 1.0])


def test_vectorized_forward_matches_teacher_and_chunking(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
) -> None:
    _, _, suite, _ = frozen_suite
    pool = module_pool_from_suite(suite, 0)
    candidates = enumerate_candidates(TEACHER.num_modules, 2)
    x = suite["tokens"][0, [1, 3, 5], : TEACHER.input_dim]
    vectorized = candidate_forward(pool, candidates, x, chunk_size=17)
    unchunked = candidate_forward(pool, candidates, x, chunk_size=len(candidates))
    np.testing.assert_array_equal(vectorized, unchunked)
    for index in (0, 17, len(candidates) - 1):
        np.testing.assert_allclose(
            vectorized[index], teacher_forward(pool, candidates[index], x), atol=2e-6, rtol=2e-6
        )


def test_oracle_resets_is_causal_and_keeps_the_true_candidate(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
) -> None:
    _, _, suite, suite_metadata = frozen_suite
    result = compute_oracle(suite, suite_metadata, OracleConfig(candidate_chunk_size=31))
    candidate_count = math.comb(TEACHER.num_modules, 2) * 6**2

    assert np.all(result["survivor_count_before"][:, :, 0] == candidate_count)
    assert np.all(result["true_posterior_mass_after"] > 0)
    assert np.all(result["sensitivity_true_survives_after"])
    assert np.all(np.diff(result["survivor_count_after"], axis=2) <= 0)
    assert np.all(result["identification_step"] >= 1)
    np.testing.assert_allclose(
        result["raw_mse"],
        np.mean((result["predictions"] - _task_targets(suite)) ** 2, axis=-1),
    )

    pool = module_pool_from_suite(suite, 0)
    candidates = result["candidates"]
    first_start = int(suite["task_spans"][0, 0, 0])
    first_x = suite["tokens"][0, first_start : first_start + 1, : TEACHER.input_dim]
    expected = candidate_forward(pool, candidates, first_x, chunk_size=19)[:, 0].mean(axis=0)
    np.testing.assert_allclose(result["predictions"][0, 0, 0], expected, rtol=1e-6)

    outputs = candidate_forward(pool, candidates, first_x, chunk_size=19)[:, 0].astype(np.float64)
    covariance = np.cov(outputs, rowvar=False, bias=True)
    np.testing.assert_allclose(
        result["predictive_variance_diag"][0, 0, 0], np.diag(covariance), rtol=1e-5
    )


def test_previous_task_changes_do_not_change_the_next_task(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
) -> None:
    _, _, suite, suite_metadata = frozen_suite
    original = {key: value[:1].copy() for key, value in suite.items()}
    changed = {key: value.copy() for key, value in original.items()}
    candidates = enumerate_candidates(TEACHER.num_modules, 2)
    replacement = candidates[-1]
    if np.array_equal(replacement, changed["latents"][0, 0]):
        replacement = candidates[-2]
    pool = module_pool_from_suite(changed, 0)
    start = int(changed["task_spans"][0, 0, 0])
    count = int(changed["demo_counts"][0, 0])
    positions = start + 2 * np.arange(count)
    x = changed["tokens"][0, positions, : TEACHER.input_dim]
    y = teacher_forward(pool, replacement, x)
    changed["latents"][0, 0] = replacement
    changed["targets"][0, positions] = y
    changed["tokens"][0, positions + 1, : TEACHER.output_dim] = y
    changed["base_mse"][0, 0] = ((y - y.mean(axis=0)) ** 2).mean(axis=0)

    left = compute_oracle(original, suite_metadata, OracleConfig())
    right = compute_oracle(changed, suite_metadata, OracleConfig())
    for key in (
        "predictions",
        "survivor_count_before",
        "survivor_count_after",
        "entropy_before",
        "entropy_after",
    ):
        np.testing.assert_array_equal(left[key][:, 1:], right[key][:, 1:])


def test_cache_reuse_and_stale_identity_detection(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    suite_path, metadata_path, _, suite_metadata = frozen_suite
    config = OracleConfig(candidate_chunk_size=29)
    cache_path, generated = generate_or_reuse_cache(
        suite_path, metadata_path, tmp_path / "cache", config
    )
    assert generated
    assert generate_or_reuse_cache(suite_path, metadata_path, tmp_path / "cache", config) == (
        cache_path,
        False,
    )
    identity = cache_identity(suite_path, metadata_path, suite_metadata, config)
    arrays, sidecar = load_oracle_cache(cache_path, expected_identity=identity)
    assert arrays["predictions"].shape[:3] == (3, 3, 3)
    assert sidecar["identity"] == identity
    stale = copy.deepcopy(identity)
    stale["oracle_config"]["atol"] = 1e-3
    with pytest.raises(ValueError, match="stale oracle cache"):
        load_oracle_cache(cache_path, expected_identity=stale)


def test_unsupported_and_zero_survivor_failures_are_explicit(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
) -> None:
    _, _, suite, suite_metadata = frozen_suite
    unsupported = copy.deepcopy(suite_metadata)
    unsupported["config"]["weighting"] = "continuous"
    with pytest.raises(ValueError, match="weighting=discrete"):
        compute_oracle(suite, unsupported, OracleConfig())

    corrupted = {key: value[:1].copy() for key, value in suite.items()}
    first = int(corrupted["task_spans"][0, 0, 0])
    corrupted["targets"][0, first] += 1000.0
    with pytest.raises(RuntimeError, match="zero survivors"):
        compute_oracle(corrupted, suite_metadata, OracleConfig())


def test_cache_sidecar_is_json_without_pickle(
    frozen_suite: tuple[Path, Path, dict[str, np.ndarray], dict[str, Any]],
    tmp_path: Path,
) -> None:
    suite_path, metadata_path, _, _ = frozen_suite
    cache_path, _ = generate_or_reuse_cache(
        suite_path, metadata_path, tmp_path / "cache", OracleConfig()
    )
    sidecar = json.loads(cache_path.with_suffix(".meta.json").read_text())
    assert sidecar["identity"]["schema_version"] == 1
    with np.load(cache_path, allow_pickle=False) as data:
        assert data["predictions"].dtype == np.float32
