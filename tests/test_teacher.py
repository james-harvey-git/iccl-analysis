import math
from dataclasses import replace
from typing import Any

import numpy as np

from iccl.data.dataset import sequence_rng
from iccl.data.teacher import (
    HyperTeacher,
    ModulePool,
    TeacherConfig,
    make_combinations,
    sample_module_pool,
    teacher_forward,
)

DEFAULT_CONFIG = TeacherConfig(
    input_dim=4,
    output_dim=3,
    hidden_dims=(5,),
    use_bias=True,
    num_modules=6,
    scale=3.0,
    weighting="discrete",
)


def make_config(**overrides: Any) -> TeacherConfig:
    return replace(DEFAULT_CONFIG, **overrides)


def test_forward_matches_hand_computed_composition() -> None:
    pool = ModulePool(
        modules=[np.stack([np.eye(2, dtype=np.float32), 2 * np.eye(2, dtype=np.float32)])],
        biases=[np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)],
        readout=np.array([[1.0], [1.0]], dtype=np.float32),
    )
    latent = np.array([1.0, 1.0], dtype=np.float32)
    x = np.array([[1.0, -1.0]], dtype=np.float32)

    # Composed weights: (I + 2I)/sqrt(2), bias (0+1)/sqrt(2); then ReLU and sum-readout.
    scale = 1 / math.sqrt(2)
    pre = x @ (3 * scale * np.eye(2)) + scale
    expected = np.maximum(pre, 0.0).sum(axis=1, keepdims=True)

    np.testing.assert_allclose(teacher_forward(pool, latent, x), expected, rtol=1e-6)


def test_module_pool_shapes_and_bias_range() -> None:
    cfg = make_config(hidden_dims=(5, 7))
    pool = sample_module_pool(cfg, sequence_rng(0, 0))
    assert [m.shape for m in pool.modules] == [(6, 4, 5), (6, 5, 7)]
    assert [b.shape for b in pool.biases] == [(6, 5), (6, 7)]
    assert pool.readout.shape == (7, 3)
    for b in pool.biases:
        assert (b >= 0).all() and (b < 0.5).all()


def test_weight_std_tracks_variance_scaling() -> None:
    cfg = make_config(input_dim=64, hidden_dims=(64,), num_modules=32, scale=3.0)
    pool = sample_module_pool(cfg, sequence_rng(0, 0))
    # Truncation at +-2 sigma with jax's correction leaves std within a few
    # percent of sqrt(scale/fan_in).
    expected = math.sqrt(3.0 / 64)
    assert abs(pool.modules[0].std() - expected) / expected < 0.05


def test_make_combinations_counts() -> None:
    combos = make_combinations(6, [1, 2])
    assert combos.shape == (6 + 15, 6)
    assert set(combos.sum(axis=1)) == {1, 2}


def test_weighting_modes() -> None:
    rng = sequence_rng(0, 0)
    pattern = np.array([1, 0, 1, 0, 0, 1], dtype=np.int8)

    binary = HyperTeacher(make_config(weighting="binary"), 3).apply_weighting(rng, pattern)
    np.testing.assert_array_equal(binary, pattern.astype(np.float32))

    for mode in ("discrete", "continuous"):
        family = HyperTeacher(make_config(weighting=mode), 3)
        latent = family.apply_weighting(rng, pattern)
        active = latent[pattern > 0]
        assert (latent[pattern == 0] == 0).all()
        assert (active >= 0.5).all() and (active <= 1.0).all()


def test_sample_pattern_has_requested_hotness() -> None:
    family = HyperTeacher(make_config(), max_hotness=2)
    rng = sequence_rng(0, 0)
    all_2hot = {tuple(row) for row in family.combos[2]}
    assert len(all_2hot) == math.comb(6, 2)
    for _ in range(20):
        pattern = family.sample_pattern(rng, 2)
        assert pattern.sum() == 2 and tuple(pattern) in all_2hot
