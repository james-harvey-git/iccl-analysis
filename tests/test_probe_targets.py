import numpy as np
import pytest
import torch

from iccl.analysis.probe_loss import unpack_parameters
from iccl.analysis.probe_targets import canonical_pool, episode_targets, flat_targets
from iccl.data.teacher import ModulePool, TeacherConfig, sample_module_pool, teacher_forward


def test_scale_convention_preserves_teacher_and_unique_modules() -> None:
    rng = np.random.default_rng(9)
    pool = sample_module_pool(TeacherConfig(16, 16, (16,), True, 4, 3**0.5, "discrete"), rng)
    augmented, readout, norms = canonical_pool(pool)
    normalized = ModulePool([augmented[:, :16]], [augmented[:, 16]], readout)
    np.testing.assert_allclose(np.linalg.norm(readout, axis=1), 1, rtol=1e-6)
    latents = np.array(
        [
            [0.5, 0.8, 0, 0],
            [0, 0.9, 0.6, 0],
            [0, 0, 1, 0.7],
            [0.7, 1, 0, 0],
            [0.5, 0.8, 0, 0],
            [0.5, 0.8, 0, 0],
            [0.5, 0.8, 0, 0],
        ],
        dtype=np.float32,
    )
    records = episode_targets(pool, latents)
    np.testing.assert_array_equal(records["modules"], augmented)
    np.testing.assert_array_equal(records["occurrence_count"], [5, 6, 2, 1])
    np.testing.assert_allclose(records["readout_norms"], norms)
    for latent in latents:
        x = rng.normal(size=(64, 16)).astype(np.float32)
        np.testing.assert_allclose(
            teacher_forward(pool, latent, x),
            teacher_forward(normalized, latent, x),
            atol=3e-6,
            rtol=1e-4,
        )
    flat = flat_targets(augmented, readout)
    assert flat.shape == (1344,)
    np.testing.assert_array_equal(flat[:256], readout.ravel())
    modules_view, readout_view = unpack_parameters(torch.from_numpy(flat[None]))
    np.testing.assert_array_equal(modules_view[0].numpy(), augmented)
    np.testing.assert_array_equal(readout_view[0].numpy(), readout)
    # Relabelling the world preserves the represented set and permutes only its metadata.
    q = np.array([3, 0, 2, 1])
    relabelled = ModulePool([pool.modules[0][q]], [pool.biases[0][q]], pool.readout)
    other = episode_targets(relabelled, latents[:, q])
    np.testing.assert_array_equal(other["modules"], records["modules"][q])
    np.testing.assert_array_equal(other["occurrence_count"], records["occurrence_count"][q])


def test_zero_readout_norm_and_unseen_module_rejected() -> None:
    pool = ModulePool([np.ones((4, 16, 16))], [np.ones((4, 16))], np.ones((16, 16)))
    with pytest.raises(ValueError, match="all four"):
        episode_targets(pool, np.array([[1, 1, 0, 0]] * 7))
    pool.readout[3] = 0
    with pytest.raises(ValueError, match="norms"):
        canonical_pool(pool)
