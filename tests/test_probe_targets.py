import numpy as np
import pytest

from iccl.analysis.probe_targets import canonical_pool, episode_targets
from iccl.data.teacher import ModulePool, TeacherConfig, sample_module_pool, teacher_forward


def test_scale_convention_preserves_teacher_and_shared_modules() -> None:
    rng = np.random.default_rng(9)
    pool = sample_module_pool(TeacherConfig(16, 16, (16,), True, 8, 3**0.5, "discrete"), rng)
    augmented, readout, norms = canonical_pool(pool)
    normalized = ModulePool([augmented[:, :16]], [augmented[:, 16]], readout)
    np.testing.assert_allclose(np.linalg.norm(readout, axis=1), 1, rtol=1e-6)
    latents = np.eye(8, dtype=np.float32) + np.roll(np.eye(8, dtype=np.float32), 1, axis=1)
    latents[1] = latents[0]
    # Include module 2 elsewhere after replacing its task support.
    latents[2, 2] = 1
    records = episode_targets(pool, latents)
    np.testing.assert_array_equal(records["modules"][0], records["modules"][1])
    np.testing.assert_allclose(records["readout_norms"], norms)
    for latent in latents:
        x = rng.normal(size=(64, 16)).astype(np.float32)
        np.testing.assert_allclose(
            teacher_forward(pool, latent, x),
            teacher_forward(normalized, latent, x),
            atol=3e-6,
            rtol=1e-4,
        )


def test_zero_readout_norm_rejected() -> None:
    pool = ModulePool([np.ones((8, 16, 16))], [np.ones((8, 16))], np.ones((16, 16)))
    pool.readout[3] = 0
    with pytest.raises(ValueError, match="norms"):
        canonical_pool(pool)
