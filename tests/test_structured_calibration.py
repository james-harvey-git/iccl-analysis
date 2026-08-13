import numpy as np

from iccl.analysis.structured_observer.calibration import convergence_statistics


def test_convergence_statistics_apply_declared_thresholds() -> None:
    curves = np.zeros((2, 2, 3, 4, 5), dtype=np.float64)
    curves[0, 0] = 0.03
    curves[0, 1] = 0.015
    curves[1, 0] = 0.01
    curves[1, 1, 0] = 0.001
    curves[1, 1, 1] = 0.0
    curves[1, 1, 2] = -0.001
    arrays, summary = convergence_statistics(
        curves,
        np.array([0.03, 0.0]),
        np.full((2, 2), 1e-8),
        kernel_threshold=0.04,
        curve_threshold=0.02,
        seed_standard_error_threshold=0.01,
        jitter_threshold=1e-5,
    )
    assert arrays["mean_curves"].shape == (2, 2, 5)
    assert arrays["reference_seed_standard_error"].shape == (5,)
    assert summary["passes"]["kernel"]
    assert summary["passes"]["feature_curve"]
    assert summary["passes"]["particle_curve"]
    assert summary["passes"]["seed_standard_error"]
    assert summary["passes"]["jitter"]
    assert summary["passes"]["all"]
