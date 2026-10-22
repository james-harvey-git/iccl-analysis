"""Numerical checks against an independent centered ridge solve."""

import numpy as np
import pytest
import torch

from iccl.analysis.probe_training import cpu_state
from iccl.analysis.probes import ArrayMatrix, fit_ridge, predict_ridge

OPTIONS = {
    "method": "direct",
    "dtype": "float64",
    "direct_max_dimension": 256,
    "max_working_mb": 32,
    "max_iterations": 500,
    "relative_tolerance": 1e-9,
    "row_batch_size": 17,
}


@pytest.mark.parametrize("shape", [(90, 12), (25, 110)])
@pytest.mark.parametrize("method", ["direct", "cg"])
def test_ridge_matches_independent_primal_and_predicts_linear_signal(shape, method):
    rng = np.random.default_rng(7)
    x = rng.normal(size=shape)
    x[:, -1] = 3
    y = x @ rng.normal(size=(shape[1], 3)) + 4
    matrix = ArrayMatrix(x)
    bundle, diagnostics = fit_ridge(
        matrix, y, 0.03, device=torch.device("cpu"), options=OPTIONS | {"method": method}
    )
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[-1] = 1
    standardized = (x - mean) / scale
    expected = np.linalg.solve(
        standardized.T @ standardized + len(x) * 0.03 * np.eye(shape[1]),
        standardized.T @ (y - y.mean(axis=0)),
    )
    actual = predict_ridge(bundle, matrix, torch.device("cpu"), 13)
    np.testing.assert_allclose(
        actual, standardized @ expected + y.mean(axis=0), rtol=1e-7, atol=1e-7
    )
    assert diagnostics["converged"]
    assert ((actual - y) ** 2).mean() < y.var() * 0.01
    assert bundle["weights"][-1].abs().max() == 0


def test_cg_resume_matches_uninterrupted_and_respects_dense_memory_bound():
    rng = np.random.default_rng(8)
    x, y = ArrayMatrix(rng.normal(size=(80, 50))), rng.normal(size=(80, 4))
    states = []
    options = OPTIONS | {"method": "cg"}
    bundle, _ = fit_ridge(
        x,
        y,
        0.1,
        device=torch.device("cpu"),
        options=options,
        log_every=1,
        callback=lambda diagnostic, state: states.append(cpu_state(state)),
    )
    resumed, _ = fit_ridge(x, y, 0.1, device=torch.device("cpu"), options=options, resume=states[4])
    torch.testing.assert_close(bundle["weights"], resumed["weights"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="dimension/memory"):
        fit_ridge(
            x, y, 0.1, device=torch.device("cpu"), options=OPTIONS | {"max_working_mb": 0.001}
        )
    _, failed = fit_ridge(
        x, y, 0.1, device=torch.device("cpu"), options=options | {"max_iterations": 1}
    )
    assert not failed["converged"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness check")
def test_cuda_ridge_matches_cpu_direct_and_cg():
    rng = np.random.default_rng(12)
    matrix = ArrayMatrix(rng.normal(size=(80, 160)))
    targets = rng.normal(size=(80, 8))
    expected, _ = fit_ridge(matrix, targets, 0.1, device=torch.device("cpu"), options=OPTIONS)
    for method in ("direct", "cg"):
        actual, diagnostic = fit_ridge(
            matrix, targets, 0.1, device=torch.device("cuda"), options=OPTIONS | {"method": method}
        )
        assert diagnostic["converged"]
        torch.testing.assert_close(actual["weights"], expected["weights"], rtol=1e-7, atol=1e-8)
