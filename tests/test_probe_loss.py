from pathlib import Path

import numpy as np
import pytest
import torch

from iccl.analysis.probe_loss import aligned_targets, parameter_errors
from iccl.analysis.probe_matching import AssignmentSolver
from iccl.analysis.probe_targets import MODULE_FEATURES, OUTPUT_FEATURES
from iccl.analysis.probes import LinearModuleDecoder


@pytest.mark.parametrize("weight", [1.0, 17.0])
def test_native_objective_and_selected_branch_gradient(tmp_path: Path, weight: float) -> None:
    rng = np.random.default_rng(302)
    target = rng.normal(size=(2, OUTPUT_FEATURES)).astype(np.float32)
    prediction = rng.normal(size=target.shape).astype(np.float32)
    with AssignmentSolver(2, tmp_path) as solver:
        assignment = solver.match(prediction, target, weight)
    manual = np.empty_like(target)
    for e in range(2):
        modules = target[e, :MODULE_FEATURES].reshape(8, 2, 17, 16)
        p = assignment.permutations[e]
        expected = np.stack(
            [
                modules[
                    t, [((assignment.masks[e] >> t) & 1), 1 ^ ((assignment.masks[e] >> t) & 1)]
                ][..., p]
                for t in range(8)
            ]
        )
        manual[e] = np.concatenate(
            (expected.ravel(), target[e, MODULE_FEATURES:].reshape(16, 16)[p].ravel())
        )
    values = torch.tensor(prediction, requires_grad=True)
    aligned = aligned_targets(torch.from_numpy(target), assignment)
    torch.testing.assert_close(aligned, torch.from_numpy(manual), rtol=0, atol=0)
    loss = parameter_errors(values, aligned, weight).joint.mean()
    np.testing.assert_allclose(loss.item(), assignment.scores.mean() / OUTPUT_FEATURES, rtol=3e-6)
    loss.backward()
    expected_gradient = 2 * (prediction - manual) / (2 * OUTPUT_FEATURES)
    expected_gradient[:, MODULE_FEATURES:] *= weight
    assert values.grad is not None
    np.testing.assert_allclose(values.grad.numpy(), expected_gradient, rtol=1e-6, atol=1e-9)


def test_full_decoder_parameter_count_without_allocating_weights() -> None:
    with torch.device("meta"):
        model = LinearModuleDecoder(131072)
    assert sum(p.numel() for p in model.parameters()) == 603984384
