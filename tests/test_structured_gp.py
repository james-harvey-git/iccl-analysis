import math

import torch

from iccl.analysis.structured_observer.gp import (
    BatchedOnlineGP,
    batch_gp_factor,
    batch_log_marginal_likelihood,
    extend_gp_factor,
    log_marginal_likelihood_from_factor,
)


def test_online_predictions_match_batch_conditioning() -> None:
    generator = torch.Generator().manual_seed(4)
    features = torch.randn((3, 5, 7), generator=generator, dtype=torch.float64)
    targets = torch.randn((5, 2), generator=generator, dtype=torch.float64)
    jitter = 1e-5
    gp = BatchedOnlineGP(
        num_hypotheses=3,
        num_features=7,
        output_dim=2,
        max_observations=5,
        jitter=jitter,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    for index in range(5):
        prediction = gp.predict(features[:, index])
        if index:
            history = features[:, :index]
            kernel = history @ history.transpose(-1, -2)
            identity = torch.eye(index, dtype=torch.float64)
            cross = torch.einsum("hnj,hj->hn", history, features[:, index])
            alpha = torch.linalg.solve(
                kernel + jitter * identity.unsqueeze(0),
                targets[:index].unsqueeze(0).expand(3, -1, -1),
            )
            expected_mean = torch.einsum("hn,hno->ho", cross, alpha)
            expected_variance = (
                torch.sum(features[:, index].square(), dim=-1)
                + jitter
                - torch.einsum(
                    "hn,hn->h",
                    cross,
                    torch.linalg.solve(
                        kernel + jitter * identity.unsqueeze(0),
                        cross.unsqueeze(-1),
                    ).squeeze(-1),
                )
            )
            assert torch.allclose(prediction.mean, expected_mean, atol=1e-10)
            assert torch.allclose(prediction.variance, expected_variance, atol=1e-10)
        gp.append(features[:, index], targets[index], prediction)


def test_sum_of_online_predictive_densities_equals_marginal_likelihood() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn((2, 6, 5), generator=generator, dtype=torch.float64)
    targets = torch.randn((6, 3), generator=generator, dtype=torch.float64)
    gp = BatchedOnlineGP(
        num_hypotheses=2,
        num_features=5,
        output_dim=3,
        max_observations=6,
        jitter=1e-4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    total = torch.zeros(2, dtype=torch.float64)
    for index in range(6):
        _, log_density = gp.predict_and_append(features[:, index], targets[index])
        total += log_density
    batch = batch_log_marginal_likelihood(features, targets, 1e-4)
    assert torch.allclose(total, batch, atol=1e-9)


def test_resampling_preserves_selected_gp_states() -> None:
    gp = BatchedOnlineGP(
        num_hypotheses=3,
        num_features=2,
        output_dim=1,
        max_observations=2,
        jitter=1e-4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    gp.predict_and_append(features, torch.tensor([0.5], dtype=torch.float64))
    before = gp.features.clone()
    indices = torch.tensor([2, 2, 0])
    gp.resample(indices)
    assert torch.equal(gp.features, before.index_select(0, indices))
    assert math.isclose(gp.jitter, 1e-4)


def test_extending_prefix_factor_matches_full_history_refactorization() -> None:
    generator = torch.Generator().manual_seed(19)
    features = torch.randn((4, 7, 6), generator=generator, dtype=torch.float64)
    targets = torch.randn((7, 2), generator=generator, dtype=torch.float64)
    jitter = 1e-4
    base_features = features[:, :4]
    base_cholesky, base_whitened = batch_gp_factor(
        base_features,
        targets[:4],
        jitter,
    )
    extended_cholesky, extended_whitened, block_log_likelihood = extend_gp_factor(
        base_features,
        base_cholesky,
        base_whitened,
        features[:, 4:],
        targets[4:],
        jitter,
    )
    full_cholesky, full_whitened = batch_gp_factor(features, targets, jitter)

    assert torch.allclose(extended_cholesky, full_cholesky, atol=1e-10)
    assert torch.allclose(extended_whitened, full_whitened, atol=1e-10)
    full_log_likelihood = log_marginal_likelihood_from_factor(
        full_cholesky,
        full_whitened,
    )
    base_log_likelihood = log_marginal_likelihood_from_factor(
        base_cholesky,
        base_whitened,
    )
    assert torch.allclose(
        block_log_likelihood,
        full_log_likelihood - base_log_likelihood,
        atol=1e-9,
    )
