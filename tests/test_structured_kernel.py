import math

import numpy as np
import torch

from iccl.analysis.structured_observer.kernel import (
    FeatureBank,
    sample_feature_bank,
    validate_observer_device,
)
from iccl.data.teacher import sample_truncated_normal


def test_feature_bank_is_deterministic_and_kernel_is_psd() -> None:
    kwargs = {
        "input_dim": 3,
        "num_modules": 4,
        "scale": math.sqrt(3.0),
        "num_features": 128,
        "seed": 7,
    }
    first = sample_feature_bank(**kwargs)
    second = sample_feature_bank(**kwargs)
    assert torch.equal(first.module_weights, second.module_weights)
    assert torch.equal(first.module_biases, second.module_biases)

    rng = np.random.default_rng(3)
    inputs = rng.normal(size=(6, 3))
    latents = np.zeros((6, 4))
    for row in latents:
        row[rng.choice(4, size=2, replace=False)] = rng.choice([0.5, 1.0], size=2)
    features = torch.cat([first.features(x, z) for x, z in zip(inputs, latents, strict=True)])
    gram = features @ features.T
    assert torch.allclose(gram, gram.T)
    assert float(torch.linalg.eigvalsh(gram).min()) > -1e-10


def test_feature_banks_are_nested_across_feature_counts() -> None:
    kwargs = {
        "input_dim": 3,
        "num_modules": 4,
        "scale": 1.5,
        "seed": 12,
    }
    small = sample_feature_bank(**kwargs, num_features=16)
    large = sample_feature_bank(**kwargs, num_features=32)
    assert torch.equal(small.module_weights, large.module_weights[:16])
    assert torch.equal(small.module_biases, large.module_biases[:16])


def test_unavailable_cuda_can_be_validated_for_cache_identity() -> None:
    device = validate_observer_device("cuda", require_available=False)
    assert device.type == "cuda"


def test_feature_scale_matches_integrated_readout_factor() -> None:
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=2.0,
        num_features=64,
        seed=1,
    )
    x = np.array([0.25, -0.5])
    z = np.array([0.5, 1.0, 0.0])
    scaled_z = torch.as_tensor(z, dtype=torch.float64) / math.sqrt(2.0)
    preactivation = torch.einsum(
        "jmd,d->jm",
        bank.module_weights,
        torch.as_tensor(x, dtype=torch.float64),
    ) + bank.module_biases
    expected = torch.relu(preactivation @ scaled_z) * math.sqrt(2.0 / 64)
    assert torch.allclose(bank.features(x, z)[0], expected)


def test_history_features_equal_repeated_single_input_evaluation() -> None:
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=1.5,
        num_features=32,
        seed=9,
    )
    inputs = np.array([[0.1, 0.2], [-0.3, 0.4]])
    latents = np.array(
        [
            [[0.5, 1.0, 0.0], [0.0, 0.7, 0.9]],
            [[0.6, 0.8, 0.0], [0.5, 0.0, 1.0]],
        ]
    )
    history = bank.features_for_history(inputs, latents)
    for hypothesis in range(2):
        repeated = torch.stack(
            [bank.features(inputs[i], latents[hypothesis, i])[0] for i in range(2)]
        )
        assert torch.allclose(history[hypothesis], repeated)


def test_cached_module_projections_equal_direct_feature_evaluation() -> None:
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=1.5,
        num_features=32,
        seed=10,
    )
    inputs = np.array([[0.1, 0.2], [-0.3, 0.4]])
    latents = np.array([[0.5, 1.0, 0.0], [0.0, 0.7, 0.9]])
    projections = bank.module_preactivations(inputs)
    cached = bank.features_from_module_preactivations(projections, latents)
    direct = bank.features_for_history(
        inputs,
        np.repeat(latents[:, None], repeats=2, axis=1),
    )
    assert torch.equal(cached, direct)


def test_kernel_is_invariant_to_joint_module_permutation() -> None:
    bank = sample_feature_bank(
        input_dim=3,
        num_modules=4,
        scale=1.5,
        num_features=64,
        seed=21,
    )
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = FeatureBank(
        module_weights=bank.module_weights[:, permutation],
        module_biases=bank.module_biases[:, permutation],
        scale=bank.scale,
        seed=bank.seed,
    )
    x_a = np.array([0.2, -0.1, 0.4])
    x_b = np.array([-0.3, 0.5, 0.1])
    z_a = np.array([0.5, 0.0, 1.0, 0.0])
    z_b = np.array([0.0, 0.7, 0.0, 0.9])
    original = bank.kernel(x_a, z_a, x_b, z_b)
    reordered = permuted.kernel(x_a, z_a[permutation], x_b, z_b[permutation])
    assert torch.equal(original, reordered)


def test_random_feature_kernel_matches_sampled_teacher_covariance() -> None:
    input_dim = 2
    num_modules = 3
    hidden_dim = 16
    scale = 1.5
    x = np.array([[0.2, -0.4], [-0.3, 0.6]], dtype=np.float64)
    z = np.array([[0.5, 1.0, 0.0], [0.0, 0.7, 0.9]], dtype=np.float64)
    bank = sample_feature_bank(
        input_dim=input_dim,
        num_modules=num_modules,
        scale=scale,
        num_features=4096,
        seed=31,
    )
    features = torch.stack([bank.features(x_i, z_i)[0] for x_i, z_i in zip(x, z, strict=True)])
    feature_gram = (features @ features.T).numpy()

    worlds = 4000
    rng = np.random.default_rng(17)
    module_weights = sample_truncated_normal(
        rng,
        (worlds, hidden_dim, num_modules, input_dim),
        math.sqrt(scale / input_dim),
    ).astype(np.float64)
    module_biases = rng.uniform(
        0.0,
        0.5,
        size=(worlds, hidden_dim, num_modules),
    )
    readouts = sample_truncated_normal(
        rng,
        (worlds, hidden_dim),
        math.sqrt(scale / hidden_dim),
    ).astype(np.float64)
    outputs = []
    for x_i, z_i in zip(x, z, strict=True):
        scaled_z = z_i / math.sqrt(np.count_nonzero(z_i))
        preactivation = np.einsum("whmd,d,m->wh", module_weights, x_i, scaled_z)
        preactivation += np.einsum("whm,m->wh", module_biases, scaled_z)
        outputs.append(np.sum(np.maximum(preactivation, 0.0) * readouts, axis=-1))
    empirical_covariance = np.cov(np.stack(outputs), bias=True)
    relative_error = np.linalg.norm(feature_gram - empirical_covariance) / np.linalg.norm(
        empirical_covariance
    )
    assert relative_error < 0.12
