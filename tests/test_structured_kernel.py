import math

import numpy as np
import torch

from iccl.analysis.structured_observer.kernel import sample_feature_bank


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
