"""The length-group experiment preserves complete episodes and the pooled objective."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

from iccl.models.model import GDNModel
from iccl.training.trainer import masked_mse

install_variant = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_training.py"))[
    "install_variant"
]


@pytest.mark.parametrize(
    "variant",
    ["optimized", "group2", "group4", "bias", "all_types", "bias_group2", "all_types_group2"],
)
def test_variant_matches_model_predictions_loss_and_gradients(variant: str) -> None:
    torch.manual_seed(7)
    baseline = GDNModel(
        d_in=4, d_out=3, d_model=16, n_heads=2, n_layers=2, d_ffw=32, backend="reference"
    ).double()
    candidate = copy.deepcopy(baseline)
    install_variant(baseline, "baseline")
    install_variant(candidate, variant)
    tokens = torch.randn(4, 23, 4, dtype=torch.float64)
    kinds = torch.arange(23).repeat(4, 1) % 3
    real = torch.arange(23)[None, :] < torch.tensor([11, 23, 7, 17])[:, None]
    kinds[~real] = 3
    tokens[~real] = 0
    targets = torch.randn(4, 23, 3, dtype=torch.float64)
    mask = ((kinds == 0) & real).double()
    expected = baseline(tokens, kinds).preds
    actual = candidate(tokens, kinds).preds
    expected_loss = masked_mse(expected, targets, mask)
    actual_loss = masked_mse(actual, targets, mask)
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(actual[real], expected[real], rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-9, atol=1e-10)
    for (name, observed), (reference_name, reference) in zip(
        candidate.named_parameters(), baseline.named_parameters(), strict=True
    ):
        assert name == reference_name
        torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-8, atol=1e-10)
    assert set(candidate.state_dict()) == set(baseline.state_dict())
