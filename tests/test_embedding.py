"""Numerical and checkpoint contracts for the per-type affine token embedding."""

import copy

import pytest
import torch

from iccl.models.model import TokenEmbedding


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1, 3, 5), (2, 19, 16, 32)])
def test_embedding_matches_per_type_affine_map_and_gradients(dtype, shape) -> None:
    torch.manual_seed(71)
    batch, length, inputs, width = shape
    module = TokenEmbedding(inputs, width).to(dtype)
    with torch.no_grad():
        module.bias.normal_()
    reference = copy.deepcopy(module)
    tokens = torch.randn(batch, length, inputs, dtype=dtype, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    kinds = torch.arange(batch * length).reshape(batch, length) % 4
    upstream = torch.randn(batch, length, width, dtype=dtype)
    expected = torch.stack(
        [
            torch.stack(
                [
                    reference.weight[kinds[b, t]] @ reference_tokens[b, t]
                    + reference.bias[kinds[b, t]]
                    for t in range(length)
                ]
            )
            for b in range(batch)
        ]
    )
    actual = module(tokens, kinds)
    expected.backward(upstream)
    actual.backward(upstream)
    tolerance = (
        {"rtol": 1e-10, "atol": 1e-11} if dtype == torch.float64 else {"rtol": 2e-5, "atol": 2e-5}
    )
    for observed, target in (
        (actual, expected),
        (tokens.grad, reference_tokens.grad),
        (module.weight.grad, reference.weight.grad),
        (module.bias.grad, reference.bias.grad),
    ):
        torch.testing.assert_close(observed, target, rtol=tolerance["rtol"], atol=tolerance["atol"])


def test_embedding_preserves_checkpoint_keys_and_content_free_tokens() -> None:
    module = TokenEmbedding(16, 256)
    state = {"weight": torch.randn(4, 256, 16), "bias": torch.randn(4, 256)}
    module.load_state_dict(state, strict=True)
    kinds = torch.tensor([[2, 3, 2, 3]])
    actual = module(torch.zeros(1, 4, 16), kinds)
    torch.testing.assert_close(actual, state["bias"][kinds], rtol=0, atol=0)
    actual.sum().backward()
    assert module.weight.grad is not None
    assert torch.count_nonzero(module.weight.grad) == 0
    assert set(module.state_dict()) == {"weight", "bias"}
    assert getattr(module.bias, "_no_weight_decay", False)
