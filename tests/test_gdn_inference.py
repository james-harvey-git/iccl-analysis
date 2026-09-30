"""A query branch must match replaying its prefix without teaching other branches."""

import pytest
import torch

from iccl.analysis.gdn_inference import continue_gdn
from iccl.models.model import GDNModel


@pytest.mark.parametrize(
    "short_conv,conv_size,gate,negative",
    [(True, 4, True, False), (True, 1, False, True), (False, 4, True, True)],
)
@torch.inference_mode()
def test_segmented_forward_and_independent_queries_match_replay(
    short_conv: bool, conv_size: int, gate: bool, negative: bool
) -> None:
    torch.manual_seed(12)
    model = GDNModel(
        d_in=4,
        d_out=4,
        d_model=8,
        n_layers=2,
        n_heads=2,
        d_ffw=16,
        use_short_conv=short_conv,
        conv_size=conv_size,
        use_gate=gate,
        allow_neg_eigval=negative,
        backend="reference",
    ).eval()
    tokens, types = torch.randn(2, 15, 4), torch.arange(15).repeat(2, 1) % 3
    full = model(tokens, types, capture_final=True)
    cache, outputs = None, []
    for start, stop in ((0, 1), (1, 3), (3, 14), (14, 15)):
        output, cache = continue_gdn(
            model, tokens[:, start:stop], types[:, start:stop], cache, backend="reference"
        )
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs, dim=1), full.preds, rtol=2e-5, atol=2e-6)
    assert cache is not None and full.final_states is not None
    for actual, expected in zip(cache.states, full.final_states, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    original = [value.clone() for value in cache.states]
    original_tails = [[value.clone() for value in tail] for tail in cache.convolutions]
    indices = torch.tensor([1, 0, 1])
    queries, query_types = torch.randn(3, 1, 4), torch.zeros(3, 1, dtype=torch.long)
    branched, _ = continue_gdn(
        model, queries, query_types, cache.select(indices), backend="reference"
    )
    expected = model(
        torch.cat((tokens[indices], queries), dim=1),
        torch.cat((types[indices], query_types), dim=1),
    ).preds[:, -1:]
    torch.testing.assert_close(branched, expected, rtol=2e-5, atol=2e-6)
    for actual, before in zip(cache.states, original, strict=True):
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
    for actual, before in zip(cache.convolutions, original_tails, strict=True):
        for a, b in zip(actual, before, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
