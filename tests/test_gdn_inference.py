"""A query branch must match replaying its prefix without teaching other branches."""

from typing import cast

import pytest
import torch

from iccl.analysis.gdn_inference import continue_gdn
from iccl.models.blocks import GDNBlock
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


@torch.inference_mode()
def test_capture_sites_match_the_ordinary_forward_path() -> None:
    torch.manual_seed(9)
    model = GDNModel(d_in=4, d_out=3, d_model=8, n_layers=2, n_heads=2, d_ffw=16).eval()
    tokens = torch.randn(3, 19, 4)
    types = torch.arange(19).repeat(3, 1) % 3
    expected = {name: [] for name in ("residual_after_mixer", "mlp_hidden", "head_read")}
    hooks = []
    for block in cast(list[GDNBlock], list(model.blocks)):
        for module, name in (
            (block.mlp_norm, "residual_after_mixer"),
            (block.mlp.down_proj, "mlp_hidden"),
            (block.mixer.o_norm, "head_read"),
        ):

            def record(module, args, key=name):
                expected[key].append(args[0][:, -1].clone())

            hooks.append(module.register_forward_pre_hook(record))
    output = model(tokens, types, backend="reference", capture=True)
    for hook in hooks:
        hook.remove()
    _, cache = continue_gdn(model, tokens[:, :13], types[:, :13], backend="reference")
    captured = {}
    prediction, _ = continue_gdn(
        model, tokens[:, 13:], types[:, 13:], cache, backend="reference", capture=captured
    )
    torch.testing.assert_close(prediction, output.preds[:, 13:], rtol=2e-5, atol=2e-6)
    for name, values in expected.items():
        torch.testing.assert_close(captured[name], torch.stack(values, dim=1), rtol=2e-5, atol=2e-6)
    assert output.hidden is not None and output.states is not None
    torch.testing.assert_close(
        captured["residual_after_mlp"],
        torch.stack([h[:, -1] for h in output.hidden], dim=1),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        captured["state"],
        torch.stack([s[:, -1] for s in output.states], dim=1),
        rtol=2e-5,
        atol=2e-6,
    )
