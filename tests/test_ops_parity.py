"""Numerical parity between the fla and reference gated-delta-rule backends.

Runs only on CUDA machines (Isambard); this is what makes "prototype on the
reference backend, train on fla kernels" trustworthy. Both backends run fp32,
but the chunked kernel and the sequential scan order their accumulations
differently, so results diverge by genuine floating-point noise: ~1e-4
absolute at the op level over a couple hundred steps, more once the backward
recomputation and layer stacking compound it. The tolerance tiers reflect
that; a semantic mismatch (wrong gate, wrong state layout) would exceed any
of them by orders of magnitude on most elements.
"""

import pytest
import torch

from iccl.models.model import GDNModel
from iccl.models.ops import Backend, gated_delta_rule

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

FWD_RTOL, FWD_ATOL = 5e-3, 5e-4
GRAD_RTOL, GRAD_ATOL = 5e-3, 2e-3
MODEL_RTOL, MODEL_ATOL = 1e-2, 2e-3


def make_inputs(
    batch: int = 2, seq: int = 192, heads: int = 4, key_dim: int = 64, value_dim: int = 128
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    inputs = {
        "q": torch.randn(batch, seq, heads, key_dim),
        "k": torch.randn(batch, seq, heads, key_dim),
        "v": torch.randn(batch, seq, heads, value_dim),
        "a": torch.randn(batch, seq, heads),
        "b": torch.randn(batch, seq, heads),
        "A_log": torch.log(torch.empty(heads).uniform_(1.0, 8.0)),
        "dt_bias": torch.randn(heads) * 0.1,
    }
    return {key: t.cuda().requires_grad_(True) for key, t in inputs.items()}


def run(inputs: dict[str, torch.Tensor], backend: Backend) -> torch.Tensor:
    out, _ = gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        backend=backend,
    )
    return out


def test_forward_and_backward_parity() -> None:
    inputs_ref = make_inputs()
    inputs_fla = {key: t.detach().clone().requires_grad_(True) for key, t in inputs_ref.items()}

    out_ref = run(inputs_ref, "reference")
    out_fla = run(inputs_fla, "fla")
    torch.testing.assert_close(out_fla, out_ref, rtol=FWD_RTOL, atol=FWD_ATOL)

    weights = torch.randn_like(out_ref)
    (out_ref * weights).sum().backward()
    (out_fla * weights).sum().backward()
    for key in inputs_ref:
        grad_ref, grad_fla = inputs_ref[key].grad, inputs_fla[key].grad
        assert grad_ref is not None and grad_fla is not None, key
        torch.testing.assert_close(grad_fla, grad_ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_full_model_parity() -> None:
    torch.manual_seed(0)
    model = GDNModel(d_in=16, d_out=16, d_model=64, n_layers=2, n_heads=2, d_ffw=128).cuda()
    tokens = torch.randn(2, 130, 16, device="cuda")
    token_type = torch.randint(0, 3, (2, 130), device="cuda")
    with torch.no_grad():
        preds_ref = model(tokens, token_type, backend="reference").preds
        preds_fla = model(tokens, token_type, backend="fla").preds
    torch.testing.assert_close(preds_fla, preds_ref, rtol=MODEL_RTOL, atol=MODEL_ATOL)
