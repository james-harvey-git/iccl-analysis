"""Numerical parity between the fla and reference gated-delta-rule backends.

Runs only on CUDA machines (Isambard); this is what makes "prototype on the
reference backend, train on fla kernels" trustworthy.

Gradients are judged the way fla's own test suite judges them: by the RMS
error ratio ``rms(x - truth) / rms(truth)`` per tensor, not elementwise
bounds. The truth anchor is the reference backend run in fp64. Elementwise
comparison is meaningless for the summed per-head gate parameters
(``A_log``/``dt_bias``): each element sums ~1e5 signed per-position terms, so
cancellation shrinks the true value without shrinking the accumulated fp32
noise. The anchored criterion handles this by letting the fp32 reference's own
distance from truth set the scale — fla passes if it is within fla's published
ratio caps or comparably accurate to the fp32 reference; a semantic bug in
either backward shows up as fla sitting far from a truth the fp32 reference
agrees with.
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
MODEL_RTOL, MODEL_ATOL = 1e-2, 2e-3

# RMS error-ratio caps vs the fp64 anchor, following fla's own per-tensor
# tiers (o 0.005, dq/dk/dv 0.008, gate-path grads 0.02); the summed per-head
# parameters instead defer to the reference's own error scale.
RATIO_CAPS = {
    "out": 0.005,
    "q": 0.008,
    "k": 0.008,
    "v": 0.008,
    "a": 0.02,
    "b": 0.02,
    "A_log": 0.02,
    "dt_bias": 0.02,
}
REF_ERROR_FACTOR = 5.0


def make_inputs(
    batch: int = 2,
    seq: int = 192,
    heads: int = 4,
    key_dim: int = 64,
    value_dim: int = 128,
    dtype: torch.dtype = torch.float32,
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
    return {key: t.to(device="cuda", dtype=dtype).requires_grad_(True) for key, t in inputs.items()}


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


def forward_and_grads(backend: Backend, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    inputs = make_inputs(dtype=dtype)
    out = run(inputs, backend)
    torch.manual_seed(1)
    weights = torch.randn(out.shape, device="cuda", dtype=torch.float32)
    (out * weights.to(dtype)).sum().backward()
    results = {"out": out.detach()}
    for key, t in inputs.items():
        assert t.grad is not None, key
        results[key] = t.grad
    return results


def rms_ratio(x: torch.Tensor, truth: torch.Tensor) -> float:
    x, truth = x.double(), truth.double()
    return ((x - truth).square().mean().sqrt() / (truth.square().mean().sqrt() + 1e-12)).item()


def test_forward_and_backward_parity() -> None:
    truth = forward_and_grads("reference", torch.float64)
    ref32 = forward_and_grads("reference", torch.float32)
    fla32 = forward_and_grads("fla", torch.float32)

    torch.testing.assert_close(fla32["out"], ref32["out"], rtol=FWD_RTOL, atol=FWD_ATOL)
    for key, cap in RATIO_CAPS.items():
        err_fla = rms_ratio(fla32[key], truth[key])
        err_ref = rms_ratio(ref32[key], truth[key])
        allowed = max(cap, REF_ERROR_FACTOR * err_ref)
        assert err_fla <= allowed, (
            f"{key}: fla error ratio {err_fla:.5f} vs truth "
            f"(fp32 reference: {err_ref:.5f}, allowed {allowed:.5f})"
        )


def test_full_model_parity() -> None:
    torch.manual_seed(0)
    model = GDNModel(d_in=16, d_out=16, d_model=64, n_layers=2, n_heads=2, d_ffw=128).cuda()
    tokens = torch.randn(2, 130, 16, device="cuda")
    token_type = torch.randint(0, 3, (2, 130), device="cuda")
    with torch.no_grad():
        preds_ref = model(tokens, token_type, backend="reference").preds
        preds_fla = model(tokens, token_type, backend="fla").preds
    torch.testing.assert_close(preds_fla, preds_ref, rtol=MODEL_RTOL, atol=MODEL_ATOL)
