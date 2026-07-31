import math

import torch
import torch.nn.functional as F

from iccl.models.ops import Backend, gated_delta_rule
from iccl.models.reference import gated_delta_rule_reference, l2norm

Inputs = dict[str, torch.Tensor]


def make_inputs(
    batch: int = 2,
    seq: int = 12,
    heads: int = 3,
    key_dim: int = 4,
    value_dim: int = 8,
    seed: int = 0,
) -> Inputs:
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    return {
        "q": randn(batch, seq, heads, key_dim),
        "k": randn(batch, seq, heads, key_dim),
        "v": randn(batch, seq, heads, value_dim),
        "a": randn(batch, seq, heads),
        "b": randn(batch, seq, heads),
        "A_log": randn(heads),
        "dt_bias": randn(heads),
    }


def run_reference(
    inputs: Inputs, *, allow_neg_eigval: bool = False, return_states: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return gated_delta_rule_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        allow_neg_eigval=allow_neg_eigval,
        return_states=return_states,
    )


def run_op(
    inputs: Inputs, *, backend: Backend = "reference", return_states: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return gated_delta_rule(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        backend=backend,
        return_states=return_states,
    )


def test_two_step_closed_form() -> None:
    # One batch/head, q and k pinned to the first basis vector at both steps, so
    # the recurrence collapses to scalars that can be worked by hand:
    #   o_1 = beta_1 * v_1 * scale
    #   o_2 = (alpha_2 * (1 - beta_2) * beta_1 * v_1 + beta_2 * v_2) * scale
    key_dim, value_dim = 3, 2
    e1 = torch.zeros(key_dim)
    e1[0] = 1.0
    q = k = torch.stack([2.0 * e1, 0.5 * e1]).view(1, 2, 1, key_dim)
    v = torch.tensor([[0.3, -1.2], [0.7, 0.4]]).view(1, 2, 1, value_dim)
    a = torch.tensor([0.4, -0.9]).view(1, 2, 1)
    b = torch.tensor([-0.2, 1.1]).view(1, 2, 1)
    A_log = torch.tensor([0.5])
    dt_bias = torch.tensor([-0.3])

    out, _ = gated_delta_rule_reference(q, k, v, a, b, A_log, dt_bias)

    scale = key_dim**-0.5
    alpha = torch.exp(-A_log.exp() * F.softplus(a + dt_bias)).squeeze()
    beta = torch.sigmoid(b).squeeze()
    v1, v2 = v[0, 0, 0], v[0, 1, 0]
    expected_1 = beta[0] * v1 * scale
    expected_2 = (alpha[1] * (1 - beta[1]) * beta[0] * v1 + beta[1] * v2) * scale
    torch.testing.assert_close(out[0, 0, 0], expected_1, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[0, 1, 0], expected_2, rtol=1e-5, atol=1e-5)


def test_states_reproduce_outputs() -> None:
    inputs = make_inputs()
    out, states = run_op(inputs, return_states=True)
    assert states is not None
    batch, seq, heads, key_dim = inputs["k"].shape
    assert states.shape == (batch, seq, heads, inputs["v"].shape[-1], key_dim)

    q_hat = l2norm(inputs["q"].float())
    recomputed = torch.einsum("bthvk,bthk->bthv", states, q_hat) * key_dim**-0.5
    torch.testing.assert_close(recomputed, out, rtol=1e-5, atol=1e-6)


def test_causality() -> None:
    inputs = make_inputs()
    perturbed = {key: t.clone() for key, t in inputs.items()}
    cut = 5
    for key in ("q", "k", "v", "a", "b"):
        perturbed[key][:, cut:] += 10.0

    out, _ = run_op(inputs)
    out_perturbed, _ = run_op(perturbed)
    torch.testing.assert_close(out[:, :cut], out_perturbed[:, :cut])
    assert not torch.allclose(out[:, cut:], out_perturbed[:, cut:])


def test_decay_shrinks_earlier_writes() -> None:
    # With orthogonal keys (no delta-rule interference), a stronger decay
    # channel must shrink the contribution of the first write to a later read.
    key_dim, value_dim, seq = 4, 4, 3
    q = torch.zeros(1, seq, 1, key_dim)
    k = torch.zeros(1, seq, 1, key_dim)
    for t in range(seq):
        k[0, t, 0, t] = 1.0
    q[0, -1, 0, 0] = 1.0  # read back the first write at the last step
    v = torch.randn(1, seq, 1, value_dim, generator=torch.Generator().manual_seed(0))
    a = torch.zeros(1, seq, 1)
    b = torch.full((1, seq, 1), 10.0)  # beta ~ 1
    dt_bias = torch.tensor([0.0])

    slow, _ = gated_delta_rule_reference(q, k, v, a, b, torch.tensor([-4.0]), dt_bias)
    fast, _ = gated_delta_rule_reference(q, k, v, a, b, torch.tensor([2.0]), dt_bias)
    assert fast[0, -1, 0].norm() < slow[0, -1, 0].norm()


def test_allow_neg_eigval_doubles_beta() -> None:
    # b pushed negative so the doubled beta stays below 1 and can be reproduced
    # through a plain sigmoid: sigmoid(b') = 2 * sigmoid(b).
    inputs = dict(make_inputs(), b=make_inputs()["b"].abs().neg() - 1.0)
    doubled, _ = run_reference(inputs, allow_neg_eigval=True)
    two_sigma = 2.0 * torch.sigmoid(inputs["b"])
    assert (two_sigma < 1.0).all()
    manual, _ = run_reference(dict(inputs, b=torch.logit(two_sigma)))
    torch.testing.assert_close(doubled, manual, rtol=1e-4, atol=1e-5)


def test_output_dtype_follows_input() -> None:
    inputs = {key: t.to(torch.bfloat16) if t.ndim > 1 else t for key, t in make_inputs().items()}
    out, states = run_reference(inputs, return_states=True)
    assert out.dtype == torch.bfloat16
    assert states is not None and states.dtype == torch.float32


def test_fp64_inputs_compute_in_fp64() -> None:
    # fp64 must survive end to end: the parity tests use this backend in fp64
    # as the ground-truth anchor.
    inputs32 = make_inputs()
    inputs64 = {key: t.double() for key, t in inputs32.items()}
    out64, states64 = run_reference(inputs64, return_states=True)
    assert out64.dtype == torch.float64
    assert states64 is not None and states64.dtype == torch.float64
    out32, _ = run_reference(inputs32)
    torch.testing.assert_close(out32, out64.float(), rtol=1e-5, atol=1e-6)


def test_l2norm_matches_fla_semantics() -> None:
    x = torch.randn(5, 7, generator=torch.Generator().manual_seed(0))
    y = l2norm(x)
    expected = x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(y, expected)
    norms = y.norm(dim=-1)
    assert (norms <= 1.0 + 1e-6).all()
    assert math.isclose(norms.max().item(), 1.0, rel_tol=1e-3)
