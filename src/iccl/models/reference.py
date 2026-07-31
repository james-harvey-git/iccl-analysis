"""Pure-PyTorch reference implementation of the gated delta rule.

A sequential recurrence over a per-head fast-weight memory matrix S: each step
decays S by the gate, applies a rank-1 delta-rule write of the current key-value
pair scaled by the write strength, and reads out with the query. Inputs are the
raw projection outputs; the transforms fla fuses into its kernel (q/k L2
normalization, the Mamba2 decay gate, the beta sigmoid) are applied here
explicitly with matching semantics. Parity with ``chunk_gated_delta_rule`` is
asserted by ``tests/test_ops_parity.py``.

The recurrence accumulates in at least fp32 (half-precision inputs are
upcast; fp64 inputs stay fp64, which makes this backend usable as a
ground-truth anchor in the parity tests). The per-step memory states S_t are
the central object of the mechanistic interpretability analysis — this
implementation can return the full state trajectory, which the chunked
kernels cannot (they only materialize state at chunk boundaries).
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float

L2NORM_EPS = 1e-6  # matches fla.modules.l2norm


def l2norm(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """L2-normalize the last dim: x / sqrt(sum(x^2) + eps), matching fla's l2norm."""
    return x * torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + eps)


def gdn_gate(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """Log-space decay gate ``-exp(A_log) * softplus(a + dt_bias)`` (Mamba2
    parameterization; matches fla's ``naive_gdn_gate``)."""
    return -A_log.exp() * F.softplus(a + dt_bias)


def gated_delta_rule_reference(
    q: Float[torch.Tensor, "batch seq heads key_dim"],
    k: Float[torch.Tensor, "batch seq heads key_dim"],
    v: Float[torch.Tensor, "batch seq heads value_dim"],
    a: Float[torch.Tensor, "batch seq heads"],
    b: Float[torch.Tensor, "batch seq heads"],
    A_log: Float[torch.Tensor, " heads"],
    dt_bias: Float[torch.Tensor, " heads"],
    *,
    allow_neg_eigval: bool = False,
    return_states: bool = False,
) -> tuple[
    Float[torch.Tensor, "batch seq heads value_dim"],
    Float[torch.Tensor, "batch seq heads value_dim key_dim"] | None,
]:
    """Causal gated delta rule; see module docstring for conventions.

    Per step: ``S_t = alpha_t * S_{t-1} @ (I - beta_t k_t k_t^T) + beta_t v_t k_t^T``
    with decay ``alpha_t = exp(gdn_gate(a_t))`` and write strength
    ``beta_t = sigmoid(b_t)`` (doubled when ``allow_neg_eigval``), then readout
    ``o_t = S_t q_t / sqrt(key_dim)``. Returns the outputs and, when
    ``return_states``, the compute-dtype state trajectory in fla's
    ``state_v_first`` layout ``[batch, seq, heads, value_dim, key_dim]``.
    """
    in_dtype = v.dtype
    compute_dtype = torch.promote_types(in_dtype, torch.float32)
    q = l2norm(q.to(compute_dtype))
    k = l2norm(k.to(compute_dtype))
    v = v.to(compute_dtype)
    alpha = torch.exp(
        gdn_gate(a.to(compute_dtype), A_log.to(compute_dtype), dt_bias.to(compute_dtype))
    )
    beta = torch.sigmoid(b.to(compute_dtype))
    if allow_neg_eigval:
        beta = beta * 2.0

    batch, seq, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    scale = key_dim**-0.5
    state = k.new_zeros(batch, heads, value_dim, key_dim)
    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor] | None = [] if return_states else None
    for t in range(seq):
        state = alpha[:, t, :, None, None] * state
        k_t, v_t = k[:, t], v[:, t]
        retrieved = torch.einsum("bhvk,bhk->bhv", state, k_t)
        state = state + torch.einsum("bh,bhv,bhk->bhvk", beta[:, t], v_t - retrieved, k_t)
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]) * scale)
        if states is not None:
            states.append(state)

    output = torch.stack(outputs, dim=1).to(in_dtype)
    trajectory = torch.stack(states, dim=1) if states is not None else None
    return output, trajectory
