"""Backend dispatch for the gated-delta-rule sequence-mixing op.

The op has two interchangeable backends:

- ``"fla"``: fused Triton kernels from flash-linear-attention. CUDA-only (the
  ``fla`` package is a Linux-only dependency); used for training runs on the
  cluster.
- ``"reference"``: pure-PyTorch recurrence in ``iccl.models.reference``. Runs on
  any device (MPS/CPU/CUDA) and can expose per-step memory states, so it is also
  the backend used for mechanistic interpretability.

The op takes *raw* projection outputs, mirroring how fla's layer calls its
kernel: the q/k L2 normalization, the Mamba2 decay gate
``-exp(A_log) * softplus(a + dt_bias)``, and the beta sigmoid are applied
inside each backend (fla via its ``*_in_kernel`` flags, the reference
explicitly), so both see identical semantics. Everything outside this op
(projections, short conv, norms, block structure) is backend-agnostic PyTorch.
Numerical parity between the backends is asserted by
``tests/test_ops_parity.py`` on CUDA machines.
"""

from typing import Literal

import torch
from jaxtyping import Float

Backend = Literal["auto", "fla", "reference"]


def resolve_backend(
    backend: Backend, device: torch.device | None = None
) -> Literal["fla", "reference"]:
    """Resolve automatic dispatch using the execution device when supplied."""
    if backend == "auto":
        cuda = device.type == "cuda" if device is not None else torch.cuda.is_available()
        return "fla" if cuda else "reference"
    if backend not in ("fla", "reference"):
        raise ValueError(f"unknown gated delta rule backend: {backend}")
    return backend


def gated_delta_rule(
    q: Float[torch.Tensor, "batch seq heads key_dim"],
    k: Float[torch.Tensor, "batch seq heads key_dim"],
    v: Float[torch.Tensor, "batch seq heads value_dim"],
    a: Float[torch.Tensor, "batch seq heads"],
    b: Float[torch.Tensor, "batch seq heads"],
    A_log: Float[torch.Tensor, " heads"],
    dt_bias: Float[torch.Tensor, " heads"],
    *,
    allow_neg_eigval: bool = False,
    backend: Backend = "auto",
    return_states: bool = False,
    return_final_state: bool = False,
) -> tuple[
    Float[torch.Tensor, "batch seq heads value_dim"],
    Float[torch.Tensor, "batch ... heads value_dim key_dim"] | None,
]:
    """Causal gated delta rule over a sequence.

    ``a`` and ``b`` are the raw ``a_proj``/``b_proj`` outputs; ``A_log`` and
    ``dt_bias`` the per-head gate parameters. Returns the per-position readouts
    and, when ``return_states`` (reference backend only), the per-step state
    trajectory. ``return_final_state`` instead returns the terminal matrix on
    either backend, in ``[batch, heads, value_dim, key_dim]`` order.
    """
    if return_states and return_final_state:
        raise ValueError("request either a state trajectory or a final state, not both")
    match resolve_backend(backend, q.device):
        case "fla":
            if return_states:
                raise NotImplementedError(
                    "per-step states require the reference backend; "
                    "chunked kernels only materialize state at chunk boundaries"
                )
            if q.device.type != "cuda":
                raise ValueError("the fla backend requires CUDA tensors")
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore

            output, final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=a,
                beta=b,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=allow_neg_eigval,
                state_v_first=True,
                output_final_state=return_final_state,
            )
            return output, final_state if return_final_state else None
        case "reference":
            from iccl.models.reference import gated_delta_rule_reference

            return gated_delta_rule_reference(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                allow_neg_eigval=allow_neg_eigval,
                return_states=return_states,
                return_final_state=return_final_state,
            )
