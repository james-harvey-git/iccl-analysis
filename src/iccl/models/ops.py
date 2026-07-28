"""Backend dispatch for the gated-delta-rule sequence-mixing op.

The op has two interchangeable backends:

- ``"fla"``: fused Triton kernels from flash-linear-attention. CUDA-only (the
  ``fla`` package is a Linux-only dependency); used for training runs on the
  cluster.
- ``"reference"``: pure-PyTorch recurrence in ``iccl.models.reference``. Runs on
  any device (MPS/CPU/CUDA) and can expose per-step memory states, so it is also
  the backend used for mechanistic interpretability.

Everything outside this op (projections, gating parameterization, norms, block
structure) is backend-agnostic PyTorch. Numerical parity between the backends is
asserted by ``tests/test_ops_parity.py`` on CUDA machines.
"""

from typing import Literal

import torch
from jaxtyping import Float

Backend = Literal["auto", "fla", "reference"]


def resolve_backend(backend: Backend) -> Literal["fla", "reference"]:
    """Map "auto" to "fla" on CUDA machines and "reference" elsewhere."""
    if backend == "auto":
        return "fla" if torch.cuda.is_available() else "reference"
    return backend


def gated_delta_rule(
    q: Float[torch.Tensor, "batch seq heads key_dim"],
    k: Float[torch.Tensor, "batch seq heads key_dim"],
    v: Float[torch.Tensor, "batch seq heads value_dim"],
    beta: Float[torch.Tensor, "batch seq heads"],
    g: Float[torch.Tensor, "batch seq heads"],
    backend: Backend = "auto",
) -> Float[torch.Tensor, "batch seq heads value_dim"]:
    """Causal gated delta rule over a sequence.

    ``beta`` is the per-step write strength and ``g`` the log of the per-step
    decay gate, following fla's ``chunk_gated_delta_rule`` conventions.
    """
    match resolve_backend(backend):
        case "fla":
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore

            output, _final_state = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
            return output
        case "reference":
            from iccl.models.reference import gated_delta_rule_reference

            return gated_delta_rule_reference(q, k, v, beta, g)
