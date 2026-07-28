"""Pure-PyTorch reference implementation of the gated delta rule.

A sequential recurrence over a per-head fast-weight memory matrix S: each step
decays S by the gate, then applies a delta-rule write of the current key-value
pair scaled by the write strength, and reads out with the query. Exact tensor
conventions must match fla's ``chunk_gated_delta_rule``; parity is asserted by
``tests/test_ops_parity.py``.

The per-step memory states S_t are the central object of the mechanistic
interpretability analysis — this implementation can return them, which the
fused kernels cannot.
"""

import torch
from jaxtyping import Float


def gated_delta_rule_reference(
    q: Float[torch.Tensor, "batch seq heads key_dim"],
    k: Float[torch.Tensor, "batch seq heads key_dim"],
    v: Float[torch.Tensor, "batch seq heads value_dim"],
    beta: Float[torch.Tensor, "batch seq heads"],
    g: Float[torch.Tensor, "batch seq heads"],
) -> Float[torch.Tensor, "batch seq heads value_dim"]:
    raise NotImplementedError("implemented alongside the model port")
