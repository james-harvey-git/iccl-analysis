"""Full sequence model: input/output embeddings around a stack of GDN blocks.

Configured by ``configs/model/gdn.yaml``; the ``backend`` field selects the
sequence-mixing backend per ``iccl.models.ops``. There is no positional
encoding — the recurrence is inherently positional.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from omegaconf import DictConfig

from iccl.models.blocks import GDNBlock
from iccl.models.ops import Backend, resolve_backend

NUM_TOKEN_TYPES = 4  # x / y / boundary / pad, matching iccl.data.sequences token codes


class TokenEmbedding(nn.Module):
    """Per-token-type affine embedding: ``h_t = W[type_t] @ x_t + b[type_t]``.

    x- and y-tokens get separate projections; content-free tokens (boundary,
    pad) are all-zero, so they embed to a purely learned per-type vector.
    """

    def __init__(self, d_in: int, d_model: int, num_types: int = NUM_TOKEN_TYPES) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_types, d_model, d_in))
        self.bias = nn.Parameter(torch.zeros(num_types, d_model))
        for w in self.weight:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq d_in"],
        token_type: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq d_model"]:
        token_type = token_type.long()
        projected = torch.einsum("btoi,bti->bto", self.weight[token_type], tokens)
        return projected + self.bias[token_type]


@dataclass
class ModelOutput:
    """``preds`` at every position; ``states``/``hidden`` are per-layer lists
    (fast-weight trajectories and post-block residual stream) filled only under
    ``capture=True``."""

    preds: Float[torch.Tensor, "batch seq d_out"]
    states: list[Float[torch.Tensor, "batch seq heads value_dim key_dim"]] | None = None
    hidden: list[Float[torch.Tensor, "batch seq d_model"]] | None = None


class GDNModel(nn.Module):
    """Stack of GDN blocks between the token embedding and a linear readout head."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ffw: int,
        expand_v: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        backend: Backend = "auto",
    ) -> None:
        super().__init__()
        self.backend: Backend = backend
        self.embed = TokenEmbedding(d_in, d_model)
        self.blocks = nn.ModuleList(
            GDNBlock(
                d_model,
                n_heads,
                d_ffw,
                expand_v=expand_v,
                use_short_conv=use_short_conv,
                conv_size=conv_size,
                use_gate=use_gate,
                allow_neg_eigval=allow_neg_eigval,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.head = nn.Linear(d_model, d_out, bias=False)

    def forward(
        self,
        tokens: Float[torch.Tensor, "batch seq d_in"],
        token_type: Int[torch.Tensor, "batch seq"],
        *,
        backend: Backend | None = None,
        capture: bool = False,
    ) -> ModelOutput:
        """Predict at every position. ``capture=True`` (reference backend only)
        additionally records per-layer fast-weight trajectories and the residual
        stream after each block."""
        backend = backend if backend is not None else self.backend
        if capture and resolve_backend(backend) != "reference":
            raise ValueError("capture=True requires the reference backend")

        h = self.embed(tokens, token_type)
        states: list[torch.Tensor] | None = [] if capture else None
        hidden: list[torch.Tensor] | None = [] if capture else None
        for block in self.blocks:
            h, block_states = block(h, backend=backend, return_states=capture)
            if states is not None and hidden is not None:
                assert block_states is not None
                states.append(block_states)
                hidden.append(h)
        preds = self.head(self.final_norm(h))
        return ModelOutput(preds=preds, states=states, hidden=hidden)


def model_from_config(cfg: DictConfig) -> GDNModel:
    """Build the model from the full hydra config (token dims from ``cfg.data``,
    architecture from ``cfg.model``)."""
    m = cfg.model
    return GDNModel(
        d_in=max(cfg.data.input_dim, cfg.data.output_dim),
        d_out=cfg.data.output_dim,
        d_model=m.d_model,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        d_ffw=m.d_ffw,
        expand_v=m.expand_v,
        use_short_conv=m.use_short_conv,
        conv_size=m.conv_size,
        use_gate=m.use_gate,
        allow_neg_eigval=m.allow_neg_eigval,
        norm_eps=m.norm_eps,
        backend=m.backend,
    )
